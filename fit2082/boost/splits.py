"""Split selection: how a new hash's partitions are chosen.

A splitter proposes `num_bits` (feature, midpoint) pairs; each pair contributes
one bit to the hash code. Swap in a different splitter to experiment with
alternative partitioning schemes (quantile splits, multiple candidates per bit,
feature subsampling, ...) without touching the rest of the model.
"""

from typing import Protocol

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # CPU-only installs; pairing then stays on the host
    HAVE_TRITON = False

# == protocol ==================================================================


class Splitter(Protocol):
    """Proposes the split points for one new hash."""

    def propose(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        probabilities: torch.Tensor,
        gradient: torch.Tensor,
        hessian: torch.Tensor,
        num_bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (feature_indices (num_bits,) int64, midpoints (num_bits,) float32).

        `gradient` and `hessian` are passed even though the default splitter
        ignores them: a gain-based splitter needs them, and recomputing them
        would duplicate the objective.
        """
        ...


# == pairs of hard, differently-classified examples ============================


def _pair(order: np.ndarray, classes: np.ndarray, num_pairs: int) -> np.ndarray:
    """Greedily form `num_pairs` pairs of differently-classified examples.

    Walks `order` (hardest example first, `classes` is already permuted to match)
    and fills pair slots: an empty slot takes the point as its anchor, an anchored
    slot is completed by the first point of a different class.

    The scan starts at `num_found`, which is only safe because of this invariant:
    a point that fails to complete slot j only ever anchors a *later* slot, so all
    pending anchors always share a class. A point therefore completes the lowest
    pending slot, slots complete in order, and no slot is ever orphaned.
    """

    num_examples = order.shape[0]

    pairs = np.zeros((num_pairs, 2), dtype=np.int64)
    anchor_class = np.zeros(num_pairs, dtype=np.int64)
    status = np.zeros(num_pairs, dtype=np.int64)

    num_found = 0
    pointer_data = 0

    while num_found < num_pairs:
        index = order[pointer_data]
        label = classes[pointer_data]

        pointer_pair = num_found
        processed = False

        while not processed and pointer_pair < num_pairs:
            if status[pointer_pair] == 0:
                status[pointer_pair] = 1
                anchor_class[pointer_pair] = label
                pairs[pointer_pair, 0] = index
                processed = True
            elif status[pointer_pair] == 1 and anchor_class[pointer_pair] != label:
                status[pointer_pair] = 2
                pairs[pointer_pair, 1] = index
                num_found += 1
                processed = True

            pointer_pair += 1

        pointer_data += 1
        if pointer_data >= num_examples:
            pointer_data = 0

    return pairs


# The same greedy pairing as `_pair`, as a one-thread GPU kernel. It is still
# a sequential walk, but running it where the order already lives spares the
# host sync the NumPy loop needs: with that sync every round made the host wait
# for the GPU to drain before it could queue the next round's work, which on
# small batches is most of a round. The walk is `_pair`'s state machine with
# the slot scan collapsed. Because pending anchors always share a class (see
# `_pair`), an example either completes the oldest pending slot (any other
# class), anchors the next free slot (the pending class, or none pending), or
# is skipped (the pending class, with every slot taken).
if HAVE_TRITON:

    @triton.jit(do_not_specialize=["n", "max_steps"])
    def _pair_kernel(order_ptr, classes_ptr, pairs_ptr, n, num_pairs, max_steps):

        found = 0
        pending = 0
        pointer = 0
        steps = 0
        pending_class = tl.load(classes_ptr)  # replaced before it is read

        # max_steps only bounds a batch of one class, which never completes
        while (found < num_pairs) & (steps < max_steps):
            index = tl.load(order_ptr + pointer)
            label = tl.load(classes_ptr + pointer)

            if pending == 0:
                tl.store(pairs_ptr + 2 * found, index)
                pending = 1
                pending_class = label
            elif label != pending_class:
                tl.store(pairs_ptr + 2 * found + 1, index)
                found += 1
                pending -= 1
            elif found + pending < num_pairs:
                tl.store(pairs_ptr + 2 * (found + pending), index)
                pending += 1

            pointer += 1
            if pointer >= n:
                pointer = 0
            steps += 1


def pair_on_device(
    order: torch.Tensor, classes: torch.Tensor, num_pairs: int
) -> torch.Tensor:
    """`_pair` on the GPU: (num_pairs, 2) example indices, without a host sync.

    `order` and `classes` are (n,) int64 CUDA tensors, `classes` already
    permuted to match `order`. A batch needs two classes, as for `_pair`: with
    one, no pair can complete, and where `_pair` would loop forever this stops
    after a bounded walk and leaves the unfinished slots pointing at example 0.
    """

    n = order.shape[0]
    pairs = torch.zeros((num_pairs, 2), dtype=torch.int64, device=order.device)

    # each completion takes at most one full pass, and one more to anchor
    max_steps = (num_pairs + 1) * (n + 1)

    _pair_kernel[(1,)](
        order.contiguous(), classes.contiguous(), pairs, n, num_pairs, max_steps
    )

    return pairs


class HardPairSplitter:
    """The original scheme.

    Ranks examples by cross entropy (hardest first), pairs up examples of
    differing classes, then for each pair picks a random feature and splits at
    the midpoint of the pair's two values for that feature.

    With `sample=True` the ranking is itself drawn at random: examples are
    ordered as successive draws without replacement, each with probability
    proportional to its cross entropy. The strict ranking pairs the same dozen
    or so hardest examples every time a batch comes round, and once the batch
    is memorised those are its persistent outliers; sampling keeps the pairs on
    hard examples but spreads them across the hard end of the batch. Screened
    as a small win on Pedestrian and neutral on four other datasets -- see the
    README.
    """

    def __init__(
        self, generator: torch.Generator | None = None, sample: bool = False
    ) -> None:

        self.generator = generator
        self.sample = sample

    def order(self, probabilities: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """-> (n,) example indices, in the order pairs are formed from them."""

        tiny = torch.finfo(probabilities.dtype).tiny

        cross_entropy = (
            -probabilities.gather(1, Y[:, None]).squeeze(1).clamp_min(tiny).log()
        )

        if not self.sample:
            return torch.argsort(cross_entropy, descending=True)

        # Gumbel-top-k: sorting log(w) plus Gumbel noise orders the examples
        # exactly as successive draws without replacement with probability
        # w / sum(w). An example the model already fits has cross entropy ~0,
        # so a log weight near log(tiny), and is in effect never drawn.
        uniform = torch.rand(
            cross_entropy.shape,
            device=cross_entropy.device,
            generator=self.generator,
        ).clamp_min(tiny)

        gumbel = -(-uniform.log()).log()

        return torch.argsort(
            cross_entropy.clamp_min(tiny).log() + gumbel, descending=True
        )

    def propose(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        probabilities: torch.Tensor,
        gradient: torch.Tensor,
        hessian: torch.Tensor,
        num_bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        order = self.order(probabilities, Y)
        classes = Y[order].to(torch.int64)

        if X.is_cuda and HAVE_TRITON:
            # on the device, so that the host never waits for the GPU here
            pair_indices = pair_on_device(order, classes, num_bits)
        else:
            pairs = _pair(
                order=order.cpu().numpy(),
                classes=classes.cpu().numpy(),
                num_pairs=num_bits,
            )
            pair_indices = torch.as_tensor(pairs, device=X.device)

        feature_indices = torch.randint(
            0,
            X.shape[1],
            (num_bits,),
            device=X.device,
            generator=self.generator,
            dtype=torch.int64,
        )

        a = X[pair_indices[:, 0], feature_indices]
        b = X[pair_indices[:, 1], feature_indices]

        return feature_indices, (a + b) / 2

    def __repr__(self) -> str:

        return f"HardPairSplitter(sample={self.sample})"
