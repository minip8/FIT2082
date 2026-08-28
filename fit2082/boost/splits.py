"""Split selection: how a new hash's partitions are chosen.

A splitter proposes `num_bits` (feature, midpoint) pairs; each pair contributes
one bit to the hash code. Swap in a different splitter to experiment with
alternative partitioning schemes (quantile splits, multiple candidates per bit,
feature subsampling, ...) without touching the rest of the model.
"""

from typing import Protocol

import numpy as np
import torch

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


class HardPairSplitter:
    """The original scheme.

    Ranks examples by cross entropy (hardest first), pairs up examples of
    differing classes, then for each pair picks a random feature and splits at
    the midpoint of the pair's two values for that feature.
    """

    def __init__(self, generator: torch.Generator | None = None) -> None:

        self.generator = generator

    def propose(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        probabilities: torch.Tensor,
        gradient: torch.Tensor,
        hessian: torch.Tensor,
        num_bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        tiny = torch.finfo(probabilities.dtype).tiny

        cross_entropy = (
            -probabilities.gather(1, Y[:, None]).squeeze(1).clamp_min(tiny).log()
        )

        order = torch.argsort(cross_entropy, descending=True)

        # the pairing loop is inherently sequential, but runs only a couple of
        # dozen iterations; this is the one host sync per batch and costs <1%
        pairs = _pair(
            order=order.cpu().numpy(),
            classes=Y[order].cpu().numpy(),
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
