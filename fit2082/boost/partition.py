"""Partition families: how an example's bits are computed, and stored.

The bits of a hash are a *family* of predicates (axis-aligned thresholds,
oblique comparisons, ...) and, separately, a *selection strategy* for choosing
particular members of that family. Those are orthogonal, so a `Partitioner`
owns the family -- the parameter storage and the encoder -- and delegates
selection to a `Splitter` (`fit2082.boost.splits`) where that makes sense.

Splitting them this way is what makes non-axis-aligned partitions expressible:
the proposer and the encoder have to agree on the parameterisation, so a seam
that only makes the proposer pluggable cannot change the family.
"""

from typing import Protocol

import torch

from fit2082.boost.splits import HardPairSplitter, Splitter

# == protocol ==================================================================


class Partitioner(Protocol):
    """Owns the split parameters for every round, and turns X into hash codes."""

    num_bits: int

    def propose(
        self,
        index: int,
        X: torch.Tensor,
        Y: torch.Tensor,
        probabilities: torch.Tensor,
        gradient: torch.Tensor,
        hessian: torch.Tensor,
    ) -> None:
        """Choose and store the split parameters for hash `index`."""
        ...

    def encode(self, Xt: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        """(p, n) feature-major X -> (hi - lo, n) int32 codes for hashes [lo, hi)."""
        ...


# == axis-aligned ==============================================================


def encode_axis_aligned(
    Xt: torch.Tensor,
    feature_indices: torch.Tensor,
    midpoints: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """(p, n) X + (c, B) splits -> (c, n) int32 codes.

    Kept at module scope so it can be handed to `torch.compile`, which fuses the
    gather, the comparison and the bit-packing into a single kernel.
    """

    bits = Xt[feature_indices] <= midpoints[:, :, None]  # (c, B, n)

    return (bits.to(torch.int32) * weights[None, :, None]).sum(1)


class AxisAlignedPartitioner:
    """`x[feature] <= midpoint` per bit -- the original scheme."""

    def __init__(
        self,
        num_bits: int,
        max_num_hashes: int,
        device: torch.device,
        splitter: Splitter | None = None,
        generator: torch.Generator | None = None,
        encode_chunk: int = 256,
        compile: bool = False,
    ) -> None:

        self.num_bits = num_bits
        self.device = device
        self.encode_chunk = encode_chunk

        self.splitter = splitter or HardPairSplitter(generator=generator)

        self.feature_indices = torch.zeros(
            (max_num_hashes, num_bits), dtype=torch.int64, device=device
        )
        self.midpoints = torch.zeros(
            (max_num_hashes, num_bits), dtype=torch.float32, device=device
        )

        self.weights = (2 ** torch.arange(num_bits, device=device)).to(torch.int32)

        self._encode = (
            torch.compile(encode_axis_aligned, dynamic=True)
            if compile
            else encode_axis_aligned
        )

    def propose(self, index, X, Y, probabilities, gradient, hessian) -> None:

        feature_indices, midpoints = self.splitter.propose(
            X, Y, probabilities, gradient, hessian, self.num_bits
        )

        self.feature_indices[index] = feature_indices
        self.midpoints[index] = midpoints

    def encode(self, Xt: torch.Tensor, lo: int, hi: int) -> torch.Tensor:

        codes = torch.empty(
            (hi - lo, Xt.shape[1]), dtype=torch.int32, device=self.device
        )

        for a in range(lo, hi, self.encode_chunk):
            b = min(a + self.encode_chunk, hi)

            codes[a - lo : b - lo] = self._encode(
                Xt, self.feature_indices[a:b], self.midpoints[a:b], self.weights
            )

        return codes


# == oblique ===================================================================


def encode_oblique(
    Xt: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    midpoints: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """`x[left] - x[right] <= midpoint` per bit -> (c, n) int32 codes."""

    bits = (Xt[left] - Xt[right]) <= midpoints[:, :, None]  # (c, B, n)

    return (bits.to(torch.int32) * weights[None, :, None]).sum(1)


class ObliquePartitioner:
    """`x[i] - x[j] <= threshold` per bit.

    For time series features -- interval quantiles, spectra -- a difference
    between two features is a shape comparison, which an axis-aligned threshold
    on either feature alone cannot express. Thresholds are taken from the
    midpoint of a random pair of hard examples, as in the axis-aligned scheme.
    """

    def __init__(
        self,
        num_bits: int,
        max_num_hashes: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        hard_pool: int = 64,
        encode_chunk: int = 256,
        compile: bool = False,
    ) -> None:

        self.num_bits = num_bits
        self.device = device
        self.generator = generator
        self.hard_pool = hard_pool
        self.encode_chunk = encode_chunk

        shape = (max_num_hashes, num_bits)

        self.left = torch.zeros(shape, dtype=torch.int64, device=device)
        self.right = torch.zeros(shape, dtype=torch.int64, device=device)
        self.midpoints = torch.zeros(shape, dtype=torch.float32, device=device)

        self.weights = (2 ** torch.arange(num_bits, device=device)).to(torch.int32)

        self._encode = (
            torch.compile(encode_oblique, dynamic=True) if compile else encode_oblique
        )

    def propose(self, index, X, Y, probabilities, gradient, hessian) -> None:

        n, p = X.shape
        b = self.num_bits

        tiny = torch.finfo(probabilities.dtype).tiny
        cross_entropy = (
            -probabilities.gather(1, Y[:, None]).squeeze(1).clamp_min(tiny).log()
        )

        pool = torch.argsort(cross_entropy, descending=True)[: min(self.hard_pool, n)]

        def draw(high, size):
            return torch.randint(
                0, high, (size,), device=X.device, generator=self.generator
            )

        left = draw(p, b)
        right = draw(p, b)

        a = pool[draw(pool.shape[0], b)]
        c = pool[draw(pool.shape[0], b)]

        da = X[a, left] - X[a, right]
        dc = X[c, left] - X[c, right]

        self.left[index] = left
        self.right[index] = right
        self.midpoints[index] = (da + dc) / 2

    def encode(self, Xt: torch.Tensor, lo: int, hi: int) -> torch.Tensor:

        codes = torch.empty(
            (hi - lo, Xt.shape[1]), dtype=torch.int32, device=self.device
        )

        for a in range(lo, hi, self.encode_chunk):
            b = min(a + self.encode_chunk, hi)

            codes[a - lo : b - lo] = self._encode(
                Xt, self.left[a:b], self.right[a:b], self.midpoints[a:b], self.weights
            )

        return codes
