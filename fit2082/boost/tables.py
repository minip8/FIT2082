"""The performance core: hash encoding, table lookup, and bucket accumulation.

All state lives on one device. Every operation is batched over boosting rounds
and chunked so peak memory stays bounded.

Layout notes (these are load-bearing -- see the comments at each use site):

* Hash codes are stored **round-major** `(rounds, n)`; they come from a
  `Partitioner` (`fit2082.boost.partition`), which owns the split parameters.
* `stats` fuses the residual numerator and hessian denominator into one tensor
  so a single scatter updates both.
* Prediction indexes a re-based table slice and so uses **chunk-local** offsets;
  accumulation scatters into the full buffer and so uses **global** offsets.
"""

import torch
import torch.nn.functional as F

# == tables ====================================================================


class HashTables:
    """Preallocated per-round split points and bucket statistics."""

    def __init__(
        self,
        num_classes: int,
        num_bits: int,
        max_num_hashes: int,
        lr: float,
        device: torch.device,
        hessian_eps: float = 1e-6,
        neighbour_shrinkage: float = 0.0,
        round_chunk: int | None = None,
        chunk_budget_bytes: int = 128 << 20,
    ) -> None:

        self.num_classes = num_classes
        self.num_bits = num_bits
        self.hash_size = 2**num_bits
        self.max_num_hashes = max_num_hashes
        self.lr = lr
        self.device = device
        self.hessian_eps = hessian_eps
        self.neighbour_shrinkage = neighbour_shrinkage

        self._round_chunk = round_chunk
        self._refresh_chunk = 128
        self._chunk_budget_bytes = chunk_budget_bytes

        k = num_classes
        m = max_num_hashes
        s = self.hash_size

        # stats[..., :k] is the accumulated *negated* residual (the numerator);
        # stats[..., k:] is the accumulated hessian (the denominator). Keeping
        # them adjacent lets one index_add_ update both, and keeping numerator
        # and denominator separately means the leaf value is always exact.
        self.stats = torch.zeros((m, s, 2 * k), dtype=torch.float32, device=device)
        self.logits = torch.zeros((m, s, k), dtype=torch.float32, device=device)

        self.offsets = torch.arange(m, device=device, dtype=torch.int64) * s

        # Bucket i's Hamming neighbours: the same partition with one bit
        # flipped. Only meaningful because the partition is a flat hypercube --
        # a tree has no equivalent "one split coarser" sibling.
        bucket = torch.arange(s, device=device)
        self.neighbours = torch.stack([bucket ^ (1 << b) for b in range(num_bits)])

    # -- chunking --------------------------------------------------------------

    def round_chunk(self, num_examples: int) -> int:
        """How many rounds to process per kernel.

        Bounds the size of the tiled scatter source, which is the largest
        intermediate. Too small and per-launch overhead dominates; too large and
        the allocator starts thrashing.
        """

        if self._round_chunk is not None:
            return self._round_chunk

        per_round = num_examples * 2 * self.num_classes * 4

        return max(8, min(256, self._chunk_budget_bytes // max(1, per_round)))

    # -- predict ---------------------------------------------------------------

    def predict_from_codes(self, codes: torch.Tensor, num_rounds: int) -> torch.Tensor:
        """Sum one table row per round -> (n, k) logits."""

        n = codes.shape[1]
        chunk = self.round_chunk(n)

        out = torch.zeros(
            (n, self.num_classes), dtype=torch.float32, device=self.device
        )

        for a in range(0, num_rounds, chunk):
            b = min(a + chunk, num_rounds)

            # chunk-local offsets: the table slice below is re-based to 0
            flat = (codes[a:b].t().to(torch.int64) + self.offsets[: b - a]).contiguous()

            out += F.embedding_bag(
                flat, self.logits[a:b].reshape(-1, self.num_classes), mode="sum"
            )

        return out

    # -- accumulate ------------------------------------------------------------

    def accumulate(
        self, codes: torch.Tensor, num_rounds: int, updates: torch.Tensor
    ) -> None:
        """Scatter-add `updates` (n, 2k) into each round's bucket for each example."""

        n = updates.shape[0]
        chunk = self.round_chunk(n)

        # Every chunk scatters the *same* per-example values, just to different
        # buckets, so the tiled source is built once and reused. Round-major
        # codes make row j*n + i correspond to (round a+j, example i), which is
        # exactly `updates.repeat(chunk, 1)` -- and lets a short tail chunk take
        # a prefix slice. (With example-major codes that slice is silently wrong.)
        tiled = updates.repeat(chunk, 1)

        stats = self.stats.view(-1, 2 * self.num_classes)

        for a in range(0, num_rounds, chunk):
            b = min(a + chunk, num_rounds)

            # global offsets: scattering into the full buffer
            flat = (codes[a:b].to(torch.int64) + self.offsets[a:b, None]).reshape(-1)

            stats.index_add_(0, flat, tiled[: (b - a) * n])

    # -- refresh ---------------------------------------------------------------

    def refresh_logits(self, num_rounds: int) -> None:
        """Recompute leaf values from the numerator/denominator buffers.

        Every bucket is refreshed, not just the ones this batch touched: an
        untouched bucket's numerator and denominator are unchanged, so it lands
        on the identical value. Skipping them would be an optimisation, not a
        semantic difference -- and the dense form is far faster here.

        With `neighbour_shrinkage` > 0 each bucket's statistics are first mixed
        with the mean of its `num_bits` Hamming neighbours. Most buckets are
        empty -- typically ~203 of 256 on real data -- and an empty bucket
        contributes exactly zero, so an example whose code was never populated
        gets nothing from that round. Mixing in the neighbours, which differ by
        exactly one split and are the nearest available evidence, lets those
        buckets say something. It does not shrink leaf magnitudes.
        """

        k = self.num_classes

        if not self.neighbour_shrinkage:
            numerator = self.stats[:num_rounds, :, :k]
            denominator = self.stats[:num_rounds, :, k:]

            self.logits[:num_rounds] = (
                numerator / (denominator + self.hessian_eps) * self.lr
            )

            return

        alpha = self.neighbour_shrinkage

        # chunked over rounds: the pooled copy is the same size as the slice
        for a in range(0, num_rounds, self._refresh_chunk):
            b = min(a + self._refresh_chunk, num_rounds)

            block = self.stats[a:b]

            pooled = torch.zeros_like(block)
            for j in range(self.num_bits):
                pooled += block[:, self.neighbours[j], :]

            smoothed = (1 - alpha) * block + (alpha / self.num_bits) * pooled

            self.logits[a:b] = (
                smoothed[..., :k] / (smoothed[..., k:] + self.hessian_eps) * self.lr
            )

    # -- per-round contributions ----------------------------------------------

    def gather_contributions(
        self, codes: torch.Tensor, lo: int, hi: int
    ) -> torch.Tensor:
        """-> (hi - lo, n, k) the logit contribution of each round separately."""

        flat = codes[lo:hi].to(torch.int64) + self.offsets[lo:hi, None]

        return self.logits.view(-1, self.num_classes)[flat]
