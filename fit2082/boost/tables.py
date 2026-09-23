"""The performance core: hash encoding, table lookup, and bucket accumulation.

All state lives on one device. Every operation is batched over boosting rounds
and chunked so peak memory stays bounded.

Layout notes (these are load-bearing -- see the comments at each use site):

* Hash codes are stored **round-major** `(rounds, n)`; they come from a
  `Partitioner` (`fit2082.boost.partition`), which owns the split parameters.
* `stats` fuses the residual numerator and hessian denominator into one tensor
  so a single scatter updates both.
* Prediction indexes a re-based, flattened table slice and so uses
  **chunk-local** offsets; accumulation scatters along each round's own bucket
  axis and so needs no offsets at all.
"""

import torch
import torch.nn.functional as F

# Below this many (row, class) outputs, prediction gathers each round's leaf
# rows and sums them, instead of calling `embedding_bag`. `embedding_bag`
# gives each output one thread, which walks every round in turn: with 50 rows
# x 2 classes that is 100 threads each summing 3,000 values, and it is 4.5x
# slower than gather-and-sum (0.47 against 0.10 ms). Above the limit the
# gather's (rounds, n, k) intermediate costs more than those serial sums:
# 600 x 60 is 3.1x slower gathered. Measured at 3,000 rounds; the crossover
# sits between 6,000 (a tie) and 12,000.
GATHER_LIMIT = 8192


def refresh_into(
    logits: torch.Tensor, stats: torch.Tensor, k: int, eps: float, lr: float
) -> None:
    """Leaf values from their statistics, written into `logits`.

    Compiled, this is one kernel that reads the statistics once, where the eager
    path in `HashTables.refresh_logits` takes three passes over the table. That
    is 2.3x faster with 60 classes (5.3 ms to 2.3 ms at 3,000 rounds). It is
    not bit-identical to the eager path: about a quarter of the leaves differ,
    by at most 2 ulps.
    """

    logits.copy_(stats[..., :k] / (stats[..., k:] + eps) * lr)


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
        shrinkage_tau: float = 0.0,
        round_chunk: int | None = None,
        chunk_budget_bytes: int = 128 << 20,
        compile: bool = False,
    ) -> None:

        self.num_classes = num_classes
        self.num_bits = num_bits
        self.hash_size = 2**num_bits
        self.max_num_hashes = max_num_hashes
        self.lr = lr
        self.device = device
        self.hessian_eps = hessian_eps
        self.neighbour_shrinkage = neighbour_shrinkage
        self.shrinkage_tau = shrinkage_tau

        self._round_chunk = round_chunk
        self._refresh_chunk = 128
        self._chunk_budget_bytes = chunk_budget_bytes

        # the unsmoothed refresh as one fused kernel; see `refresh_into`
        self._refresh = torch.compile(refresh_into, dynamic=True) if compile else None

        k = num_classes
        m = max_num_hashes
        s = self.hash_size

        # stats[..., :k] is the accumulated *negated* residual (the numerator);
        # stats[..., k:] is the accumulated hessian (the denominator). Keeping
        # them adjacent lets one scatter update both, and keeping numerator
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

        Bounds the largest per-chunk intermediate, the (chunk, n, k) floats that
        `gather_contributions` returns, with room to spare. Too small and
        per-launch overhead dominates.

        There used to be a cap of 256 rounds as well. For a big batch the budget
        binds first anyway: 4,096 rows x 82 classes allows 47 rounds. For a
        small one the cap alone set the chunk, so 3,000 rounds took 12 chunks of
        a few launches each where one chunk fits. Accumulating 3,000 rounds for
        50 rows went from 0.44 to 0.045 ms without it.
        """

        if self._round_chunk is not None:
            return self._round_chunk

        per_round = num_examples * 2 * self.num_classes * 4

        return max(8, self._chunk_budget_bytes // max(1, per_round))

    # -- predict ---------------------------------------------------------------

    def predict_from_codes(
        self,
        codes: torch.Tensor,
        num_rounds: int,
        lo: int = 0,
        skip_below: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum one table row per round over rounds [lo, num_rounds) -> (n, k).

        `codes[0]` holds round `lo`. `skip_below`, if given, is (n,) per
        example: example i leaves out every round below `skip_below[i]`.
        """

        n = codes.shape[1]
        k = self.num_classes
        chunk = self.round_chunk(n)

        out = torch.zeros((n, k), dtype=torch.float32, device=self.device)

        # few outputs: gather each round's leaf row and sum, see GATHER_LIMIT
        gather = n * k <= GATHER_LIMIT

        for a in range(lo, num_rounds, chunk):
            b = min(a + chunk, num_rounds)

            if gather:
                # (chunk, n) indices into the re-based table slice
                index = (
                    codes[a - lo : b - lo].to(torch.int64) + self.offsets[: b - a, None]
                )
                contributions = self.logits[a:b].reshape(-1, k)[index]

                if skip_below is not None:
                    rounds = torch.arange(a, b, device=self.device)
                    keep = rounds[:, None] >= skip_below[None, :]
                    contributions = contributions * keep[..., None]

                out += contributions.sum(0)
                continue

            # chunk-local offsets: the table slice below is re-based to 0
            flat = (
                codes[a - lo : b - lo].t().to(torch.int64) + self.offsets[: b - a]
            ).contiguous()

            weights = None

            if skip_below is not None:
                rounds = torch.arange(a, b, device=self.device)
                weights = (rounds[None, :] >= skip_below[:, None]).to(torch.float32)

            out += F.embedding_bag(
                flat,
                self.logits[a:b].reshape(-1, self.num_classes),
                mode="sum",
                per_sample_weights=weights,
            )

        return out

    # -- accumulate ------------------------------------------------------------

    def accumulate(
        self,
        codes: torch.Tensor,
        num_rounds: int,
        updates: torch.Tensor,
        lo: int = 0,
    ) -> None:
        """Scatter-add `updates` (n, 2k) into each example's bucket, per round.

        Covers rounds [lo, num_rounds); `codes[0]` holds round `lo`.
        """

        n, width = updates.shape
        chunk = self.round_chunk(n)

        for a in range(lo, num_rounds, chunk):
            b = min(a + chunk, num_rounds)

            # Every round scatters the *same* per-example values, just to
            # different buckets. `expand` lends that one (n, 2k) block to each
            # round in the chunk as a view, so neither the index nor the source
            # is ever copied out to (chunk, n, 2k). Tiling the source with
            # `updates.repeat(chunk, 1)` and scattering into the flattened
            # buffer did copy it, and was 1.6-2.4x slower for that.
            index = (
                codes[a - lo : b - lo, :, None].to(torch.int64).expand(-1, -1, width)
            )

            self.stats[a:b].scatter_add_(1, index, updates.expand(b - a, -1, -1))

    # -- refresh ---------------------------------------------------------------

    def refresh_logits(self, num_rounds: int, lo: int = 0) -> None:
        """Recompute leaf values of rounds [lo, num_rounds) from their buffers.

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

        `shrinkage_tau` > 0 selects the same mixing with a *per bucket* weight
        instead of one global alpha:

            alpha_s = tau / (tau + mass_s),   mass_s = sum_k hessian[s, k]

        A bucket the data never reached has mass 0 and takes its neighbours'
        evidence in full; a well-populated one is left almost untouched. A
        single global alpha has to serve both, which is the likeliest reason
        the fixed version measured as noise (it helps in two paired
        comparisons and hurts in two) despite the mechanism demonstrably
        working. `tau` is in units of accumulated hessian, so it reads as "the
        bucket mass at which a bucket trusts itself and its neighbours
        equally".
        """

        k = self.num_classes

        if not (self.neighbour_shrinkage or self.shrinkage_tau) and self._refresh:
            self._refresh(
                self.logits[lo:num_rounds],
                self.stats[lo:num_rounds],
                k,
                self.hessian_eps,
                self.lr,
            )

            return

        if not (self.neighbour_shrinkage or self.shrinkage_tau):
            # The same three operations in the same order as
            # `numerator / (denominator + eps) * lr`, so bit-identical to it,
            # but written straight into the logits: that form allocated two
            # table-sized temporaries per hash.
            out = self.logits[lo:num_rounds]

            torch.add(self.stats[lo:num_rounds, :, k:], self.hessian_eps, out=out)
            torch.div(self.stats[lo:num_rounds, :, :k], out, out=out)
            out.mul_(self.lr)

            return

        # chunked over rounds: the pooled copy is the same size as the slice
        for a in range(lo, num_rounds, self._refresh_chunk):
            b = min(a + self._refresh_chunk, num_rounds)

            block = self.stats[a:b]

            pooled = torch.zeros_like(block)
            for j in range(self.num_bits):
                pooled += block[:, self.neighbours[j], :]

            if self.shrinkage_tau:
                tau = self.shrinkage_tau

                # (chunk, buckets, 1), broadcasting over numerator and
                # denominator alike. An empty bucket has mass exactly 0 and so
                # alpha exactly 1 -- no division by zero, because tau > 0 here.
                mass = block[..., k:].sum(-1, keepdim=True)
                alpha = tau / (tau + mass)
            else:
                alpha = self.neighbour_shrinkage

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
