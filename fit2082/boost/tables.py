"""The performance core: hash encoding, table lookup, and bucket accumulation.

All state lives on one device. Every operation is batched over boosting rounds
and chunked so peak memory stays bounded.

Layout notes (these are load-bearing -- see the comments at each use site):

* Hash codes are stored **round-major** `(rounds, n)`.
* `stats` fuses the residual numerator and hessian denominator into one tensor
  so a single scatter updates both.
* Prediction indexes a re-based table slice and so uses **chunk-local** offsets;
  accumulation scatters into the full buffer and so uses **global** offsets.
"""

import torch
import torch.nn.functional as F

# == encoding ==================================================================


def encode_chunk(
    Xt: torch.Tensor,
    feature_indices: torch.Tensor,
    midpoints: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """(p, n) feature-major X + (c, B) splits -> (c, n) int32 hash codes.

    Kept at module scope so it can be handed to `torch.compile`, which fuses the
    gather, the comparison and the bit-packing into a single kernel.
    """

    bits = Xt[feature_indices] <= midpoints[:, :, None]  # (c, B, n)

    return (bits.to(torch.int32) * weights[None, :, None]).sum(1)


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

        self._round_chunk = round_chunk
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

        self.feature_indices = torch.zeros(
            (m, num_bits), dtype=torch.int64, device=device
        )
        self.midpoints = torch.zeros((m, num_bits), dtype=torch.float32, device=device)

        self.weights = (2 ** torch.arange(num_bits, device=device)).to(torch.int32)
        self.offsets = torch.arange(m, device=device, dtype=torch.int64) * s

        self._encode = (
            torch.compile(encode_chunk, dynamic=True) if compile else encode_chunk
        )

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

    # -- encode ----------------------------------------------------------------

    def encode(self, Xt: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        """-> (hi - lo, n) int32 hash codes for rounds [lo, hi)."""

        n = Xt.shape[1]
        chunk = self.round_chunk(n)

        codes = torch.empty((hi - lo, n), dtype=torch.int32, device=self.device)

        for a in range(lo, hi, chunk):
            b = min(a + chunk, hi)

            codes[a - lo : b - lo] = self._encode(
                Xt, self.feature_indices[a:b], self.midpoints[a:b], self.weights
            )

        return codes

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
        """

        k = self.num_classes

        numerator = self.stats[:num_rounds, :, :k]
        denominator = self.stats[:num_rounds, :, k:]

        self.logits[:num_rounds] = (
            numerator / (denominator + self.hessian_eps) * self.lr
        )

    # -- per-round contributions ----------------------------------------------

    def gather_contributions(
        self, codes: torch.Tensor, lo: int, hi: int
    ) -> torch.Tensor:
        """-> (hi - lo, n, k) the logit contribution of each round separately."""

        flat = codes[lo:hi].to(torch.int64) + self.offsets[lo:hi, None]

        return self.logits.view(-1, self.num_classes)[flat]
