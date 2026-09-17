"""Boosted hash ensemble, torch-native and GPU-resident."""

import math
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import numpy.typing as npt
import torch

from fit2082.boost.objective import Objective, SoftmaxObjective
from fit2082.boost.partition import AxisAlignedPartitioner, Partitioner, code_dtype
from fit2082.boost.splits import HardPairSplitter, Splitter
from fit2082.boost.tables import HashTables

Array = torch.Tensor | npt.NDArray[Any]


class _FrozenRows(NamedTuple):
    """One batch's view of the frozen-round cache; see `HashBoost.fit_batch`."""

    ids: torch.Tensor  # (n,) row ids, on the host
    index: torch.Tensor  # (n,) the same ids, on the device
    logits: torch.Tensor  # (n, k) each row's logits summed over its cached rounds
    lo: int  # fewest rounds any row has cached
    hi: int  # most rounds any row has cached
    upto: torch.Tensor | None  # (n,) rounds cached per row, on the device, if lo < hi


# == model =====================================================================


class HashBoost:
    """Gradient boosting over random hash partitions.

    Each round adds one hash: `num_bits` (feature, midpoint) comparisons whose
    bits index a `2**num_bits x num_classes` table of leaf values. Prediction
    sums one row per round. Every mini-batch updates the buckets of *all*
    existing rounds, so cost per batch grows linearly with the number of rounds.

    `hashes_per_round` decouples model capacity from the number of mini-batches:
    with H > 1 each batch contributes H hashes instead of one. Measured on
    Pedestrian + QUANT, H=2 reached 0.2124 validation error against a baseline
    of 0.2205 +- 0.0029, at the same number of passes over the data.

    `active_rounds` caps the per-batch cost instead of letting it grow with the
    model: only the newest `active_rounds` rounds keep accumulating, and older
    rounds are frozen. A frozen round's contribution to an example never
    changes again, so when `fit_batch` is told which rows it is seeing, each
    row's frozen contributions are summed once, cached, and never encoded or
    gathered again. Per-batch work becomes O(active_rounds + rounds frozen
    since the row was last seen) rather than O(rounds), which turns the
    quadratic total into a linear one. Freezing does change the model -- how
    much depends on the dataset; see the README.

    Inputs may be numpy or torch; outputs are always torch tensors on `device`.
    """

    def __init__(
        self,
        num_classes: int,
        num_bits: int = 8,
        lr: float = 0.1,
        max_num_hashes: int = 100,
        hashes_per_round: int = 1,
        device: str | torch.device | None = None,
        splitter: Splitter | None = None,
        partitioner: Partitioner | Callable[..., Partitioner] | None = None,
        objective: Objective | None = None,
        round_chunk: int | None = None,
        compile: bool = False,
        hessian_eps: float = 1e-6,
        neighbour_shrinkage: float = 0.0,
        shrinkage_tau: float = 0.0,
        generator: torch.Generator | None = None,
        active_rounds: int | None = None,
    ) -> None:

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.num_classes = int(num_classes)
        self.num_bits = int(num_bits)
        self.lr = float(lr)
        self.max_num_hashes = int(max_num_hashes)
        self.hashes_per_round = int(hashes_per_round)

        if active_rounds is not None and active_rounds < 1:
            raise ValueError(
                f"active_rounds must be at least 1, got {active_rounds}: a new "
                "round has to accumulate the batch that created it"
            )

        self.active_rounds = None if active_rounds is None else int(active_rounds)

        # Per row id: how many leading rounds are summed into `_frozen_logits`
        # (the count on the host, the logits on the device). Grown on demand,
        # and only used with `active_rounds`.
        self._frozen_upto: torch.Tensor | None = None
        self._frozen_logits: torch.Tensor | None = None

        self.objective = objective or SoftmaxObjective(self.num_classes)

        # `splitter` selects members of a family; `partitioner` defines the
        # family itself and owns the split parameters.
        resolved: Partitioner

        if partitioner is None:
            resolved = AxisAlignedPartitioner(
                num_bits=self.num_bits,
                max_num_hashes=self.max_num_hashes,
                device=self.device,
                splitter=splitter or HardPairSplitter(generator=generator),
                compile=compile,
            )
        elif callable(partitioner):
            # A class or factory rather than an instance -- the partitioners
            # here define no `__call__`, so the two are distinguishable. This
            # is what lets an ensemble give each member its own partition
            # family: passing one *instance* would share a single set of split
            # tables between every estimator.
            factory = cast(Callable[..., Partitioner], partitioner)

            resolved = factory(
                num_bits=self.num_bits,
                max_num_hashes=self.max_num_hashes,
                device=self.device,
                compile=compile,
            )
        else:
            resolved = partitioner

        self.partitioner = resolved

        self.tables = HashTables(
            num_classes=self.num_classes,
            num_bits=self.num_bits,
            max_num_hashes=self.max_num_hashes,
            lr=self.lr,
            device=self.device,
            hessian_eps=hessian_eps,
            neighbour_shrinkage=neighbour_shrinkage,
            shrinkage_tau=shrinkage_tau,
            round_chunk=round_chunk,
        )

        self.num_rounds = 0

    # -- input handling --------------------------------------------------------

    def _X(self, X: Array) -> torch.Tensor:

        Z = torch.as_tensor(X, device=self.device)

        return Z.to(torch.float32).reshape(Z.shape[0], -1)

    def _Y(self, Y: Array) -> torch.Tensor:

        return torch.as_tensor(Y, device=self.device).to(torch.int64).reshape(-1)

    # -- fit -------------------------------------------------------------------

    def fit_batch(self, X: Array, Y: Array, rows: Array | None = None) -> "HashBoost":
        """Run `hashes_per_round` rounds of boosting on a (mini)batch.

        `rows` is (n,) stable integer ids for the examples -- their indices into
        the training set, say -- and matters only with `active_rounds`, where it
        lets each row's frozen rounds be summed once and skipped thereafter.
        Without it the model trains identically, re-reading frozen rounds.
        """

        Xd = self._X(X)
        Yd = self._Y(Y)

        # feature-major, so each round's comparisons read contiguous memory
        Xt = Xd.t().contiguous()

        n = Xd.shape[0]
        r0 = self.num_rounds

        # Each row's logits summed over its first `upto` rounds, cached when
        # those rounds froze. Rows last seen at different times disagree on
        # `upto`, but no row in the batch needs any round below the least of
        # them, `lo`.
        cache = self._recall_frozen(rows, n)

        frozen = None if cache is None else cache.logits
        upto = None if cache is None else cache.upto
        lo, hi = (0, 0) if cache is None else (cache.lo, cache.hi)

        # Existing rounds' split points are fixed for the whole batch, so their
        # codes are encoded once, into a buffer with room for every hash this
        # batch adds -- rather than re-encoded, or re-concatenated, per hash.
        # Its first row is round `base`.
        base = lo

        codes = torch.empty(
            (r0 + self.hashes_per_round - base, n),
            dtype=code_dtype(self.num_bits),
            device=self.device,
        )

        if r0 > base:
            codes[: r0 - base] = self.partitioner.encode(Xt, base, r0)

        for _ in range(self.hashes_per_round):
            if self.num_rounds >= self.max_num_hashes:
                raise RuntimeError(
                    f"already at max_num_hashes ({self.max_num_hashes}); "
                    "construct the model with a larger value to keep boosting"
                )

            r = self.num_rounds

            # This hash updates rounds from `first` on, and no later hash starts
            # earlier, so every round below `first` is final. That bound only
            # grows, and `upto` was set from an earlier one, so `hi <= first`:
            # rows can only disagree on `upto` when something is left to fold,
            # and after folding they all agree.
            first = self._first_active(r + 1)

            if first > lo:
                folded = self.tables.predict_from_codes(
                    codes[lo - base : first - base],
                    first,
                    lo=lo,
                    skip_below=upto if lo < hi else None,
                )

                frozen = folded if frozen is None else frozen + folded
                lo = hi = first

            logits = (
                torch.zeros(
                    (n, self.num_classes), dtype=torch.float32, device=self.device
                )
                if frozen is None
                else frozen
            )

            if r > first:
                logits = logits + self.tables.predict_from_codes(
                    codes[first - base : r - base], r, lo=first
                )

            probabilities = self.objective.probabilities(logits)
            gradient, hessian = self.objective.gradients(probabilities, Yd)

            self.partitioner.propose(r, Xd, Yd, probabilities, gradient, hessian)

            # the new round is just round r: update it in the same scatter
            codes[r - base : r + 1 - base] = self.partitioner.encode(Xt, r, r + 1)

            self.tables.accumulate(
                codes[first - base : r + 1 - base],
                r + 1,
                torch.cat([-gradient, hessian], -1),
                lo=first,
            )
            self.tables.refresh_logits(r + 1, lo=first)

            self.num_rounds = r + 1

        if cache is not None:
            assert frozen is not None

            self._remember_frozen(cache, lo, frozen)

        return self

    # -- frozen rounds ---------------------------------------------------------

    def _first_active(self, num_rounds: int) -> int:
        """The oldest round a batch still updates once `num_rounds` exist."""

        if self.active_rounds is None:
            return 0

        return max(0, num_rounds - self.active_rounds)

    def _recall_frozen(self, rows: Array | None, n: int) -> _FrozenRows | None:
        """-> this batch's view of the frozen-round cache, or None without one.

        There is no cache without both `rows` and `active_rounds`. A row never
        seen before has summed zero rounds, to zero logits.

        The per-row round counts live on the host. `lo` decides which rounds
        get encoded, so reading it off the device would stall the host on the
        GPU queue every batch -- measured, that cost more than the cache saved.
        Row ids already on the GPU cost one such stall to bring back, so pass
        them from the host.
        """

        if rows is None or self.active_rounds is None:
            return None

        ids = rows.detach().cpu() if isinstance(rows, torch.Tensor) else rows
        ids = torch.as_tensor(ids).to(torch.int64).reshape(-1)

        if ids.shape[0] != n:
            raise ValueError(f"got {ids.shape[0]} row ids for {n} examples")

        size = 0 if self._frozen_upto is None else self._frozen_upto.shape[0]
        needed = int(ids.max()) + 1

        if needed > size:
            # doubling, so a stream of ever-larger ids costs amortised O(1)
            grown = max(needed, 2 * size)

            upto = torch.zeros(grown, dtype=torch.int64)
            logits = torch.zeros(
                (grown, self.num_classes), dtype=torch.float32, device=self.device
            )

            if self._frozen_upto is not None and self._frozen_logits is not None:
                upto[:size] = self._frozen_upto
                logits[:size] = self._frozen_logits

            self._frozen_upto, self._frozen_logits = upto, logits

        assert self._frozen_upto is not None and self._frozen_logits is not None

        counts = self._frozen_upto[ids]
        lo, hi = int(counts.min()), int(counts.max())

        index = self._to_device(ids)

        return _FrozenRows(
            ids=ids,
            index=index,
            logits=self._frozen_logits[index],
            lo=lo,
            hi=hi,
            upto=self._to_device(counts) if lo < hi else None,
        )

    def _remember_frozen(
        self, cache: _FrozenRows, upto: int, logits: torch.Tensor
    ) -> None:

        assert self._frozen_upto is not None and self._frozen_logits is not None

        self._frozen_upto[cache.ids] = upto
        self._frozen_logits[cache.index] = logits

    def _to_device(self, host: torch.Tensor) -> torch.Tensor:
        """Copy a host tensor to the device without waiting on the GPU queue.

        A copy from pageable memory synchronises the stream; a non-blocking one
        from pinned memory does not.
        """

        if self.device.type != "cuda":
            return host.to(self.device)

        return host.pin_memory().to(self.device, non_blocking=True)

    # -- predict ---------------------------------------------------------------

    def _codes(self, X: Array) -> torch.Tensor:

        Xt = self._X(X).t().contiguous()

        return self.partitioner.encode(Xt, 0, self.num_rounds)

    def predict(self, X: Array) -> torch.Tensor:
        """-> (n, k) raw logits from the current ensemble."""

        return self.tables.predict_from_codes(self._codes(X), self.num_rounds)

    def predict_proba(self, X: Array) -> torch.Tensor:
        """-> (n, k) probabilities."""

        return self.objective.probabilities(self.predict(X))

    def predict_all(
        self, X: Array, out_device: str | torch.device | None = None
    ) -> torch.Tensor:
        """-> (num_rounds + 1, n, k) per-round contributions, row 0 the prior.

        Beware the size: this is `(rounds + 1) * n * k` floats -- over a gigabyte
        for a few hundred classes and a thousand rounds. Pass `out_device="cpu"`
        to keep it off the GPU, or use `staged_error`, which needs only O(n * k).
        """

        codes = self._codes(X)

        n = codes.shape[1]
        r = self.num_rounds

        out = torch.empty(
            (r + 1, n, self.num_classes),
            dtype=torch.float32,
            device=torch.device(out_device) if out_device is not None else self.device,
        )
        out[0] = math.log(1 / self.num_classes)

        chunk = self.tables.round_chunk(n)

        for a in range(0, r, chunk):
            b = min(a + chunk, r)

            out[a + 1 : b + 1] = self.tables.gather_contributions(codes, a, b).to(
                out.device
            )

        return out

    def staged_error(self, X: Array, Y: Array) -> torch.Tensor:
        """-> (num_rounds + 1,) misclassification rate after each round.

        The streaming equivalent of
        `(predict_all(X).cumsum(0).argmax(-1) != Y).mean(-1)`, without ever
        materialising the full per-round tensor.
        """

        codes = self._codes(X)
        Yd = self._Y(Y)

        n = codes.shape[1]
        r = self.num_rounds

        errors = torch.empty(r + 1, dtype=torch.float32, device=self.device)

        running = torch.full(
            (n, self.num_classes),
            math.log(1 / self.num_classes),
            dtype=torch.float32,
            device=self.device,
        )
        errors[0] = (running.argmax(-1) != Yd).to(torch.float32).mean()

        chunk = self.tables.round_chunk(n)

        for a in range(0, r, chunk):
            b = min(a + chunk, r)

            cumulative = running[None] + self.tables.gather_contributions(
                codes, a, b
            ).cumsum(0)

            errors[a + 1 : b + 1] = (
                (cumulative.argmax(-1) != Yd).to(torch.float32).mean(-1)
            )

            running = cumulative[-1]

        return errors

    # -- misc ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:

        return {
            "num_rounds": self.num_rounds,
            "stats": self.tables.stats,
            "logits": self.tables.logits,
            "partitioner": {
                name: value
                for name, value in vars(self.partitioner).items()
                if isinstance(value, torch.Tensor)
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> "HashBoost":

        self.num_rounds = int(state["num_rounds"])

        for name in ("stats", "logits"):
            getattr(self.tables, name).copy_(state[name].to(self.device))

        for name, value in state["partitioner"].items():
            getattr(self.partitioner, name).copy_(value.to(self.device))

        # cached sums of the old tables' rounds would be wrong for the new ones
        self._frozen_upto = None
        self._frozen_logits = None

        return self

    def __repr__(self) -> str:

        return (
            f"HashBoost(num_classes={self.num_classes}, num_bits={self.num_bits}, "
            f"lr={self.lr}, num_rounds={self.num_rounds}/{self.max_num_hashes}, "
            f"device='{self.device}')"
        )
