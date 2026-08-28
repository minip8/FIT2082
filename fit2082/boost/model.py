"""Boosted hash ensemble, torch-native and GPU-resident."""

import math
from collections.abc import Callable
from typing import Any, cast

import numpy.typing as npt
import torch

from fit2082.boost.objective import Objective, SoftmaxObjective
from fit2082.boost.partition import AxisAlignedPartitioner, Partitioner
from fit2082.boost.splits import HardPairSplitter, Splitter
from fit2082.boost.tables import HashTables

Array = torch.Tensor | npt.NDArray[Any]

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
    ) -> None:

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.num_classes = int(num_classes)
        self.num_bits = int(num_bits)
        self.lr = float(lr)
        self.max_num_hashes = int(max_num_hashes)
        self.hashes_per_round = int(hashes_per_round)

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

    def fit_batch(self, X: Array, Y: Array) -> "HashBoost":
        """Run `hashes_per_round` rounds of boosting on a (mini)batch."""

        Xd = self._X(X)
        Yd = self._Y(Y)

        # feature-major, so each round's comparisons read contiguous memory
        Xt = Xd.t().contiguous()

        # Existing rounds' split points are fixed for the whole batch, so their
        # codes are encoded once and extended by one column per added hash --
        # rather than re-encoding every round for each hash.
        codes = None

        for _ in range(self.hashes_per_round):
            if self.num_rounds >= self.max_num_hashes:
                raise RuntimeError(
                    f"already at max_num_hashes ({self.max_num_hashes}); "
                    "construct the model with a larger value to keep boosting"
                )

            r = self.num_rounds

            if r:
                if codes is None:
                    codes = self.partitioner.encode(Xt, 0, r)

                logits = self.tables.predict_from_codes(codes, r)
            else:
                logits = torch.zeros(
                    (Xd.shape[0], self.num_classes),
                    dtype=torch.float32,
                    device=self.device,
                )

            probabilities = self.objective.probabilities(logits)
            gradient, hessian = self.objective.gradients(probabilities, Yd)

            self.partitioner.propose(r, Xd, Yd, probabilities, gradient, hessian)

            # the new round is just round r: update it in the same scatter
            new_codes = self.partitioner.encode(Xt, r, r + 1)
            codes = new_codes if codes is None else torch.cat([codes, new_codes], 0)

            self.tables.accumulate(codes, r + 1, torch.cat([-gradient, hessian], -1))
            self.tables.refresh_logits(r + 1)

            self.num_rounds = r + 1

        return self

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

        return self

    def __repr__(self) -> str:

        return (
            f"HashBoost(num_classes={self.num_classes}, num_bits={self.num_bits}, "
            f"lr={self.lr}, num_rounds={self.num_rounds}/{self.max_num_hashes}, "
            f"device='{self.device}')"
        )
