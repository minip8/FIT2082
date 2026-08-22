"""Boosted hash ensemble, torch-native and GPU-resident."""

import math
from typing import Any

import numpy.typing as npt
import torch

from fit2082.boost.objective import Objective, SoftmaxObjective
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

    Inputs may be numpy or torch; outputs are always torch tensors on `device`.
    """

    def __init__(
        self,
        num_classes: int,
        num_bits: int = 8,
        lr: float = 0.1,
        max_num_hashes: int = 100,
        device: str | torch.device | None = None,
        splitter: Splitter | None = None,
        objective: Objective | None = None,
        round_chunk: int | None = None,
        compile: bool = False,
        hessian_eps: float = 1e-6,
        generator: torch.Generator | None = None,
    ) -> None:

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.num_classes = int(num_classes)
        self.num_bits = int(num_bits)
        self.lr = float(lr)
        self.max_num_hashes = int(max_num_hashes)

        self.objective = objective or SoftmaxObjective(self.num_classes)
        self.splitter = splitter or HardPairSplitter(generator=generator)

        self.tables = HashTables(
            num_classes=self.num_classes,
            num_bits=self.num_bits,
            max_num_hashes=self.max_num_hashes,
            lr=self.lr,
            device=self.device,
            hessian_eps=hessian_eps,
            round_chunk=round_chunk,
            compile=compile,
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
        """Run one round of boosting on a (mini)batch."""

        if self.num_rounds >= self.max_num_hashes:
            raise RuntimeError(
                f"already at max_num_hashes ({self.max_num_hashes}); "
                "construct the model with a larger value to keep boosting"
            )

        Xd = self._X(X)
        Yd = self._Y(Y)

        r = self.num_rounds

        # feature-major, so each round's comparisons read contiguous memory
        Xt = Xd.t().contiguous()

        if r:
            codes = self.tables.encode(Xt, 0, r)
            logits = self.tables.predict_from_codes(codes, r)
        else:
            codes = None
            logits = torch.zeros(
                (Xd.shape[0], self.num_classes),
                dtype=torch.float32,
                device=self.device,
            )

        probabilities = self.objective.probabilities(logits)
        gradient, hessian = self.objective.gradients(probabilities, Yd)

        feature_indices, midpoints = self.splitter.propose(
            Xd, Yd, probabilities, self.num_bits
        )

        self.tables.feature_indices[r] = feature_indices
        self.tables.midpoints[r] = midpoints

        # the new round is just round r: update it in the same scatter as the rest
        new_codes = self.tables.encode(Xt, r, r + 1)
        codes = new_codes if codes is None else torch.cat([codes, new_codes], 0)

        self.tables.accumulate(codes, r + 1, torch.cat([-gradient, hessian], -1))
        self.tables.refresh_logits(r + 1)

        self.num_rounds = r + 1

        return self

    # -- predict ---------------------------------------------------------------

    def _codes(self, X: Array) -> torch.Tensor:

        Xt = self._X(X).t().contiguous()

        return self.tables.encode(Xt, 0, self.num_rounds)

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
            "feature_indices": self.tables.feature_indices,
            "midpoints": self.tables.midpoints,
            "stats": self.tables.stats,
            "logits": self.tables.logits,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "HashBoost":

        self.num_rounds = int(state["num_rounds"])

        for name in ("feature_indices", "midpoints", "stats", "logits"):
            getattr(self.tables, name).copy_(state[name].to(self.device))

        return self

    def __repr__(self) -> str:

        return (
            f"HashBoost(num_classes={self.num_classes}, num_bits={self.num_bits}, "
            f"lr={self.lr}, num_rounds={self.num_rounds}/{self.max_num_hashes}, "
            f"device='{self.device}')"
        )
