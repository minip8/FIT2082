# Angus Dempster, Chang Wei Tan, Lynn Miller
# Navid Mohammadi Foumani, Daniel F Schmidt, and Geoffrey I Webb
# Highly Scalable Time Series Classification for Very Large Datasets
# AALTD 2024 (ECML PKDD 2024)

# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
# QUANT: A Minimalist Interval Method for Time Series Classification
# ECML PKDD 2024

from collections.abc import Callable, Iterable, Iterator
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier
from tqdm import tqdm

# == types =====================================================================

Batch = tuple[npt.NDArray[Any], npt.NDArray[Any]]


class TrainingData(Protocol):
    """Batched dataset as consumed by QuantClassifier.fit."""

    _num_batches: int

    def set_batch_size(self, limit_mb: int) -> None: ...

    def __iter__(self) -> Iterator[Batch]: ...


# == eps =======================================================================

eps: float = torch.finfo(torch.float32).eps

# == generate intervals ========================================================


def make_intervals(input_length: int, depth: int) -> torch.Tensor:

    exponent = min(depth, int(np.log2(input_length)) + 1)

    intervals = []

    for n in 2 ** torch.arange(exponent):
        indices = torch.linspace(0, input_length, n + 1).long()

        intervals_n = torch.stack((indices[:-1], indices[1:]), 1)

        intervals.append(intervals_n)

        if n > 1 and intervals_n.diff().median() > 1:
            shift = int(np.ceil(input_length / n / 2))

            intervals.append(intervals_n[:-1] + shift)

    return torch.cat(intervals)


# == quantile function =========================================================


def f_quantile(X: torch.Tensor, div: int = 4) -> torch.Tensor:

    n = X.shape[-1]

    if n == 1:
        return X.view(X.shape[0], 1, X.shape[1] * X.shape[2])

    else:
        num_quantiles = 1 + (n - 1) // div

        if num_quantiles == 1:
            quantiles = X.quantile(
                torch.tensor([0.5], device=X.device), dim=-1
            ).permute(1, 2, 0)

            return quantiles.view(
                quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2]
            )

        else:
            quantiles = X.quantile(
                torch.linspace(0, 1, num_quantiles, device=X.device), dim=-1
            ).permute(1, 2, 0)
            quantiles[..., 1::2] = quantiles[..., 1::2] - X.mean(-1, keepdim=True)

            return quantiles.view(
                quantiles.shape[0], 1, quantiles.shape[1] * quantiles.shape[2]
            )


# == interval model (per representation) =======================================


class IntervalModel:
    def __init__(self, input_length: int, depth: int = 6, div: int = 4) -> None:

        assert div >= 1
        assert depth >= 1

        self.div = div

        self.intervals = make_intervals(
            input_length=input_length,
            depth=depth,
        )

    def fit(self, X: torch.Tensor, Y: npt.NDArray[Any] | None = None) -> None:

        pass

    def transform(self, X: torch.Tensor) -> torch.Tensor:

        features = []

        for a, b in self.intervals:
            features.append(f_quantile(X[..., a:b], div=self.div).squeeze(1))

        return torch.cat(features, -1)

    def fit_transform(
        self, X: torch.Tensor, Y: npt.NDArray[Any] | None = None
    ) -> torch.Tensor:

        self.fit(X, Y)

        return self.transform(X)


# == quant =====================================================================


class Quant:
    def __init__(self, depth: int = 6, div: int = 4) -> None:

        assert depth >= 1
        assert div >= 1

        self.depth = depth
        self.div = div

        self.representation_functions: tuple[
            Callable[[torch.Tensor], torch.Tensor], ...
        ] = (
            lambda X: X,
            lambda X: F.avg_pool1d(F.pad(X.diff(), (2, 2), "replicate"), 5, 1),
            lambda X: X.diff(n=2),
            lambda X: torch.fft.rfft(X).abs(),
        )

        self.models: dict[int, IntervalModel] = {}

        self.fitted = False

    def transform(self, X: torch.Tensor) -> torch.Tensor:

        assert self.fitted, "not fitted"

        features = []

        for index, function in enumerate(self.representation_functions):
            Z = function(X)

            features.append(self.models[index].transform(Z))

        return torch.cat(features, -1)

    def fit_transform(
        self, X: torch.Tensor, Y: npt.NDArray[Any] | None = None
    ) -> torch.Tensor:

        features = []

        for index, function in enumerate(self.representation_functions):
            Z = function(X)

            self.models[index] = IntervalModel(
                input_length=Z.shape[-1], depth=self.depth, div=self.div
            )

            features.append(self.models[index].fit_transform(Z, Y))

        self.fitted = True

        return torch.cat(features, -1)


# ==============================================================================


class QuantClassifier:
    def __init__(self, num_estimators: int = 200, **kwargs: Any) -> None:

        self.transform = Quant()

        self.num_estimators = num_estimators

        # print(f"self.num_estimators -> {self.num_estimators}", flush = True)

        self.classifier = ExtraTreesClassifier(
            n_estimators=0,
            criterion="entropy",
            max_features=0.1,
            n_jobs=-1,
            warm_start=True,
        )

        self.verbose: bool = kwargs.get("verbose", False)

        self._limit_mb: int = kwargs.get("limit_mb", 100)

        self._is_fitted = False

    def fit(self, training_data: TrainingData) -> None:

        training_data.set_batch_size(self._limit_mb)

        # print(f"training_data.batch_size -> {training_data.batch_size}", flush = True)

        num_batches = training_data._num_batches
        num_estimators_per_batch = self._set_num_estimators(num_batches)

        # print(f"num_batches -> {num_batches}", flush = True)
        # print(f"num_estimators_per_batch -> {num_estimators_per_batch}", flush = True)

        for i, (X, Y) in enumerate(
            tqdm(training_data, total=num_batches, disable=not self.verbose)
        ):
            self.classifier.n_estimators += num_estimators_per_batch[i]

            if i == 0:
                Z = self.transform.fit_transform(torch.tensor(X.astype(np.float32)), Y)
            else:
                Z = self.transform.transform(torch.tensor(X.astype(np.float32)))

            self.classifier.fit(Z, Y)

        self._is_fitted = True

    def _set_num_estimators(self, num_batches: int) -> npt.NDArray[np.int32]:

        num_estimators_per = max(1, int(self.num_estimators / num_batches))

        num_estimators_per_batch = (
            np.ones(num_batches, dtype=np.int32) * num_estimators_per
        )

        _total = num_estimators_per_batch.sum()
        _diff = self.num_estimators - _total
        if _diff > 0:
            num_estimators_per_batch[:_diff] += 1

        return num_estimators_per_batch

    def score(self, data: Iterable[Batch]) -> float:

        assert self._is_fitted

        num_incorrect = 0
        count = 0

        for X, Y in data:
            Z = self.transform.transform(torch.tensor(X.astype(np.float32)))

            num_incorrect += (self.classifier.predict(Z) != Y).sum()
            count += X.shape[0]

        return num_incorrect / count

    def score_logloss(self, data: Iterable[Batch]) -> tuple[float, torch.Tensor]:

        assert self._is_fitted

        num_incorrect = 0
        loss: torch.Tensor = torch.zeros(())
        count = 0

        for X, Y in data:
            Z = self.transform.transform(torch.tensor(X.astype(np.float32)))

            num_incorrect += (self.classifier.predict(Z) != Y).sum()
            # loss += F.nll_loss(torch.tensor(self.classifier.predict_proba(Z), dtype = torch.float32).log(), torch.tensor(Y, dtype = torch.int64), reduction = "sum")
            loss += F.nll_loss(
                torch.tensor(self.classifier.predict_proba(Z), dtype=torch.float32)
                .clip(eps, 1 - eps)
                .log(),
                torch.tensor(Y, dtype=torch.int64),
                reduction="sum",
            )
            count += X.shape[0]

        return num_incorrect / count, loss / count
