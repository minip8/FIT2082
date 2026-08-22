"""Bagged ensembles of independent boosters.

Screening showed this model is a variance-reduction machine: every attempt to
make individual hashes *smarter* (gain-based split selection, statistic decay,
row subsampling) made the ensemble worse, while simply adding partitions helped.
Averaging independent boosters is the other way to buy variance reduction, and
it is the honest control for `hashes_per_round`: E models of R rounds and one
model of E*R rounds fit the same number of hashes, so comparing them separates
"more capacity" from "more averaging".
"""

from typing import Any

import torch

from fit2082.boost.model import Array, HashBoost

# == bagging ===================================================================


class BaggedHashBoost:
    """`num_estimators` independent `HashBoost` models, averaged.

    Each estimator sees the same batches but draws its own splits, so they
    decorrelate through the splitter's randomness alone.
    """

    def __init__(self, num_estimators: int = 4, **kwargs: Any) -> None:

        assert num_estimators >= 1

        self.estimators = [HashBoost(**kwargs) for _ in range(num_estimators)]

    @property
    def num_estimators(self) -> int:

        return len(self.estimators)

    @property
    def num_rounds(self) -> int:
        """Rounds per estimator (they advance in lockstep)."""

        return self.estimators[0].num_rounds

    @property
    def device(self) -> torch.device:

        return self.estimators[0].device

    def fit_batch(self, X: Array, Y: Array) -> "BaggedHashBoost":

        for estimator in self.estimators:
            estimator.fit_batch(X, Y)

        return self

    def predict_proba(self, X: Array) -> torch.Tensor:
        """Mean of the estimators' probabilities.

        Averaged in probability space, not logit space: the estimators are
        separate models of the same target, not additive stages of one model.
        """

        total = self.estimators[0].predict_proba(X)

        for estimator in self.estimators[1:]:
            total = total + estimator.predict_proba(X)

        return total / self.num_estimators

    def predict(self, X: Array) -> torch.Tensor:
        """-> (n, k) log of the averaged probabilities.

        Not a sum of logits -- only monotone-equivalent to `predict_proba` for
        argmax purposes.
        """

        return self.predict_proba(X).clamp_min(torch.finfo(torch.float32).tiny).log()

    def __repr__(self) -> str:

        return (
            f"BaggedHashBoost(num_estimators={self.num_estimators}, "
            f"num_rounds={self.num_rounds})"
        )
