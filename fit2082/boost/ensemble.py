"""Bagged ensembles of independent boosters.

Screening showed this model is a variance-reduction machine: every attempt to
make individual hashes *smarter* (gain-based split selection, statistic decay,
row subsampling) made the ensemble worse, while simply adding partitions helped.
Averaging independent boosters is the other way to buy variance reduction, and
it is the honest control for `hashes_per_round`: E models of R rounds and one
model of E*R rounds fit the same number of hashes, so comparing them separates
"more capacity" from "more averaging".

Members built from identical kwargs decorrelate through the splitter's
randomness alone, which leaves the main lever unused -- the variance of an
average falls with the *correlation* between its members, not just their count.
`overrides` gives each member a different configuration, so they differ in what
they *are* (bit width, learning rate, partition family) and not only in which
random draws they got. The ingredients worth mixing are the ones already
measured as individually break-even (`bits_6`, `lr_0.05`): near-neutral in mean
quality is exactly what makes a good ensemble member, since the mixture keeps
the mean and cuts the correlation. `estimator_agreement` measures whether that
second half actually happened.
"""

from typing import Any

import torch

from fit2082.boost.model import Array, HashBoost

# == bagging ===================================================================


class BaggedHashBoost:
    """`num_estimators` independent `HashBoost` models, averaged.

    With no `overrides` each estimator sees the same batches under the same
    configuration and they decorrelate through the splitter's randomness alone.

    `overrides` is a list of kwarg patches, cycled to `num_estimators` and
    merged over the shared `kwargs`, so a two-entry list drives four estimators
    as ABAB. To vary the partition family, pass the *class* (or a factory) as
    `partitioner` rather than an instance -- `HashBoost` constructs one per
    estimator, where a shared instance would give every member the same split
    tables:

        BaggedHashBoost(
            num_estimators=4,
            overrides=[{}, {"partitioner": ObliquePartitioner}],
            num_classes=k,
        )
    """

    def __init__(
        self,
        num_estimators: int = 4,
        overrides: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:

        assert num_estimators >= 1

        patches = overrides or [{}]

        self.overrides = overrides
        self.estimators = [
            HashBoost(**{**kwargs, **patches[i % len(patches)]})
            for i in range(num_estimators)
        ]

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

    # -- diagnostics -----------------------------------------------------------

    def estimator_agreement(self, X: Array) -> float:
        """-> mean fraction of examples on which two estimators pick the same class.

        The quantity bagging trades on. Reported alongside error so that a
        heterogeneous mixture can be judged on the mechanism -- lower agreement
        at equal member quality is the thing that is supposed to buy accuracy --
        and not only on the outcome, which the +-0.003 noise floor makes hard to
        read on its own.
        """

        if self.num_estimators < 2:
            return 1.0

        # (E, n): one argmax per estimator, computed one at a time so the
        # probabilities of all E models are never live at once
        predictions = torch.stack(
            [estimator.predict(X).argmax(-1) for estimator in self.estimators]
        )

        agreements = [
            (predictions[i] == predictions[j]).to(torch.float32).mean()
            for i in range(self.num_estimators)
            for j in range(i + 1, self.num_estimators)
        ]

        return torch.stack(agreements).mean().item()

    def __repr__(self) -> str:

        mixed = ", heterogeneous" if self.overrides else ""

        return (
            f"BaggedHashBoost(num_estimators={self.num_estimators}, "
            f"num_rounds={self.num_rounds}{mixed})"
        )
