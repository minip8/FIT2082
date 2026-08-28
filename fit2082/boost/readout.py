"""Refitting the readout: the leaf tables as a model, not an accumulator.

`HashTables.predict_from_codes` sums one table row per round, and each row is
`lr * G / (H + eps)` -- a Newton step computed for that round *in isolation*,
under the assumption that every other round is held fixed. That is the correct
update while boosting. It is not the best set of leaf values for the finished
ensemble, and nothing here has ever measured the gap.

The sizing makes the point. `tables.logits` is `(rounds, 2**num_bits, k)`; a
multiclass linear model on the one-hot codes has `rounds * 2**num_bits` binary
features with exactly one nonzero per round, so it has *precisely the same
parameters*, because it is the same tensor. Refitting the readout is therefore
not a head bolted on top -- it is `tables.logits`, fit jointly against cross
entropy instead of round by round. The forward pass is already written and
already differentiable: `F.embedding_bag(..., mode="sum")`.

The risk is capacity: 16.8M parameters against 65k rows on Pedestrian. So the
fit is warm-started at the boosted tables (step zero reproduces the model
exactly), penalised toward them rather than toward zero (they are a good prior,
and `lam -> inf` recovers today's model), and offered as a ladder:

    "round"        one gain per round                 rounds params
    "round_class"  one gain per (round, class)        rounds * k
    "table"        the full table, shrunk to prior    rounds * 2**bits * k

`"round"` cannot meaningfully overfit and is nearly free -- it rides
`per_sample_weights`, so nothing is materialised -- which makes it the right
first test of whether the additive Newton readout is suboptimal at all.

Nothing here mutates the model. A `Readout` owns its weights and predicts from
them, so a refit can be compared against its own source model, and so a later
`fit_batch` -- which recomputes every leaf from `stats` in `refresh_logits` --
cannot silently erase the fit.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F

from fit2082.boost.model import Array, HashBoost

RUNGS = ("round", "round_class", "table")

# == reading a trained model ===================================================


def _estimators(model: Any) -> tuple[list[HashBoost], float]:
    """-> the boosters inside `model`, and the scale their logits combine at.

    A bare `HashBoost` is one booster at unit scale. `BaggedHashBoost` averages
    *probabilities*, which has no exact additive form, so the prior used for a
    bagged model is the logit-space average -- the geometric rather than the
    arithmetic mean. Close to the bagged model but not identical to it, which
    is why a bagged readout's step-zero error is reported alongside the fit.
    """

    estimators = getattr(model, "estimators", None)

    if estimators is None:
        return [model], 1.0

    return list(estimators), 1.0 / len(estimators)


def code_matrix(estimators: Iterable[HashBoost], X: Array) -> torch.Tensor:
    """-> (total_rounds, n) int32 codes, every estimator's rounds stacked."""

    return torch.cat([e._codes(X) for e in estimators], 0)


def prior_weights(estimators: list[HashBoost], scale: float) -> torch.Tensor:
    """-> (total_rounds, max_hash_size, k) the boosted tables, stacked.

    Estimators may differ in `num_bits` -- that is the whole point of a
    heterogeneous ensemble -- so narrower tables are zero-padded out to the
    widest. The padding is unreachable: a code from a `b`-bit hash is always
    below `2**b`.
    """

    size = max(e.tables.hash_size for e in estimators)
    k = estimators[0].num_classes

    weights = torch.zeros(
        (sum(e.num_rounds for e in estimators), size, k),
        dtype=torch.float32,
        device=estimators[0].device,
    )

    at = 0
    for e in estimators:
        r = e.num_rounds
        weights[at : at + r, : e.tables.hash_size] = e.tables.logits[:r] * scale
        at += r

    return weights


# == the refit readout =========================================================


class Readout:
    """A trained model's hash codes, combined by fitted weights.

    Exposes `predict`/`predict_proba` with the same signatures as `HashBoost`,
    so it drops into `experiment.evaluate` and the notebooks unchanged.
    """

    def __init__(
        self,
        estimators: list[HashBoost],
        weights: torch.Tensor,
        rung: str = "round",
    ) -> None:

        self.estimators = estimators
        self.weights = weights
        self.rung = rung

        self.num_classes = estimators[0].num_classes
        self.device = estimators[0].device

        self.offsets = (
            torch.arange(weights.shape[0], device=self.device) * weights.shape[1]
        )

    @property
    def num_rounds(self) -> int:
        """Total rounds across every estimator."""

        return int(self.weights.shape[0])

    def _flat(self, X: Array) -> torch.Tensor:
        """-> (n, total_rounds) int64 indices into the flattened weight table."""

        return code_matrix(self.estimators, X).t().to(torch.int64) + self.offsets

    def predict(self, X: Array) -> torch.Tensor:
        """-> (n, k) logits."""

        return F.embedding_bag(
            self._flat(X),
            self.weights.reshape(-1, self.num_classes),
            mode="sum",
        )

    def predict_proba(self, X: Array) -> torch.Tensor:

        return torch.softmax(self.predict(X), dim=-1)

    def __repr__(self) -> str:

        return f"Readout(rung={self.rung!r}, num_rounds={self.num_rounds})"


# == fitting ===================================================================


def _forward(
    flat: torch.Tensor,
    prior: torch.Tensor,
    parameter: torch.Tensor,
    rung: str,
    num_classes: int,
) -> torch.Tensor:
    """-> (n, k) logits under the given rung's parameterisation."""

    if rung == "round":
        # `per_sample_weights` scales each gathered row before the sum, so the
        # scaled table is never materialised: one gain per round, no memory.
        return F.embedding_bag(
            flat,
            prior.reshape(-1, num_classes),
            mode="sum",
            per_sample_weights=parameter.expand(flat.shape[0], -1),
        )

    if rung == "round_class":
        weights = parameter[:, None, :] * prior
    elif rung == "table":
        weights = prior + parameter
    else:
        raise ValueError(f"unknown rung {rung!r}; try one of {RUNGS}")

    return F.embedding_bag(flat, weights.reshape(-1, num_classes), mode="sum")


def _effective(prior: torch.Tensor, parameter: torch.Tensor, rung: str) -> torch.Tensor:
    """-> the (rounds, size, k) table the fitted parameters amount to."""

    if rung == "round":
        return parameter[:, None, None] * prior
    if rung == "round_class":
        return parameter[:, None, :] * prior

    return prior + parameter


def fit_readout(
    model: Any,
    X: Array,
    Y: Array,
    X_tune: Array | None = None,
    Y_tune: Array | None = None,
    rung: str = "round",
    lam: float = 1e-3,
    steps: int = 300,
    lr: float = 0.05,
    batch_size: int = 4096,
    eval_every: int = 10,
    generator: torch.Generator | None = None,
) -> tuple[Readout, dict[str, Any]]:
    """Refit `model`'s combination rule; -> (readout, diagnostics).

    `X_tune`/`Y_tune` are held-out *training* data used for early stopping. They
    must not be the reported validation set -- selecting on it would invalidate
    every comparison in the README. With no tune set the last step is kept.

    `lam` penalises the *mean* squared deviation from the boosted solution, so
    it means roughly the same thing across rungs despite their parameter counts
    differing by four orders of magnitude.
    """

    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r}; try one of {RUNGS}")

    estimators, scale = _estimators(model)

    prior = prior_weights(estimators, scale)
    rounds, size, k = prior.shape

    device = prior.device

    Xd = torch.as_tensor(X, device=device).to(torch.float32)
    Xd = Xd.reshape(Xd.shape[0], -1)
    Yd = torch.as_tensor(Y, device=device).to(torch.int64).reshape(-1)

    offsets = torch.arange(rounds, device=device) * size

    # warm start at the prior: "round"/"round_class" gains of 1, "table" at no
    # deviation. Step zero therefore reproduces the source model exactly (up to
    # the logit-space averaging noted in `_estimators` for bagged models).
    if rung == "round":
        parameter = torch.ones(rounds, device=device)
    elif rung == "round_class":
        parameter = torch.ones((rounds, k), device=device)
    else:
        parameter = torch.zeros_like(prior)

    baseline = torch.ones_like(parameter) if rung != "table" else None

    parameter = parameter.requires_grad_(True)
    optimiser = torch.optim.Adam([parameter], lr=lr)

    # -- evaluation on the tune slice, codes cached (it is small and fixed) ---

    # kept as one optional pair rather than two optional tensors, so that a
    # single `tune is not None` guard establishes both
    tune: tuple[torch.Tensor, torch.Tensor] | None = None

    if X_tune is not None:
        tune = (
            code_matrix(estimators, X_tune).t().to(torch.int64) + offsets,
            torch.as_tensor(Y_tune, device=device).to(torch.int64).reshape(-1),
        )

    def evaluate(pair: tuple[torch.Tensor, torch.Tensor]) -> float:

        flat, target = pair

        with torch.no_grad():
            logits = _forward(flat, prior, parameter, rung, k)

            return (logits.argmax(-1) != target).to(torch.float32).mean().item()

    n = Xd.shape[0]

    # the warm-start error: what the source model scores under this readout,
    # and the number every later step has to beat
    warm_start = evaluate(tune) if tune is not None else float("nan")

    best_error = warm_start
    best_parameter = parameter.detach().clone()

    history: list[dict[str, float]] = []

    for step in range(steps):
        index = torch.randint(0, n, (batch_size,), device=device, generator=generator)

        flat = code_matrix(estimators, Xd[index]).t().to(torch.int64) + offsets

        logits = _forward(flat, prior, parameter, rung, k)

        deviation = parameter if baseline is None else parameter - baseline
        loss = F.cross_entropy(logits, Yd[index]) + lam * deviation.pow(2).mean()

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        if tune is not None and (step + 1) % eval_every == 0:
            error = evaluate(tune)
            history.append({"step": step + 1, "loss": loss.item(), "tune": error})

            if error < best_error:
                best_error = error
                best_parameter = parameter.detach().clone()

    kept = best_parameter if tune is not None else parameter.detach()

    readout = Readout(estimators, _effective(prior, kept, rung), rung)

    return readout, {
        "rung": rung,
        "lam": lam,
        "steps": steps,
        "rounds": rounds,
        "num_parameters": int(kept.numel()),
        "tune_warm_start": warm_start,
        "tune_best": best_error,
        "history": history,
    }
