"""Sweep runner for HashBoost variants.

Screening variants one run at a time is misleading: the run-to-run standard
deviation on Pedestrian + QUANT is about 0.003, so a single run can easily show
a 0.006 "improvement" that is pure noise. Several plausible ideas looked like
wins until they were run over multiple seeds. Every variant here is therefore
run over `--seeds` seeds and reported as mean +- sd.

    uv run python -m fit2082.boost.experiment --list
    uv run python -m fit2082.boost.experiment --variants baseline,capacity_2 --seeds 4
"""

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fit2082.boost import BaggedHashBoost, HashBoost, ObliquePartitioner
from fit2082.boost.readout import fit_readout
from fit2082.demo.utils import Dataset
from fit2082.quant.quant import Quant

# == data ======================================================================


@dataclass
class Split:
    """Cached, device-resident features for one train/validation split.

    `X_tune`/`Y_tune` is a second held-out slice, for choosing anything that
    has to be chosen after training -- the readout's early stopping and `lam`.
    It exists so that none of those decisions touch `X_va`, which is the number
    every result in the README is quoted against; selecting on `X_va` would
    quietly invalidate the lot.
    """

    batches: list[tuple[torch.Tensor, torch.Tensor]]
    X_va: torch.Tensor
    Y_va: torch.Tensor
    num_classes: int
    X_tune: torch.Tensor | None = None
    Y_tune: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def load_split(
    path: str = "data",
    dataset: str = "Pedestrian",
    num_train: int = 32768 * 2,
    num_valid: int = 4096,
    num_tune: int = 4096,
    batch_size: int = 4096,
    seed: int = 123,
    transform: str = "quant",
    device: str = "cuda",
) -> Split:
    """Load, transform and cache one split on the device.

    The QUANT transform dominates setup, so it is applied once here and the
    features are reused by every variant in the sweep.

    The tune slice is taken from *after* the validation slice rather than out
    of the training indices, so that adding it leaves both the training set and
    the validation set byte-identical to every earlier sweep. Pedestrian has
    151,696 rows outside fold 0 against the 69,632 used here, so there is ample
    unseen data to take it from.
    """

    data = Dataset(
        f"{path}/{dataset}/{dataset}_X.npy", f"{path}/{dataset}/{dataset}_y.npy"
    )
    data.batch_size = batch_size

    num_classes = len(data.classes)

    fold = np.loadtxt(f"{path}/{dataset}/test_indices_fold_0.txt")
    indices = np.setdiff1d(np.arange(data.shape[0]), fold)

    np.random.seed(seed)
    np.random.shuffle(indices)

    def to_device(subset):
        return [
            (
                torch.as_tensor(X.astype(np.float32), device=device),
                torch.as_tensor(Y.astype(np.int64), device=device),
            )
            for X, Y in subset
        ]

    training = to_device(data[indices[:num_train]])
    validation = to_device(data[indices[num_train : num_train + num_valid]])

    start = num_train + num_valid
    tuning = to_device(data[indices[start : start + num_tune]]) if num_tune else []

    data.close()

    if transform == "quant":
        quant = Quant()
        quant.fit_transform(training[0][0])

        def features(subset):
            return torch.cat([quant.transform(X) for X, _ in subset])

        batches = [(quant.transform(X), Y) for X, Y in training]
    elif transform == "none":

        def features(subset):
            return torch.cat([X.reshape(X.shape[0], -1) for X, _ in subset])

        batches = [(X.reshape(X.shape[0], -1), Y) for X, Y in training]
    else:
        raise ValueError(f"unknown transform {transform!r}")

    X_va = features(validation)
    Y_va = torch.cat([Y for _, Y in validation])

    X_tune = features(tuning) if tuning else None
    Y_tune = torch.cat([Y for _, Y in tuning]) if tuning else None

    return Split(
        batches=batches,
        X_va=X_va,
        Y_va=Y_va,
        num_classes=num_classes,
        X_tune=X_tune,
        Y_tune=Y_tune,
        meta={
            "dataset": dataset,
            "fold": 0,
            "seed": seed,
            "n_tr": num_train,
            "n_va": num_valid,
            "n_tune": num_tune,
            "batch_size": batch_size,
            "num_classes": num_classes,
            "transform": transform,
            "num_features": batches[0][0].shape[1],
        },
    )


# == variants ==================================================================

# Each entry is kwargs for HashBoost, plus three keys the runner consumes
# itself: "estimators" (how many to bag), "overrides" (per-estimator kwarg
# patches) and "readout" (kwargs for a post-fit `fit_readout`).
# Notes record what screening already measured, so results stay comparable.
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    # measured win: 0.2124 vs 0.2205 +- 0.0029 baseline, same data passes
    "capacity_2": {"hashes_per_round": 2},
    "capacity_4": {"hashes_per_round": 4},
    # leaf-variance reduction over the bucket hypercube
    "smooth_0.1": {"neighbour_shrinkage": 0.1},
    "smooth_0.3": {"neighbour_shrinkage": 0.3},
    "smooth_0.6": {"neighbour_shrinkage": 0.6},
    "smooth_0.3_capacity_2": {"neighbour_shrinkage": 0.3, "hashes_per_round": 2},
    # bagging: the control for capacity_N at matched total hashes
    "bagged_2": {"estimators": 2},
    "bagged_4": {"estimators": 4},
    # combinations: bagging divides the quadratic round cost by E, so it buys
    # total hashes far more cheaply than a single deep model
    "bagged_4_capacity_2": {"estimators": 4, "hashes_per_round": 2},
    "smooth_0.3_bagged_4": {"neighbour_shrinkage": 0.3, "estimators": 4},
    "smooth_0.3_bagged_4_capacity_2": {
        "neighbour_shrinkage": 0.3,
        "estimators": 4,
        "hashes_per_round": 2,
    },
    # measured as noise or worse -- kept so the negative results are reproducible
    "leaf_l2_10": {"hessian_eps": 10.0},
    "bits_6": {"num_bits": 6},
    "lr_0.05": {"lr": 0.05},
    # -- heterogeneous bagging ------------------------------------------------
    # Members built from identical kwargs decorrelate through splitter noise
    # alone. The variance of an average falls with the correlation between its
    # members, so making them differ in *what they are* attacks the term that
    # bagging actually trades on. The ingredients are the ones already measured
    # as individually break-even (`bits_6`, `lr_0.05`): near-neutral quality is
    # what makes a good member, since the mixture should keep the mean and cut
    # the correlation. Watch the reported agreement, not just the error.
    "bagged_4_mixed_bits": {
        "estimators": 4,
        "overrides": [
            {"num_bits": 6},
            {"num_bits": 7},
            {"num_bits": 8},
            {"num_bits": 9},
        ],
    },
    "bagged_4_mixed_family": {
        "estimators": 4,
        "overrides": [{}, {"partitioner": ObliquePartitioner}],
    },
    "bagged_4_mixed_all": {
        "estimators": 4,
        "overrides": [
            {"num_bits": 7},
            {"partitioner": ObliquePartitioner},
            {"lr": 0.05},
            {"num_bits": 9, "partitioner": ObliquePartitioner},
        ],
    },
    "bagged_4_mixed_all_capacity_2": {
        "estimators": 4,
        "hashes_per_round": 2,
        "overrides": [
            {"num_bits": 7},
            {"partitioner": ObliquePartitioner},
            {"lr": 0.05},
            {"num_bits": 9, "partitioner": ObliquePartitioner},
        ],
    },
    # implemented and unit-tested since c1b64fe, never once measured
    "oblique": {"partitioner": ObliquePartitioner},
    # -- mass-adaptive neighbour shrinkage ------------------------------------
    # Fixed `neighbour_shrinkage` helps in two paired comparisons and hurts in
    # two, despite the mechanism working (non-zero leaves 31% -> 78%). That
    # signature says one global alpha is helping sparse buckets and damaging
    # dense ones; tau makes the borrowing proportional to how little evidence a
    # bucket has of its own.
    "adaptive_smooth_1": {"shrinkage_tau": 1.0},
    "adaptive_smooth_10": {"shrinkage_tau": 10.0},
    "adaptive_smooth_100": {"shrinkage_tau": 100.0},
    # the single-model sweep is monotone out to tau=100 without turning over,
    # so the optimum is past it; as tau -> inf every bucket is replaced by its
    # neighbour mean, which must eventually break
    "adaptive_smooth_300": {"shrinkage_tau": 300.0},
    "adaptive_smooth_1000": {"shrinkage_tau": 1000.0},
    "adaptive_smooth_3000": {"shrinkage_tau": 3000.0},
    # still descending at 3000. Typical bucket mass here is ~1e4 (65k rows x 50
    # epochs / 256 buckets), so these are the values at which dense buckets --
    # not just empty ones -- start borrowing, and where it must eventually break.
    "adaptive_smooth_10000": {"shrinkage_tau": 10000.0},
    "adaptive_smooth_30000": {"shrinkage_tau": 30000.0},
    "adaptive_smooth_100000": {"shrinkage_tau": 100000.0},
    "adaptive_smooth_1000000": {"shrinkage_tau": 1000000.0},
    "adaptive_smooth_10_bagged_4_capacity_2": {
        "shrinkage_tau": 10.0,
        "estimators": 4,
        "hashes_per_round": 2,
    },
    # tau=10 *hurt* the bagged+capacity model where it helped the plain one;
    # these check that against the tau that was actually best single-model,
    # and separate "bagging absorbs it" from "capacity absorbs it"
    "adaptive_smooth_100_bagged_4": {"shrinkage_tau": 100.0, "estimators": 4},
    "adaptive_smooth_100_capacity_2": {"shrinkage_tau": 100.0, "hashes_per_round": 2},
    "adaptive_smooth_100_bagged_4_capacity_2": {
        "shrinkage_tau": 100.0,
        "estimators": 4,
        "hashes_per_round": 2,
    },
    # tau=3000 was far better than tau=100 on the plain model, and smoothing
    # combined with capacity looked strong at the wrong tau -- so these are the
    # combinations at the tau the single-model sweep actually chose
    "adaptive_smooth_3000_capacity_2": {
        "shrinkage_tau": 3000.0,
        "hashes_per_round": 2,
    },
    "adaptive_smooth_3000_bagged_4": {"shrinkage_tau": 3000.0, "estimators": 4},
    "adaptive_smooth_3000_bagged_4_capacity_2": {
        "shrinkage_tau": 3000.0,
        "estimators": 4,
        "hashes_per_round": 2,
    },
    # -- refit readout --------------------------------------------------------
    # The leaf tables refit jointly against cross entropy rather than round by
    # round. The ladder runs cheapest first: if "round" (one gain per round)
    # buys nothing, the premise that the additive Newton readout is suboptimal
    # is wrong and the larger rungs are not worth running.
    "readout_round": {"readout": {"rung": "round"}},
    "readout_round_class": {"readout": {"rung": "round_class"}},
    "readout_table": {"readout": {"rung": "table", "lam": 0.1, "lr": 0.01}},
    "readout_round_bagged_4_capacity_2": {
        "estimators": 4,
        "hashes_per_round": 2,
        "readout": {"rung": "round"},
    },
    # Paired, "round" is worth exactly +0.0000 while "table" is worth +0.0078:
    # the Newton leaves are wrong *within* a round, not mis-weighted between
    # rounds. So the combination worth testing on the best ensemble is a table
    # refit, not the round rescaling measured above. "round_class" carries most
    # of the gain for 1/256th of the parameters, which matters here -- a table
    # over 6400 rounds is 537 MB before Adam's two moments.
    "readout_round_class_bagged_4_capacity_2": {
        "estimators": 4,
        "hashes_per_round": 2,
        "readout": {"rung": "round_class"},
    },
    "readout_table_bagged_4_capacity_2": {
        "estimators": 4,
        "hashes_per_round": 2,
        "readout": {"rung": "table", "lam": 0.1, "lr": 0.01},
    },
}


# == running ===================================================================


def evaluate(model, X: torch.Tensor, Y: torch.Tensor) -> float:

    return (model.predict(X).argmax(-1) != Y).float().mean().item()


def run_once(
    split: Split,
    config: dict[str, Any],
    epochs: int,
    seed: int,
    eval_every: int,
    device: str,
) -> dict[str, Any]:
    """Train one variant with one seed."""

    config = dict(config)
    estimators = config.pop("estimators", None)
    overrides = config.pop("overrides", None)
    readout = config.pop("readout", None)

    per_batch = config.get("hashes_per_round", 1)

    kwargs: dict[str, Any] = dict(
        num_classes=split.num_classes,
        max_num_hashes=epochs * len(split.batches) * per_batch + 1,
        device=device,
        **config,
    )

    torch.manual_seed(seed)

    model = (
        BaggedHashBoost(num_estimators=estimators, overrides=overrides, **kwargs)
        if estimators
        else HashBoost(**kwargs)
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    wall, cpu = time.perf_counter(), time.process_time()

    curve_x: list[int] = []
    curve_y: list[float] = []

    for epoch in range(epochs):
        for X, Y in split.batches:
            model.fit_batch(X, Y)

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            curve_x.append(model.num_rounds)
            curve_y.append(evaluate(model, split.X_va, split.Y_va))

    boosted_final = curve_y[-1]
    readout_info = None

    if readout is not None:
        # The readout is fit on the training data it was boosted on, early
        # stopped on the tune slice, and only then scored on X_va -- which is
        # never used to choose anything.
        fitted, readout_info = fit_readout(
            model,
            torch.cat([X for X, _ in split.batches]),
            torch.cat([Y for _, Y in split.batches]),
            split.X_tune,
            split.Y_tune,
            **readout,
        )

        curve_x.append(model.num_rounds)
        curve_y.append(evaluate(fitted, split.X_va, split.Y_va))

        del fitted

    # how often two members pick the same class: the mechanism heterogeneous
    # bagging is supposed to move, which the +-0.003 noise floor makes hard to
    # read from the error alone
    agreement = (
        model.estimator_agreement(split.X_va)
        if isinstance(model, BaggedHashBoost)
        else None
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    result = {
        "seed": seed,
        "final": curve_y[-1],
        "best": min(curve_y),
        "boosted_final": boosted_final,
        "agreement": agreement,
        "readout": readout_info,
        "rounds": model.num_rounds,
        "wall_s": time.perf_counter() - wall,
        "cpu_s": time.process_time() - cpu,
        "peak_mb": (
            torch.cuda.max_memory_allocated() / 1e6
            if device.startswith("cuda")
            else 0.0
        ),
        "curve": {"x": curve_x, "y": curve_y},
    }

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return result


def run_variant(
    split: Split,
    name: str,
    config: dict[str, Any],
    epochs: int,
    seeds: int,
    eval_every: int,
    device: str,
) -> dict[str, Any]:

    runs = [
        run_once(split, config, epochs, seed, eval_every, device)
        for seed in range(seeds)
    ]

    finals = [r["final"] for r in runs]
    agreements = [r["agreement"] for r in runs if r["agreement"] is not None]

    return {
        "params": config,
        "x_name": "round",
        "runs": runs,
        "timings": [{"wall_s": r["wall_s"], "cpu_s": r["cpu_s"]} for r in runs],
        "summary": {
            "mean": statistics.mean(finals),
            "sd": statistics.stdev(finals) if len(finals) > 1 else 0.0,
            "best": min(finals),
            "boosted_mean": statistics.mean(r["boosted_final"] for r in runs),
            "agreement": statistics.mean(agreements) if agreements else None,
            "rounds": runs[0]["rounds"],
            "wall_s": statistics.mean(r["wall_s"] for r in runs),
            "peak_mb": max(r["peak_mb"] for r in runs),
        },
    }


# == main ======================================================================


def commit_hash() -> str:

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data")
    parser.add_argument("--dataset", default="Pedestrian")
    parser.add_argument("--transform", default="quant", choices=("quant", "none"))
    parser.add_argument("--variants", default="baseline")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-train", type=int, default=32768 * 2)
    parser.add_argument("--out", default="results")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.list:
        for name, config in VARIANTS.items():
            print(f"{name:24s} {config}")
        return

    names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; try --list")

    split = load_split(
        path=args.path,
        dataset=args.dataset,
        num_train=args.num_train,
        batch_size=args.batch_size,
        transform=args.transform,
        device=args.device,
    )

    print(
        f"{args.dataset}: {len(split.batches)} batches x {tuple(split.batches[0][0].shape)}, "
        f"{split.num_classes} classes, transform={args.transform}, "
        f"{args.seeds} seeds x {args.epochs} epochs\n"
    )
    print(
        f"{'variant':30s} {'val err (mean+-sd)':>22s} {'agree':>7s} "
        f"{'rounds':>7s} {'wall':>8s} {'peak':>8s}"
    )

    out = Path(args.out) / f"{args.dataset}-sweep-{commit_hash()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    models: dict[str, Any] = {}

    def write() -> None:
        """Serialise what has finished so far.

        Called after every variant, not once at the end: a sweep is tens of
        minutes of GPU time and the expensive variants run last, so a crash
        there (an OOM on a wide ensemble, say) would otherwise discard every
        result before it.
        """

        out.write_text(
            json.dumps(
                {
                    "commit": commit_hash(),
                    "dataset": args.dataset,
                    "device": args.device,
                    "split": split.meta,
                    "epochs": args.epochs,
                    "seeds": args.seeds,
                    "models": models,
                },
                indent=2,
                # variant params may hold a partitioner class, which has no
                # JSON form; its repr is what a reader of the file wants anyway
                default=str,
            )
        )

    for name in names:
        entry = run_variant(
            split,
            name,
            VARIANTS[name],
            args.epochs,
            args.seeds,
            args.eval_every,
            args.device,
        )
        models[name] = entry
        write()

        s = entry["summary"]
        agreement = f"{s['agreement']:7.3f}" if s["agreement"] is not None else " " * 7
        print(
            f"{name:30s} {s['mean']:12.4f} +- {s['sd']:.4f} {agreement} "
            f"{s['rounds']:7d} {s['wall_s']:7.1f}s {s['peak_mb']:7.0f}M",
            flush=True,
        )

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
