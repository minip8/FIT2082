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

from fit2082.boost import BaggedHashBoost, HashBoost
from fit2082.demo.utils import Dataset
from fit2082.quant.quant import Quant

# == data ======================================================================


@dataclass
class Split:
    """Cached, device-resident features for one train/validation split."""

    batches: list[tuple[torch.Tensor, torch.Tensor]]
    X_va: torch.Tensor
    Y_va: torch.Tensor
    num_classes: int
    meta: dict[str, Any] = field(default_factory=dict)


def load_split(
    path: str = "data",
    dataset: str = "Pedestrian",
    num_train: int = 32768 * 2,
    num_valid: int = 4096,
    batch_size: int = 4096,
    seed: int = 123,
    transform: str = "quant",
    device: str = "cuda",
) -> Split:
    """Load, transform and cache one split on the device.

    The QUANT transform dominates setup, so it is applied once here and the
    features are reused by every variant in the sweep.
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

    data.close()

    if transform == "quant":
        quant = Quant()
        quant.fit_transform(training[0][0])

        batches = [(quant.transform(X), Y) for X, Y in training]
        X_va = torch.cat([quant.transform(X) for X, _ in validation])
    elif transform == "none":
        batches = [(X.reshape(X.shape[0], -1), Y) for X, Y in training]
        X_va = torch.cat([X.reshape(X.shape[0], -1) for X, _ in validation])
    else:
        raise ValueError(f"unknown transform {transform!r}")

    Y_va = torch.cat([Y for _, Y in validation])

    return Split(
        batches=batches,
        X_va=X_va,
        Y_va=Y_va,
        num_classes=num_classes,
        meta={
            "dataset": dataset,
            "fold": 0,
            "seed": seed,
            "n_tr": num_train,
            "n_va": num_valid,
            "batch_size": batch_size,
            "num_classes": num_classes,
            "transform": transform,
            "num_features": batches[0][0].shape[1],
        },
    )


# == variants ==================================================================

# Each entry is kwargs for HashBoost, plus optional "estimators" to bag.
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
    # measured as noise or worse -- kept so the negative results are reproducible
    "leaf_l2_10": {"hessian_eps": 10.0},
    "bits_6": {"num_bits": 6},
    "lr_0.05": {"lr": 0.05},
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

    per_batch = config.get("hashes_per_round", 1)

    kwargs: dict[str, Any] = dict(
        num_classes=split.num_classes,
        max_num_hashes=epochs * len(split.batches) * per_batch + 1,
        device=device,
        **config,
    )

    torch.manual_seed(seed)

    model = (
        BaggedHashBoost(num_estimators=estimators, **kwargs)
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

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    result = {
        "seed": seed,
        "final": curve_y[-1],
        "best": min(curve_y),
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

    return {
        "params": config,
        "x_name": "round",
        "runs": runs,
        "timings": [{"wall_s": r["wall_s"], "cpu_s": r["cpu_s"]} for r in runs],
        "summary": {
            "mean": statistics.mean(finals),
            "sd": statistics.stdev(finals) if len(finals) > 1 else 0.0,
            "best": min(finals),
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
        f"{'variant':24s} {'val err (mean+-sd)':>22s} {'rounds':>7s} {'wall':>8s} {'peak':>8s}"
    )

    models: dict[str, Any] = {}

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

        s = entry["summary"]
        print(
            f"{name:24s} {s['mean']:12.4f} +- {s['sd']:.4f} {s['rounds']:7d} "
            f"{s['wall_s']:7.1f}s {s['peak_mb']:7.0f}M",
            flush=True,
        )

    out = Path(args.out) / f"{args.dataset}-sweep-{commit_hash()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
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
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
