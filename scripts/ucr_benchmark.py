"""Default HashBoost against ExtraTrees and XGBoost on the UCR archive.

Each dataset in `fit2082.ucr.UCR112` runs on its default train/test split.
QUANT runs once on the GPU, and then every model trains on the same features:

* `hashboost`: the MONSTER default (8 bits, lr 0.1) for a fixed 800 rounds,
  in batches of at most 4,096 rows cycled until the budget is spent. `--bits`
  and `--rounds` change it, and `--label` names its entry, so that several
  configurations can share one results file.
* `extratrees`: QUANT's own classifier (200 trees, entropy, max_features 0.1),
  on the host.
* `xgboost`: library defaults (100 rounds, eta 0.3, depth 6), on the GPU.

Nothing is tuned or early-stopped: there is no validation split to choose on,
so the reported number is the test error of the finished model. Results go to
`results/UCR/<Name>-<commit>.json`, one file per dataset in the shared schema
(`fit2082.results`), written after every model so a crash late in the run
costs nothing before it.

    uv run python scripts/ucr_benchmark.py --compile
    uv run python scripts/ucr_benchmark.py --datasets GunPoint,Wafer --models hashboost
    uv run python scripts/ucr_benchmark.py --compile --skip-done    # resume a run
    uv run python scripts/ucr_benchmark.py --compile --models hashboost \
        --bits 4 --rounds 3200 --label hashboost_b4_r3200

`--stdin` takes the same arguments on a pipe; see `fit2082.cli`.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fit2082.cli import parse_args
from fit2082.results import commit_hash, write_results
from fit2082.ucr import (
    DEFAULT_PATH,
    SEED,
    UCR112,
    fit_extratrees,
    fit_hashboost,
    fit_xgboost,
    load_ucr,
    quant_features,
)

MODELS = ("hashboost", "extratrees", "xgboost")

# == helpers ===================================================================


def resolve(name: str) -> str:
    """Match a dataset name case-insensitively, or exit with the near misses."""

    for known in UCR112:
        if known.lower() == name.lower():
            return known

    close = [k for k in UCR112 if name.lower() in k.lower()]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    sys.exit(f"{name!r} is not in UCR112.{hint}")


def warm_up(device: str, num_bits: int) -> None:
    """Compile the hash encoding on throwaway data before any timed fit.

    Compilation takes several seconds on first use and would otherwise be
    charged to whichever dataset happens to come first. Shapes are dynamic, so
    the graphs built here serve every dataset, but the encoder loops once per
    bit, so they are specialised to one bit width: compile at the run's own.
    """

    Z = torch.randn(64, 16, device=device)
    y = np.arange(64) % 2

    fit_hashboost(
        Z,
        y,
        Z,
        y,
        num_classes=2,
        rounds=4,
        device=device,
        compile=True,
        num_bits=num_bits,
    )


def fit_seconds(entry: dict[str, Any]) -> float:

    return next(t["wall_s"] for t in entry["timings"] if t["phase"] == "fit")


# == main ======================================================================


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument(
        "--datasets", default="all", help="comma-separated names, or 'all'"
    )
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--bits", type=int, default=8, help="HashBoost's num_bits")
    parser.add_argument(
        "--label",
        default="hashboost",
        help="the key HashBoost's entry is written under, so that several of its "
        "configurations can share a dataset's results file",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--out", default="results/UCR")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile HashBoost's hash encoding and leaf refresh (faster; "
        "leaves may differ in the last 2 ulps)",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="skip models already in this commit's file for a dataset",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parse_args(parser)

    names = (
        list(UCR112)
        if args.datasets == "all"
        else [resolve(n) for n in args.datasets.split(",")]
    )

    models = args.models.split(",")
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        parser.error(f"unknown models {unknown}; choose from {', '.join(MODELS)}")

    commit = commit_hash()
    out_dir = Path(args.out)

    if args.compile and "hashboost" in models:
        began = time.perf_counter()
        warm_up(args.device, args.bits)
        print(f"compiled the hash encoding ({time.perf_counter() - began:.1f}s)\n")

    print(f"{len(names)} datasets, models: {', '.join(models)}, writing {out_dir}/")

    errors: dict[str, list[float]] = {m: [] for m in models}
    seconds: dict[str, float] = {m: 0.0 for m in models}

    for i, name in enumerate(names, 1):
        out = out_dir / f"{name}-{commit}.json"

        done: set[str] = set()
        if args.skip_done and out.exists():
            done = set(json.loads(out.read_text()).get("models", {}))

        # a model's entry is keyed by its name, except HashBoost's, which takes
        # --label so that configurations do not overwrite each other
        keys = {m: args.label if m == "hashboost" else m for m in models}

        todo = [m for m in models if keys[m] not in done]
        if not todo:
            print(f"{i:3d}/{len(names)} {name:30s} done")
            continue

        data = load_ucr(name, args.path)
        num_classes = len(data.classes)

        Z_tr, Z_te, transform = quant_features(data.X_tr, data.X_te, args.device)

        info = {
            "commit": commit,
            "dataset": name,
            "device": args.device,
            "split": {
                "source": "UCR 2018",
                "split": "default train/test",
                "seed": SEED,
                "n_tr": int(data.X_tr.shape[0]),
                "n_te": int(data.X_te.shape[0]),
                "length": int(data.X_tr.shape[-1]),
                "num_classes": num_classes,
            },
            "transform": transform,
        }

        line = (
            f"{i:3d}/{len(names)} {name:30s} {data.X_tr.shape[0]:5d}/"
            f"{data.X_te.shape[0]:<5d} L={data.X_tr.shape[-1]:<4d} "
            f"k={num_classes:<2d} p={transform['num_features']:<5d} |"
        )

        for model in todo:
            if model == "hashboost":
                entry = fit_hashboost(
                    Z_tr,
                    data.y_tr,
                    Z_te,
                    data.y_te,
                    num_classes,
                    rounds=args.rounds,
                    batch_size=args.batch_size,
                    device=args.device,
                    compile=args.compile,
                    num_bits=args.bits,
                )
            elif model == "extratrees":
                entry = fit_extratrees(Z_tr, data.y_tr, Z_te, data.y_te)
            else:
                entry = fit_xgboost(Z_tr, data.y_tr, Z_te, data.y_te, args.device)

            write_results(out, {keys[model]: entry}, info)

            errors[model].append(entry["final"]["te"])
            seconds[model] += fit_seconds(entry)

            line += (
                f" {keys[model]} {entry['final']['te']:.4f} ({fit_seconds(entry):.1f}s)"
            )

        print(line, flush=True)

        del Z_tr, Z_te
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\nthis invocation, test error over the datasets each model ran on:")
    for model in models:
        if errors[model]:
            name = args.label if model == "hashboost" else model
            print(
                f"  {name:10s} mean {statistics.mean(errors[model]):.4f} over "
                f"{len(errors[model])} datasets, {seconds[model]:.0f}s fitting"
            )


if __name__ == "__main__":
    main()
