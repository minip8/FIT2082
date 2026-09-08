"""Memory-lean XGBoost baseline on QUANT features.

The dense QUANT matrix for LenDB is 65,536 x 14,940 float32 -- 3.9 GB. Building
a DMatrix from it needs the source and a second copy to coexist, which does not
fit on an 8 GB card next to the desktop; that is why the xgboost entry is
commented out of the LenDB write in `notebooks/compare.ipynb` while every other
dataset has one.

Nothing here holds the full feature matrix. A `DataIter` transforms one raw
batch at a time and hands it straight to `QuantileDMatrix`, which keeps only the
binned ellpack (about one byte per element, ~1 GB) and discards each batch as it
goes. Peak device usage is the ellpack plus a single batch rather than the
matrix plus a copy of it.

The split is reproduced exactly as the notebook draws it -- same fold, same
`np.random.seed(42)` shuffle -- so the curves are comparable to the hashboost
run already in `results/`.

    uv run python scripts/xgboost_baseline.py --dataset LenDB
"""

import argparse
import gc
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xgboost as xgb

from fit2082.demo.utils import Dataset
from fit2082.quant.quant import Quant

# == data ======================================================================


def load_raw(
    path: str, dataset: str, n_tr: int, n_va: int, n_te: int, batch_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Draw the notebook's split and return the *raw* series, not features.

    Raw LenDB rows are 3 x 540 float32, so the training subset is 425 MB -- it
    is only the QUANT expansion to 14,940 features that is large. Keeping the
    raw data and expanding on demand is what makes the rest of this fit.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    data = Dataset(
        f"{path}/{dataset}/{dataset}_X.npy", f"{path}/{dataset}/{dataset}_y.npy"
    )
    data.batch_size = batch_size

    num_classes = len(data.classes)

    fold = np.loadtxt(f"{path}/{dataset}/test_indices_fold_0.txt")
    ix = np.setdiff1d(np.arange(data.shape[0]), fold)

    np.random.shuffle(ix)

    def collect(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs, Ys = zip(*((X, Y) for X, Y in data[indices]))
        return np.concatenate(Xs).astype(np.float32), np.concatenate(Ys)

    X_tr, Y_tr = collect(ix[:n_tr])
    X_va, Y_va = collect(ix[n_tr : n_tr + n_va])

    data.close()

    return X_tr, Y_tr.astype(np.int32), X_va, Y_va.astype(np.int32), num_classes


class QuantBatches(xgb.DataIter):
    """Streams QUANT features to XGBoost one batch at a time.

    `QuantileDMatrix` makes more than one pass over the iterator, and the
    features are recomputed on every pass rather than cached: the cache would be
    the 3.9 GB matrix this path exists to avoid. The transform is a few hundred
    milliseconds a batch on the GPU, so the recompute is far cheaper than the
    memory it saves.
    """

    def __init__(
        self,
        X_raw: np.ndarray,
        Y: np.ndarray,
        quant: Quant,
        batch_size: int,
        device: str,
    ) -> None:

        self._X_raw = X_raw
        self._Y = Y
        self._quant = quant
        self._batch_size = batch_size
        self._device = device
        self._i = 0
        self.passes = 0

        super().__init__(release_data=True)

    def reset(self) -> None:

        self._i = 0
        self.passes += 1

    def next(self, input_data: Any) -> bool:

        if self._i >= self._X_raw.shape[0]:
            return False

        stop = self._i + self._batch_size

        X = torch.as_tensor(self._X_raw[self._i : stop], device=self._device)
        Z = self._quant.transform(X)

        input_data(data=Z, label=self._Y[self._i : stop])

        self._i = stop

        del X, Z

        return True


# == memory ====================================================================


def gpu_used_mb(device: str) -> float:
    """Whole-process device usage, not just torch's.

    XGBoost allocates through its own CUDA allocator, so `max_memory_allocated`
    cannot see the ellpack or the histograms -- the number that matters here.
    `mem_get_info` reads the driver, which sees both.
    """

    if not device.startswith("cuda"):
        return 0.0

    free, total = torch.cuda.mem_get_info()

    return (total - free) / 1e6


class MemoryProbe(xgb.callback.TrainingCallback):
    """Samples device usage once a round, to record the training peak."""

    def __init__(self, device: str) -> None:

        self.device = device
        self.peak_mb = gpu_used_mb(device)

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:

        self.peak_mb = max(self.peak_mb, gpu_used_mb(self.device))

        return False


def host_peak_mb() -> float:

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# == results ===================================================================


def commit_hash() -> str:

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def curve(values: list[float]) -> dict[str, list]:

    return {"x": list(range(len(values))), "y": [float(v) for v in values]}


def write_results(out: Path, entry: dict[str, Any], info: dict[str, Any]) -> None:
    """Merge one model entry into the dataset's results file.

    Read-modify-write rather than overwrite, so a later run of another model --
    or a rerun of this one -- accumulates into the same file the way the
    notebook's single `write_results` call does.
    """

    payload: dict[str, Any] = {}

    if out.exists():
        payload = json.loads(out.read_text())

    payload.update({k: v for k, v in info.items() if k != "models"})
    payload.setdefault("models", {}).update(entry)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))


# == main ======================================================================


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data")
    parser.add_argument("--dataset", default="LenDB")
    parser.add_argument("--out", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-tr", type=int, default=32768 * 2)
    parser.add_argument("--n-va", type=int, default=4096)
    parser.add_argument("--n-te", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--iter-batch-size", type=int, default=4096)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--merge-from", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = args.device

    print(f"loading {args.dataset} (raw series, features built on the fly)", flush=True)

    load_wall = time.perf_counter()

    X_tr_raw, Y_tr, X_va_raw, Y_va, num_classes = load_raw(
        args.path,
        args.dataset,
        args.n_tr,
        args.n_va,
        args.n_te,
        args.batch_size,
        args.seed,
    )

    print(
        f"  raw train {X_tr_raw.shape} {X_tr_raw.nbytes / 1e9:.2f} GB, "
        f"raw valid {X_va_raw.shape}, {num_classes} classes "
        f"({time.perf_counter() - load_wall:.1f}s)",
        flush=True,
    )

    # QUANT carries no fitted state -- the intervals come from the series length
    # alone -- so fitting on one row gives byte-identical features to fitting on
    # all 65,536, without building the matrix that would not fit.
    probe_row = torch.as_tensor(X_tr_raw[:1], device=device)

    quant = Quant()
    quant.fit_transform(probe_row)

    num_features = quant.transform(probe_row).shape[1]

    dense_gb = X_tr_raw.shape[0] * num_features * 4 / 1e9
    print(
        f"  QUANT -> {num_features} features "
        f"(dense train matrix would be {dense_gb:.2f} GB; not materialised)",
        flush=True,
    )

    xgb_params: dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "hist",
        "device": device,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "max_bin": args.max_bin,
        "random_state": args.seed,
        "eval_metric": "merror",
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
    }

    torch.cuda.empty_cache() if device.startswith("cuda") else None

    baseline_mb = gpu_used_mb(device)

    build_wall, build_cpu = time.perf_counter(), time.process_time()

    train_iter = QuantBatches(X_tr_raw, Y_tr, quant, args.iter_batch_size, device)
    dtrain = xgb.QuantileDMatrix(train_iter, max_bin=args.max_bin)

    # ref=dtrain reuses the training cuts, so validation does not sketch its own
    valid_iter = QuantBatches(X_va_raw, Y_va, quant, args.iter_batch_size, device)
    dvalid = xgb.QuantileDMatrix(valid_iter, max_bin=args.max_bin, ref=dtrain)

    # the notebook's `timings[name]` is a list of *reruns* of one cell; here the
    # two entries are the two phases of a single run, so they are labelled
    build_time = {
        "phase": "build",
        "wall_s": time.perf_counter() - build_wall,
        "cpu_s": time.process_time() - build_cpu,
    }
    build_mb = gpu_used_mb(device)

    passes = train_iter.passes

    print(
        f"  built ellpack in {build_time['wall_s']:.1f}s over "
        f"{passes} passes; device {baseline_mb:.0f} -> {build_mb:.0f} MB",
        flush=True,
    )

    # hand back the scratch torch cached while transforming batches: xgboost
    # allocates through its own CUDA allocator and cannot see, let alone reuse,
    # blocks torch is holding -- the "Free memory: 0B" failure mode
    del train_iter, valid_iter
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    freed_mb = gpu_used_mb(device)
    print(
        f"  released torch scratch; device {build_mb:.0f} -> {freed_mb:.0f} MB",
        flush=True,
    )

    evals_result: dict[str, Any] = {}
    probe = MemoryProbe(device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    train_wall, train_cpu = time.perf_counter(), time.process_time()

    booster = xgb.train(
        {
            k: v
            for k, v in xgb_params.items()
            if k not in ("num_boost_round", "early_stopping_rounds")
        },
        dtrain,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        evals=[(dtrain, "tr"), (dvalid, "va")],
        evals_result=evals_result,
        callbacks=[probe],
        verbose_eval=False,
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    train_time = {
        "phase": "train",
        "wall_s": time.perf_counter() - train_wall,
        "cpu_s": time.process_time() - train_cpu,
    }

    tr = evals_result["tr"]["merror"]
    va = evals_result["va"]["merror"]

    print(
        f"  trained {len(va)} rounds in {train_time['wall_s']:.1f}s  "
        f"tr={tr[-1]:.4f} va={va[-1]:.4f} best_va={min(va):.4f} "
        f"(iter {booster.best_iteration})",
        flush=True,
    )
    print(
        f"  peak device {probe.peak_mb:.0f} MB of "
        f"{torch.cuda.mem_get_info()[1] / 1e6:.0f} MB, host RSS {host_peak_mb():.0f} MB"
        if device.startswith("cuda")
        else f"  host RSS {host_peak_mb():.0f} MB",
        flush=True,
    )

    entry = {
        "xgboost": {
            "x_name": "round",
            "params": xgb_params,
            "timings": [build_time, train_time],
            "memory": {
                "device_baseline_mb": baseline_mb,
                "device_after_build_mb": build_mb,
                "device_before_train_mb": freed_mb,
                "device_peak_mb": probe.peak_mb,
                "device_total_mb": (
                    torch.cuda.mem_get_info()[1] / 1e6
                    if device.startswith("cuda")
                    else 0.0
                ),
                "host_peak_rss_mb": host_peak_mb(),
                "dense_train_matrix_gb": dense_gb,
                "streamed": True,
                "iter_batch_size": args.iter_batch_size,
                "build_passes": passes,
            },
            "best_iteration": int(booster.best_iteration),
            "results": {"tr": {"merror": curve(tr)}, "va": {"merror": curve(va)}},
        }
    }

    info = {
        "commit": commit_hash(),
        "dataset": args.dataset,
        "device": device,
        "split": {
            "fold": 0,
            "seed": args.seed,
            "n_tr": args.n_tr,
            "n_va": args.n_va,
            "n_te": args.n_te,
            "n_eval_subsample": args.n_va,
            "batch_size": args.batch_size,
            "num_classes": num_classes,
        },
        "transform": {
            "name": "quant",
            "depth": quant.depth,
            "div": quant.div,
            "num_features": num_features,
        },
    }

    out = Path(args.out) / f"{args.dataset}-{commit_hash()}.json"

    if args.merge_from:
        # carry an earlier run's models into this file so one file holds the
        # whole comparison, tagged with the commit they were actually run at
        source = json.loads(Path(args.merge_from).read_text())
        for name, model in source.get("models", {}).items():
            if name not in entry:
                entry[name] = {**model, "source_commit": source.get("commit")}

    write_results(out, entry, info)

    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
