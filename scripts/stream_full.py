"""Train HashBoost and XGBoost on the *whole* LenDB training pool by streaming.

The earlier runs use 65,536 rows because that is what fits. The full pool is
975,291 rows: 6.3 GB as raw series against 7.9 GB of host RAM, and 58 GB once
QUANT expands it to 14,940 features. Nothing here is ever fully resident --
batches are read from the .npy memmap, transformed on the GPU, and dropped.

The two models meet that constraint very differently, which is the comparison
worth having:

* HashBoost is an online learner. `fit_batch` folds one batch into the model
  and the batch is then free, so peak memory is one batch no matter how much
  data streams past. What it pays instead is time: every batch updates the
  buckets of all existing rounds, so total work is quadratic in rounds.
* XGBoost needs the whole binned matrix available for every boosting round.
  Streaming only changes where that matrix lives -- `ExtMemQuantileDMatrix`
  writes the ellpack to disk once and pages it back on each round.

The validation and test rows are held byte-identical to the 65,536-row runs, so
the curves here are directly comparable to `results/LenDB-*.json`; the training
pool is everything else.

    uv run python scripts/stream_full.py --model hashboost --epochs 5
    uv run python scripts/stream_full.py --model xgboost --num-boost-round 50
"""

import argparse
import mmap
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xgboost as xgb

from fit2082.boost import HashBoost
from fit2082.quant.quant import Quant
from fit2082.results import (
    commit_hash,
    curve,
    gpu_total_mb,
    gpu_used_mb,
    host_available_mb,
    host_peak_rss_mb,
    write_results,
)

# == split =====================================================================


def split_indices(
    path: str, dataset: str, seed: int, n_ref: int, n_va: int, n_te: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The notebook's split, with everything it left on the floor added to train.

    `notebooks/compare.ipynb` shuffles the non-fold-0 rows and takes the first
    `n_ref` for training, then `n_va` and `n_te`. Reproducing that shuffle and
    keeping the same validation and test slices means the only thing that
    changes here is the size of the training set -- the numbers stay comparable
    to every earlier LenDB run.
    """

    np.random.seed(seed)

    Y = np.load(f"{path}/{dataset}/{dataset}_y.npy", mmap_mode="r")
    fold = np.loadtxt(f"{path}/{dataset}/test_indices_fold_0.txt")

    ix = np.setdiff1d(np.arange(Y.shape[0]), fold)

    np.random.shuffle(ix)

    va = ix[n_ref : n_ref + n_va]
    te = ix[n_ref + n_va : n_ref + n_va + n_te]

    # the reference run's training rows, plus everything it never looked at
    tr = np.concatenate([ix[:n_ref], ix[n_ref + n_va + n_te :]])

    return tr, va, te


# == streaming =================================================================


class RawStream:
    """Batches of raw series read straight from the .npy memmap.

    Indices are permuted globally each epoch and then sorted *within* each
    batch. That leaves the batch the same set of rows, but turns the memmap
    reads from scattered into near-sequential: measured on LenDB with a cold
    cache, 593 ms/batch down to 64 ms/batch. Over a 238-batch epoch that is the
    difference between 182 s and 51 s of streaming.
    """

    def __init__(
        self,
        path_X: str,
        path_Y: str,
        indices: np.ndarray,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        drop_cache_every: int = 50,
    ) -> None:

        self._X = np.load(path_X, mmap_mode="r")
        self._Y = np.load(path_Y, mmap_mode="r")

        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_cache_every = drop_cache_every

        self._rng = np.random.default_rng(seed)

    def drop_cache(self) -> None:
        """Hand the pages we have already consumed back to the kernel.

        An epoch pulls 6.3 GB of a 8.1 GB file through the page cache, which on
        a 7.9 GB machine leaves `free` at a few hundred MB. The pages are clean
        and reclaimable, so nothing is actually short of memory -- but enough
        things watch `free` rather than `available` that the run gets killed
        anyway. MADV_DONTNEED drops them outright; re-reading costs the 64
        ms/batch that the sorted access pattern already assumes.
        """

        self._X._mmap.madvise(mmap.MADV_DONTNEED)

    def __len__(self) -> int:

        return int(np.ceil(self.indices.shape[0] / self.batch_size))

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:

        order = self._rng.permutation(self.indices) if self.shuffle else self.indices

        for i, start in enumerate(range(0, order.shape[0], self.batch_size)):
            batch = np.sort(order[start : start + self.batch_size])

            yield np.array(self._X[batch], dtype=np.float32), np.array(self._Y[batch])

            if self.drop_cache_every and (i + 1) % self.drop_cache_every == 0:
                self.drop_cache()

    def gather(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Read one fixed set of rows, for the held-out slices."""

        order = np.sort(indices)

        return np.array(self._X[order], dtype=np.float32), np.array(self._Y[order])


def fit_quant(stream: RawStream, device: str) -> tuple[Quant, int]:
    """QUANT carries no data-dependent state, so one row fixes the transform."""

    raw, _ = stream.gather(stream.indices[:1])

    quant = Quant()
    quant.fit_transform(torch.as_tensor(raw, device=device))

    return quant, quant.transform(torch.as_tensor(raw, device=device)).shape[1]


def features(raw: np.ndarray, quant: Quant, device: str) -> torch.Tensor:

    return quant.transform(torch.as_tensor(raw, device=device))


# == hashboost =================================================================


def run_hashboost(
    stream: RawStream,
    quant: Quant,
    holdouts: dict[str, tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    args: argparse.Namespace,
    device: str,
    checkpoint: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:

    total_rounds = args.epochs * len(stream)

    params = {
        "num_classes": num_classes,
        "num_bits": args.num_bits,
        "lr": args.lr,
        "max_num_hashes": total_rounds + 1,
        "hashes_per_round": 1,
        "max_epochs": args.epochs,
    }

    torch.manual_seed(args.seed)

    model = HashBoost(
        num_classes=num_classes,
        num_bits=args.num_bits,
        lr=args.lr,
        max_num_hashes=total_rounds + 1,
        hashes_per_round=1,
        device=device,
    )

    print(
        f"  {len(stream)} batches/epoch x {args.epochs} epochs = {total_rounds} rounds",
        flush=True,
    )

    records: dict[str, list[tuple[int, float]]] = {"tr": [], "va": []}
    peak_mb = gpu_used_mb(device)
    stream_s = 0.0
    rounds = 0

    def evaluate() -> None:
        with torch.no_grad():
            for name, (X, Y) in holdouts.items():
                error = (model.predict(X).argmax(-1) != Y).to(torch.float32).mean()
                records[name].append((rounds, error.item()))

    def entry(elapsed: dict[str, Any]) -> dict[str, Any]:
        return {
            "x_name": "round",
            "params": params,
            "timings": [elapsed],
            "memory": {
                "device_peak_mb": peak_mb,
                "device_total_mb": gpu_total_mb(device),
                "host_peak_rss_mb": host_peak_rss_mb(),
                "streamed": True,
                "batch_size": stream.batch_size,
                "resident": "one batch",
            },
            "rounds": rounds,
            "stream_s": stream_s,
            "results": {
                name: {"merror": curve([e for _, e in rec], [r for r, _ in rec])}
                for name, rec in records.items()
            },
        }

    wall, cpu = time.perf_counter(), time.process_time()

    for _ in range(args.epochs):
        for raw, y in stream:
            read = time.perf_counter()
            Z = features(raw, quant, device)
            Y = torch.as_tensor(y.astype(np.int64), device=device)
            stream_s += time.perf_counter() - read

            model.fit_batch(Z, Y)
            rounds += 1

            del Z, Y

            if rounds % args.eval_every == 0 or rounds == total_rounds:
                evaluate()
                peak_mb = max(peak_mb, gpu_used_mb(device))

                # a streamed run is long and has already been killed once by
                # a low-memory watchdog; write what exists after every eval
                checkpoint(
                    entry(
                        {
                            "phase": "train",
                            "wall_s": time.perf_counter() - wall,
                            "cpu_s": time.process_time() - cpu,
                        }
                    )
                )

                print(
                    f"    round {rounds:5d}/{total_rounds}  "
                    f"tr={records['tr'][-1][1]:.4f} va={records['va'][-1][1]:.4f}  "
                    f"{time.perf_counter() - wall:6.0f}s  gpu {gpu_used_mb(device):5.0f} MB"
                    f"  rss {host_peak_rss_mb():5.0f} MB  avail {host_available_mb():5.0f} MB",
                    flush=True,
                )

    elapsed = {
        "phase": "train",
        "wall_s": time.perf_counter() - wall,
        "cpu_s": time.process_time() - cpu,
    }

    va = [e for _, e in records["va"]]

    print(
        f"  {rounds} rounds in {elapsed['wall_s']:.0f}s "
        f"({stream_s:.0f}s of it streaming)  va_final={va[-1]:.4f} va_best={min(va):.4f}",
        flush=True,
    )

    return entry(elapsed)


# == xgboost ===================================================================


class StreamedQuant(xgb.DataIter):
    """Feeds the streamed QUANT features to XGBoost's external-memory builder.

    `on_host=False` sends the cached ellpack pages to `cache_prefix` on disk:
    the full pool bins to roughly 40 GB, which fits neither the 8 GB card nor
    the 7.9 GB of host RAM.
    """

    def __init__(
        self, stream: RawStream, quant: Quant, device: str, cache_prefix: str
    ) -> None:

        self._stream = stream
        self._quant = quant
        self._device = device
        self._it: Iterator[tuple[np.ndarray, np.ndarray]] | None = None
        self.passes = 0

        super().__init__(cache_prefix=cache_prefix, release_data=True, on_host=False)

    def reset(self) -> None:

        self._it = None
        self.passes += 1

    def next(self, input_data: Any) -> bool:

        if self._it is None:
            self._it = iter(self._stream)

        batch = next(self._it, None)

        if batch is None:
            return False

        raw, y = batch
        Z = features(raw, self._quant, self._device)

        input_data(data=Z, label=y)

        del Z

        return True


class MemoryProbe(xgb.callback.TrainingCallback):
    """Samples device usage and per-round wall time once a round."""

    def __init__(
        self, device: str, on_round: Callable[[Any], None] | None = None
    ) -> None:

        self.device = device
        self.peak_mb = gpu_used_mb(device)
        self.round_s: list[float] = []
        self._last = time.perf_counter()
        self._on_round = on_round

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:

        now = time.perf_counter()

        self.round_s.append(now - self._last)
        self._last = now
        self.peak_mb = max(self.peak_mb, gpu_used_mb(self.device))

        # each round pages the whole ellpack off disk, so rounds are minutes
        # apart -- write the curve out as it goes rather than only at the end
        if self._on_round is not None:
            self._on_round(evals_log)

        print(
            f"    round {epoch + 1:4d}  {self.round_s[-1]:6.1f}s  "
            f"gpu {gpu_used_mb(self.device):5.0f} MB  "
            f"rss {host_peak_rss_mb():5.0f} MB  avail {host_available_mb():5.0f} MB",
            flush=True,
        )

        return False


def run_xgboost(
    stream: RawStream,
    quant: Quant,
    holdouts: dict[str, tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    args: argparse.Namespace,
    device: str,
    checkpoint: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:

    params = {
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

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # xgboost warns that paging external memory without a pooled allocator is
    # slow. `use_cuda_async_pool` is cudaMallocAsync, built into the driver --
    # unlike `use_rmm` it needs nothing installed.
    xgb.set_config(use_cuda_async_pool=args.cuda_async_pool)

    baseline_mb = gpu_used_mb(device)
    build_wall, build_cpu = time.perf_counter(), time.process_time()

    it = StreamedQuant(stream, quant, device, str(cache / "lendb"))
    dtrain = xgb.ExtMemQuantileDMatrix(it, max_bin=args.max_bin)

    # the held-out slices are small enough to bin in one go; ref=dtrain reuses
    # the training cuts so they do not sketch their own
    evals = []
    for name, (X, Y) in holdouts.items():
        evals.append(
            (xgb.QuantileDMatrix(X, label=Y, ref=dtrain, max_bin=args.max_bin), name)
        )

    build = {
        "phase": "build",
        "wall_s": time.perf_counter() - build_wall,
        "cpu_s": time.process_time() - build_cpu,
    }

    cache_bytes = sum(f.stat().st_size for f in cache.glob("*"))

    print(
        f"  ellpack built in {build['wall_s']:.0f}s over {it.passes} passes; "
        f"{cache_bytes / 1e9:.1f} GB on disk; device {baseline_mb:.0f} -> "
        f"{gpu_used_mb(device):.0f} MB",
        flush=True,
    )

    torch.cuda.empty_cache() if device.startswith("cuda") else None

    evals_result: dict[str, Any] = {}

    def entry(log: Any, train: dict[str, Any]) -> dict[str, Any]:
        return {
            "x_name": "round",
            "params": params,
            "timings": [build, train],
            "memory": {
                "device_peak_mb": probe.peak_mb,
                "device_total_mb": gpu_total_mb(device),
                "host_peak_rss_mb": host_peak_rss_mb(),
                "streamed": True,
                "external_memory": True,
                "cache_gb": cache_bytes / 1e9,
                "build_passes": it.passes,
                "cuda_async_pool": args.cuda_async_pool,
                "batch_size": stream.batch_size,
                "resident": "paged from disk",
            },
            "seconds_per_round": float(np.mean(probe.round_s))
            if probe.round_s
            else 0.0,
            "results": {
                name: {"merror": curve(log[name]["merror"])}
                for _, name in evals
                if name in log
            },
        }

    train_wall, train_cpu = time.perf_counter(), time.process_time()

    probe = MemoryProbe(
        device,
        on_round=lambda log: checkpoint(
            entry(
                log,
                {
                    "phase": "train",
                    "wall_s": time.perf_counter() - train_wall,
                    "cpu_s": time.process_time() - train_cpu,
                },
            )
        ),
    )

    booster = xgb.train(
        {
            k: v
            for k, v in params.items()
            if k not in ("num_boost_round", "early_stopping_rounds")
        },
        dtrain,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        evals=evals,
        evals_result=evals_result,
        callbacks=[probe],
        verbose_eval=False,
    )

    train = {
        "phase": "train",
        "wall_s": time.perf_counter() - train_wall,
        "cpu_s": time.process_time() - train_cpu,
    }

    va = evals_result["va"]["merror"]

    print(
        f"  {len(va)} rounds in {train['wall_s']:.0f}s "
        f"({np.mean(probe.round_s):.1f}s/round)  "
        f"va_final={va[-1]:.4f} va_best={min(va):.4f} (iter {booster.best_iteration})",
        flush=True,
    )

    return {
        **entry(evals_result, train),
        "best_iteration": int(booster.best_iteration),
    }


# == main ======================================================================


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data")
    parser.add_argument("--dataset", default="LenDB")
    parser.add_argument("--out", default="results")
    parser.add_argument(
        "--model", default="hashboost", choices=("hashboost", "xgboost")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-ref",
        type=int,
        default=32768 * 2,
        help="training rows the reference run used",
    )
    parser.add_argument("--n-va", type=int, default=4096)
    parser.add_argument("--n-te", type=int, default=4096)
    parser.add_argument(
        "--n-tr", type=int, default=0, help="cap the training pool (0 = all of it)"
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-subsample", type=int, default=4096)
    parser.add_argument(
        "--drop-cache-every",
        type=int,
        default=50,
        help="batches between MADV_DONTNEED on the data file (0 = never)",
    )
    # hashboost
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-bits", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=20)
    # xgboost
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--cache-dir", default="/tmp/xgb-extmem")
    parser.add_argument(
        "--no-cuda-async-pool",
        dest="cuda_async_pool",
        action="store_false",
        help="disable cudaMallocAsync when paging external memory",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = args.device

    tr, va, te = split_indices(
        args.path, args.dataset, args.seed, args.n_ref, args.n_va, args.n_te
    )

    if args.n_tr:
        tr = tr[: args.n_tr]

    path_X = f"{args.path}/{args.dataset}/{args.dataset}_X.npy"
    path_Y = f"{args.path}/{args.dataset}/{args.dataset}_y.npy"

    stream = RawStream(
        path_X,
        path_Y,
        tr,
        args.batch_size,
        seed=args.seed,
        drop_cache_every=args.drop_cache_every,
    )

    raw_gb = tr.shape[0] * np.prod(stream._X.shape[1:]) * 4 / 1e9

    quant, num_features = fit_quant(stream, device)

    print(
        f"{args.dataset}: streaming {tr.shape[0]} training rows "
        f"({raw_gb:.1f} GB raw, {tr.shape[0] * num_features * 4 / 1e9:.0f} GB as "
        f"{num_features} QUANT features -- neither materialised)",
        flush=True,
    )

    # the held-out slices are small enough to keep as features on the device
    holdouts = {}
    for name, indices in (("tr", tr[: args.eval_subsample]), ("va", va)):
        raw, y = stream.gather(indices)
        holdouts[name] = (
            features(raw, quant, device),
            torch.as_tensor(y.astype(np.int64), device=device),
        )

    num_classes = int(np.unique(np.load(path_Y, mmap_mode="r")).shape[0])

    print(
        f"  holdouts: tr {tuple(holdouts['tr'][0].shape)} (subsample), "
        f"va {tuple(holdouts['va'][0].shape)}; {num_classes} classes",
        flush=True,
    )

    out = Path(args.out) / f"{args.dataset}-full-{commit_hash()}.json"

    info = {
        "commit": commit_hash(),
        "dataset": args.dataset,
        "device": device,
        "split": {
            "fold": 0,
            "seed": args.seed,
            "n_tr": int(tr.shape[0]),
            "n_va": int(va.shape[0]),
            "n_te": int(te.shape[0]),
            "n_eval_subsample": args.eval_subsample,
            "batch_size": args.batch_size,
            "num_classes": num_classes,
            "streamed": True,
            "reference_n_tr": args.n_ref,
        },
        "transform": {
            "name": "quant",
            "depth": quant.depth,
            "div": quant.div,
            "num_features": num_features,
        },
    }

    def checkpoint(entry: dict[str, Any]) -> None:
        write_results(out, {args.model: entry}, info)

    runner = run_hashboost if args.model == "hashboost" else run_xgboost
    checkpoint(runner(stream, quant, holdouts, num_classes, args, device, checkpoint))

    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
