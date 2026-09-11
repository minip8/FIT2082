"""Train HashBoost and XGBoost on a dataset's *whole* training pool by streaming.

The fixed-size runs use 65,536 rows because that is what fits once QUANT has
expanded them. Nothing here is ever fully resident -- batches are read from the
.npy memmap, transformed on the GPU, and dropped -- so the pool can be as large
as the dataset is.

LenDB is the case that forced it, and the figures quoted throughout are
measured there: 975,291 training rows, 6.3 GB as raw series against 7.9 GB of
host RAM, and 58 GB once QUANT expands them to 14,940 features. None of it is
specific to LenDB. `--dataset` takes any directory under `--path` laid out as
`<dataset>/<dataset>_X.npy`, `<dataset>/<dataset>_y.npy` and
`test_indices_fold_0.txt`; the machinery that only earns its keep on a dataset
too big for RAM -- evicting the page cache, paging the ellpack off disk --
sizes itself to the data and turns itself off when it is not needed.

The two models meet that constraint very differently, which is the comparison
worth having:

* HashBoost is an online learner. `fit_batch` folds one batch into the model
  and the batch is then free, so peak memory is one batch no matter how much
  data streams past. What it pays instead is time: every batch updates the
  buckets of all existing rounds, so total work is quadratic in rounds.
* XGBoost needs the whole binned matrix available for every boosting round.
  Streaming only changes where that matrix lives -- `ExtMemQuantileDMatrix`
  writes the ellpack to disk once and pages it back on each round.

The validation and test rows are held byte-identical to the fixed-size run on
the same dataset, so the curves here are directly comparable to
`results/<dataset>-*.json`; the training pool is everything else. That hinges on
`--n-ref` matching the row count that run trained on, because it is the offset
the held-out slices are cut at -- `REFERENCE_N_REF` records the ones already
run, and is what `--n-ref` defaults to.

    uv run python scripts/stream_full.py --dataset LenDB --model hashboost --epochs 5
    uv run python scripts/stream_full.py --dataset Traffic --model xgboost \
        --num-boost-round 50

Arguments can also be piped in with `--stdin`, which is easier to generate than
a command line when the same script is being run over several datasets --
`fit2082.cli` has the details.

    jq -c '.runs[]' sweep.json | while read -r run; do
        echo "$run" | uv run python scripts/stream_full.py --stdin
    done
"""

import argparse
import mmap
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xgboost as xgb
import xgboost.callback

from fit2082.boost import BaggedHashBoost, HashBoost
from fit2082.cli import parse_args
from fit2082.quant.quant import Quant
from fit2082.results import (
    commit_hash,
    curve,
    gpu_total_mb,
    gpu_used_mb,
    host_available_mb,
    host_free_mb,
    host_peak_rss_mb,
    host_rss_mb,
    write_results,
)

# == split =====================================================================

# `n_ref` decides where the validation and test slices are cut out of the
# shuffle, so a streamed run has to reuse the value of the run it is being
# compared against even though it trains on the whole pool regardless. These are
# what the fixed-size runs in `results/` used: everything took 65,536 except
# InsectSound, whose 40,000 rows outside fold 0 could not hold it.
DEFAULT_N_REF = 32768 * 2
REFERENCE_N_REF = {"InsectSound": 32768}


def split_indices(
    path: str, dataset: str, seed: int, n_ref: int, n_va: int, n_te: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The notebook's split, with everything it left on the floor added to train.

    `notebooks/compare.ipynb` shuffles the non-fold-0 rows and takes the first
    `n_ref` for training, then `n_va` and `n_te`. Reproducing that shuffle and
    keeping the same validation and test slices means the only thing that
    changes here is the size of the training set -- the numbers stay comparable
    to every earlier run on the same dataset.
    """

    np.random.seed(seed)

    Y = np.load(f"{path}/{dataset}/{dataset}_y.npy", mmap_mode="r")
    fold = np.loadtxt(f"{path}/{dataset}/test_indices_fold_0.txt")

    ix = np.setdiff1d(np.arange(Y.shape[0]), fold)

    # a pool too small to reach the validation slice at all would leave the run
    # with nothing to evaluate on, several hours in
    if n_ref + n_va > ix.shape[0]:
        raise SystemExit(
            f"{dataset}: {ix.shape[0]} rows outside fold 0 cannot hold "
            f"--n-ref {n_ref} plus --n-va {n_va}. Lower --n-ref to whatever the "
            f"run this one is meant to be comparable with trained on."
        )

    np.random.shuffle(ix)

    va = ix[n_ref : n_ref + n_va]

    # slicing truncates rather than raises: a pool with no room for a full test
    # slice gets the short one the reference run got, and te is held out either
    # way. Only `n_te` rows the reference run *did* see would be a problem.
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
        random_access: bool = False,
    ) -> None:

        self._X = np.load(path_X, mmap_mode="r")
        self._Y = np.load(path_Y, mmap_mode="r")

        # numpy closes the file it mmapped, so keep an fd of our own to advise on
        self._fd = os.open(path_X, os.O_RDONLY)

        # Readahead reads far more than these scattered rows need -- 38x on a
        # sorted 4,096-row batch -- but leave it on anyway: sorting the batch
        # makes the pattern semi-sequential, and the kernel's bulk fetches are
        # 20x *faster* than the page-at-a-time faulting MADV_RANDOM gives
        # (78 ms/batch against 1571 ms). The read amplification is paid for by
        # dropping the cache periodically, not by defeating readahead.
        #
        # MADV_RANDOM is offered for the memory-starved case. Note the advice
        # has to go on the *mapping*: these reads are page faults through the
        # memmap, and posix_fadvise on the fd measured byte-for-byte identical.
        if random_access:
            self._X._mmap.madvise(mmap.MADV_RANDOM)

        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_cache_every = drop_cache_every

        self._rng = np.random.default_rng(seed)

    def drop_cache(self) -> None:
        """Hand the pages we have already consumed back to the kernel.

        A LenDB epoch pulls 6.3 GB of an 8.1 GB file through the page cache,
        which on a 7.9 GB machine leaves `free` at a few hundred MB. The pages
        are clean and reclaimable, so nothing is actually short of memory -- but
        enough things watch `free` rather than `available` that the run gets
        killed anyway. MADV_DONTNEED drops them outright; re-reading costs the 64
        ms/batch that the sorted access pattern already assumes.
        """

        # MADV_DONTNEED on a shared file mapping only zaps our page table
        # entries -- the page cache keeps the pages, and MemFree stays low.
        # Dropping the PTEs first is what then lets fadvise evict them.
        self._X._mmap.madvise(mmap.MADV_DONTNEED)
        os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)

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


def auto_drop_cache_every(path_X: str, every: int = 50) -> int:
    """Evict the page cache only for a file large enough to crowd host RAM.

    Dropping the cache is what keeps a LenDB-sized run alive, but it is pure
    loss on a dataset that fits: Pedestrian's series are 18 MB, so the kernel
    would happily hold the whole file for the length of the run and re-reading
    it every 50 batches buys nothing. Compare the file against what the machine
    can actually spare rather than against a fixed threshold -- the same dataset
    is worth streaming carefully on a small box and not on a large one.
    """

    size_mb = Path(path_X).stat().st_size / 1e6

    return every if size_mb > 0.5 * host_available_mb() else 0


def drop_page_cache(directory: Path) -> None:
    """Flush and evict a directory's files from the page cache.

    XGBoost writes ~15 GB of ellpack pages into its cache directory, and those
    writes land in the page cache as *dirty* pages. POSIX_FADV_DONTNEED cannot
    evict a dirty page -- it has to be written back first -- so the cache grows
    until the machine has nothing free and the run is killed, which is exactly
    what happened at 4.8 GB of ellpack. Sync, then advise.
    """

    for path in sorted(directory.glob("*")):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue

        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        finally:
            os.close(fd)


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

    # `rounds` below counts batches, as it always has, so curves stay comparable
    # across runs. Each batch adds `hashes_per_round` hashes to every estimator,
    # which is what `max_num_hashes` has to be sized for.
    total_rounds = args.epochs * len(stream)

    kwargs: dict[str, Any] = {
        "num_classes": num_classes,
        "num_bits": args.num_bits,
        "lr": args.lr,
        "max_num_hashes": total_rounds * args.hashes_per_round + 1,
        "hashes_per_round": args.hashes_per_round,
        "shrinkage_tau": args.shrinkage_tau,
    }

    params = {**kwargs, "max_epochs": args.epochs, "estimators": args.estimators}

    torch.manual_seed(args.seed)

    # Bagging shares the stream: each batch is read and transformed once and
    # handed to every estimator, so the I/O is not paid E times over.
    model: HashBoost | BaggedHashBoost = (
        HashBoost(**kwargs, device=device)
        if args.estimators == 1
        else BaggedHashBoost(num_estimators=args.estimators, **kwargs, device=device)
    )

    print(
        f"  {len(stream)} batches/epoch x {args.epochs} epochs = {total_rounds} rounds"
        f" ({total_rounds * args.hashes_per_round * args.estimators} hashes over "
        f"{args.estimators} estimator(s))",
        flush=True,
    )

    records: dict[str, list[tuple[int, float]]] = {"tr": [], "va": []}
    peak_mb = gpu_used_mb(device)
    read_s = 0.0
    transform_s = 0.0
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
            "read_s": read_s,
            "transform_s": transform_s,
            "n_tr": int(stream.indices.shape[0]),
            "results": {
                name: {"merror": curve([e for _, e in rec], [r for r, _ in rec])}
                for name, rec in records.items()
            },
        }

    wall, cpu = time.perf_counter(), time.process_time()

    for _ in range(args.epochs):
        # `for raw, y in stream` would fold the memmap read into the loop
        # machinery, where it cannot be timed -- and the read is the part that
        # dropping the page cache makes expensive, so it is the part worth
        # measuring. Pull each batch explicitly instead.
        batches = iter(stream)

        while True:
            mark = time.perf_counter()
            batch = next(batches, None)
            read_s += time.perf_counter() - mark

            if batch is None:
                break

            raw, y = batch

            mark = time.perf_counter()
            Z = features(raw, quant, device)
            Y = torch.as_tensor(y.astype(np.int64), device=device)
            transform_s += time.perf_counter() - mark

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
                    f"  rss {host_rss_mb():5.0f} MB  avail {host_available_mb():5.0f} MB",
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
        f"({read_s:.0f}s reading, {transform_s:.0f}s transforming)  "
        f"va_final={va[-1]:.4f} va_best={min(va):.4f}",
        flush=True,
    )

    return entry(elapsed)


# == xgboost ===================================================================


class StreamedQuant(xgb.DataIter):
    """Feeds the streamed QUANT features to XGBoost's external-memory builder.

    `on_host=False` sends the cached ellpack pages to `cache_prefix` on disk:
    LenDB's full pool bins to roughly 40 GB, which fits neither the 8 GB card
    nor the 7.9 GB of host RAM. A dataset whose ellpack would fit is written to
    disk all the same -- external memory is the thing being measured.
    """

    def __init__(
        self,
        stream: RawStream,
        quant: Quant,
        device: str,
        cache_prefix: str,
        drop_cache_every: int = 20,
    ) -> None:

        self._stream = stream
        self._quant = quant
        self._device = device
        self._cache_dir = Path(cache_prefix).parent
        self._drop_cache_every = drop_cache_every
        self._batches_seen = 0
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

        self._batches_seen += 1

        if self._drop_cache_every and self._batches_seen % self._drop_cache_every == 0:
            # the input's own pages are dropped by RawStream; these are the
            # ellpack pages xgboost has just written
            drop_page_cache(self._cache_dir)

        # the build is several minutes of silence otherwise, and it is where
        # both kills happened -- say what memory is doing while it runs
        if self._batches_seen % 40 == 0:
            written = sum(f.stat().st_size for f in self._cache_dir.glob("*"))

            print(
                f"    build pass {self.passes} batch {self._batches_seen:4d}  "
                f"cache {written / 1e9:5.2f} GB  gpu {gpu_used_mb(self._device):5.0f} MB"
                f"  rss {host_rss_mb():5.0f} MB  free {host_free_mb():5.0f} MB"
                f"  avail {host_available_mb():5.0f} MB",
                flush=True,
            )

        return True


class MemoryProbe(xgb.callback.TrainingCallback):
    """Samples device usage and per-round wall time once a round."""

    def __init__(
        self,
        device: str,
        on_round: Callable[[Any], None] | None = None,
        cache_dir: Path | None = None,
    ) -> None:

        self.device = device
        self._cache_dir = cache_dir
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
        if self._cache_dir is not None:
            drop_page_cache(self._cache_dir)

        if self._on_round is not None:
            self._on_round(evals_log)

        print(
            f"    round {epoch + 1:4d}  {self.round_s[-1]:6.1f}s  "
            f"gpu {gpu_used_mb(self.device):5.0f} MB  "
            f"rss {host_rss_mb():5.0f} MB  avail {host_available_mb():5.0f} MB",
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

    # one directory per dataset, so the size reported below is this run's
    # ellpack and not whatever an interrupted run on another dataset left
    # behind. xgboost overwrites its own pages, but a shorter run leaves the
    # tail of a longer one lying there, so clear the prefix first.
    stale = sorted(cache.glob(f"{args.dataset.lower()}*"))

    if stale:
        print(f"  clearing {len(stale)} stale cache file(s) from {cache}", flush=True)
        for path in stale:
            path.unlink()

    # xgboost warns that paging external memory without a pooled allocator is
    # slow. `use_cuda_async_pool` is cudaMallocAsync, built into the driver --
    # unlike `use_rmm` it needs nothing installed.
    xgb.set_config(use_cuda_async_pool=args.cuda_async_pool)

    baseline_mb = gpu_used_mb(device)
    build_wall, build_cpu = time.perf_counter(), time.process_time()

    it = StreamedQuant(
        stream,
        quant,
        device,
        str(cache / args.dataset.lower()),
        drop_cache_every=args.drop_cache_every,
    )
    dtrain = xgb.ExtMemQuantileDMatrix(
        it, max_bin=args.max_bin, cache_host_ratio=args.cache_host_ratio
    )

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

    cache_bytes = sum(f.stat().st_size for f in cache.glob(f"{args.dataset.lower()}*"))

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
            "n_tr": int(stream.indices.shape[0]),
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
        cache_dir=cache,
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
    parser.add_argument(
        "--label",
        default=None,
        help="key to store this run under (default: the model name). Lets a "
        "control on a smaller pool sit beside the full run in one file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-ref",
        type=int,
        default=None,
        help="training rows the reference run used -- the offset the held-out "
        "slices are cut at (default: "
        + "".join(f"{k} {v}, " for k, v in REFERENCE_N_REF.items())
        + f"else {DEFAULT_N_REF})",
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
        default=None,
        help="batches between MADV_DONTNEED on the data file (0 = never; "
        "default: every 50 if the series file would crowd host RAM, else never)",
    )
    # hashboost
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-bits", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument(
        "--hashes-per-round",
        type=int,
        default=1,
        help="hashes added per batch; cost is quadratic in the hash count",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=1,
        help="bag this many independent models (capacity at ~linear cost)",
    )
    parser.add_argument(
        "--shrinkage-tau",
        type=float,
        default=0.0,
        help="mass-adaptive leaf smoothing; scale with rows x epochs / 2**num_bits",
    )
    # xgboost
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="where xgboost writes its ellpack pages "
        "(default: /tmp/xgb-extmem/<dataset>)",
    )
    parser.add_argument(
        "--random-access",
        action="store_true",
        help="MADV_RANDOM on the input: 24x fewer bytes read, 20x slower",
    )
    parser.add_argument(
        "--cache-host-ratio",
        type=float,
        default=0.0,
        help="fraction of the external-memory cache xgboost may hold in host RAM",
    )
    parser.add_argument(
        "--no-cuda-async-pool",
        dest="cuda_async_pool",
        action="store_false",
        help="disable cudaMallocAsync when paging external memory",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parse_args(parser)

    device = args.device

    if args.n_ref is None:
        args.n_ref = REFERENCE_N_REF.get(args.dataset, DEFAULT_N_REF)

    if args.cache_dir is None:
        args.cache_dir = f"/tmp/xgb-extmem/{args.dataset}"

    tr, va, te = split_indices(
        args.path, args.dataset, args.seed, args.n_ref, args.n_va, args.n_te
    )

    if args.n_tr:
        tr = tr[: args.n_tr]

    path_X = f"{args.path}/{args.dataset}/{args.dataset}_X.npy"
    path_Y = f"{args.path}/{args.dataset}/{args.dataset}_y.npy"

    if args.drop_cache_every is None:
        args.drop_cache_every = auto_drop_cache_every(path_X)

    stream = RawStream(
        path_X,
        path_Y,
        tr,
        args.batch_size,
        seed=args.seed,
        drop_cache_every=args.drop_cache_every,
        random_access=args.random_access,
    )

    # a previous run leaves LenDB's 8 GB .npy sitting in the page cache -- 5 GB
    # of it here -- so every run after the first starts with MemFree near zero
    # and is the one that gets killed. Start from a clean slate. A file small
    # enough that we are not evicting during the run is not worth evicting now.
    if args.drop_cache_every:
        stream.drop_cache()

    raw_gb = tr.shape[0] * np.prod(stream._X.shape[1:]) * 4 / 1e9

    quant, num_features = fit_quant(stream, device)

    print(
        f"{args.dataset}: streaming {tr.shape[0]} training rows "
        f"({raw_gb:.1f} GB raw, {tr.shape[0] * num_features * 4 / 1e9:.1f} GB as "
        f"{num_features} QUANT features -- neither materialised)",
        flush=True,
    )
    print(
        f"  reference run trained on {args.n_ref}; "
        + (
            f"evicting the page cache every {args.drop_cache_every} batches"
            if args.drop_cache_every
            else "leaving the series in the page cache (it fits)"
        ),
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

    label = args.label or args.model

    def checkpoint(entry: dict[str, Any]) -> None:
        write_results(out, {label: entry}, info)

    runner = run_hashboost if args.model == "hashboost" else run_xgboost
    checkpoint(runner(stream, quant, holdouts, num_classes, args, device, checkpoint))

    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
