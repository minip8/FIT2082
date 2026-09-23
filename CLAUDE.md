# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FIT2082 research project: **HashBoost**, gradient boosting over random hash
partitions ("Global Partitions for Boosting Time Series Classification", see
`spec/spec.tex`), GPU-resident in torch, benchmarked on MONSTER time-series
datasets (Pedestrian, InsectSound, LenDB, Tiselac, Traffic) and the UCR 112 after
a QUANT feature transform, against XGBoost/LightGBM/CatBoost and QUANT's own
ExtraTrees. The paper draft is `paper/paper.tex`. `README.md` is the lab notebook:
one section per commit or branch, holding measured results, the commands that
produced them and practical notes. Read the relevant section before changing a
component, because it records what has already been tried and what failed.

Hardware budget: one 8 GB RTX 4060 and 7.9 GB host RAM. Memory usage shapes a lot
of the design (chunking, streaming, page-cache eviction).

## Commands

Everything runs through `uv` (Python 3.12).

    uv run pytest                                   # 214 tests; CUDA-only ones skip without a GPU
    uv run pytest tests/test_boost.py::test_name    # single test
    uv run ruff check . && uv run ruff format .
    uv run ty check

    uv run python scripts/fetch_datasets.py ...     # MONSTER data from Hugging Face into data/
    uv run python -m fit2082.boost.experiment --list
    uv run python -m fit2082.boost.experiment --dataset Pedestrian --seeds 1 --compile \
        --variants baseline,capacity_2
    uv run python -m fit2082.boost.benchmark        # torch vs numba reference
    uv run python scripts/stream_full.py --dataset LenDB --model hashboost --epochs 5
    uv run python scripts/xgboost_baseline.py --dataset LenDB
    uv run python scripts/ucr_benchmark.py --compile   # UCR 112: HashBoost, ExtraTrees, XGBoost
    uv run python scripts/ucr_benchmark.py --compile --models hashboost --bits 2 \
        --rounds 3200 --label hb_b2_r3200            # a HashBoost variant beside the rest
    uv run python scripts/ucr_benchmark.py --compile --shard 0/3   # 1 of 3 concurrent shards

`--compile` (torch.compile the hash encoding and the leaf refresh) is faster
but costs a few seconds on first use. The encoding is exact; the fused refresh
can round a leaf differently in its last 2 ulps, well below run-to-run noise. Scripts under `scripts/` accept
`--stdin` to read their arguments as JSON or as flag text (`fit2082/cli.py`).

## Architecture

`fit2082/boost/` is the model. It is split along pluggable seams (typing
`Protocol`s) so that an experiment can swap one piece at a time:

- `model.py`: `HashBoost`. Each `fit_batch` adds `hashes_per_round` rounds. A
  round is `num_bits` binary predicates that index a `2**num_bits x k` leaf
  table. **Every batch updates the buckets of all existing rounds**, so the cost
  per batch grows linearly with the number of rounds, and that growth is the
  scaling bottleneck. `active_rounds` freezes rounds older than a window. If you
  pass `rows=` (stable row ids, kept on the host to avoid GPU syncs), each row's
  hash codes are cached (up to `code_cache_bytes`), so a batch only encodes new
  rounds, and with `active_rounds` its frozen contribution is cached too. On
  small batches a round is launch overhead, not arithmetic: see the README's
  `cheaper-rounds` section before adding per-round kernels.
- `tables.py`: the performance core (encode/predict/accumulate/refresh_logits).
  The layout notes in its docstring are load-bearing: codes are round-major
  `(rounds, n)`, and gradient plus hessian are fused into one `stats` tensor.
  Prediction picks gather-and-sum or `embedding_bag` by batch shape
  (`GATHER_LIMIT`), and with `compile=True` the refresh is one fused kernel.
- `partition.py`: `Partitioner` owns the predicate family, its parameter
  storage and its encoder (axis-aligned or oblique).
- `splits.py`: a `Splitter` chooses the members of that family
  (`HardPairSplitter`, with `sample=True` for Gumbel-sampled pairs). On CUDA
  the greedy pairing runs as a one-thread Triton kernel (`pair_on_device`), so
  a round needs no host sync. It must stay pair-for-pair equal to `_pair`,
  and a test checks this.
- `objective.py`: maps logits to (gradient, hessian).
- `ensemble.py`: `BaggedHashBoost` with per-member `overrides`.
- `readout.py`: refits the leaf tables jointly after boosting.
- `experiment.py`: the sweep runner. `VARIANTS` maps names to HashBoost kwargs
  plus the runner-only keys `estimators`, `overrides`, `readout` and
  `active_epochs`. Its comments record what each variant measured, so add new
  variants there with a note. It writes
  `results/{dataset}-sweep-{commit}.json` after every variant.

Also:

- `fit2082/demo/boost.py`: the original numba implementation. It is kept as an
  **independent reference implementation**. `tests/test_boost.py` replays the
  same hashes through both implementations and requires agreement within a
  tolerance. CUDA float32 scatter-adds are nondeterministic, so tests compare
  approximately. The comparison uses a default-constructed HashBoost and skips
  split selection, so **add new behaviour as an opt-in kwarg or variant**, and
  keep the defaults equal to the reference algorithm. If a change to the default
  maths breaks these tests, the default model no longer matches the reference.
  Change the tests only as a deliberate decision.
- `fit2082/quant/quant.py`: QUANT transform (third-party research code, adapted
  for torch). `IntervalModel` sorts every interval of one length in a single
  call. Upstream's per-interval `f_quantile` stays as the reference that
  `tests/test_quant.py` checks it against. `fit2082/demo/utils.py` has
  `Dataset`, the memmapped `.npy` loader.
- `fit2082/pulsar/pulsar.py`: PULSAR transform, a torch port of GPL-3.0 upstream
  code. Unlike QUANT it is supervised: `Pulsar().fit(batches)` needs labels.
  `--transform pulsar` selects it in `experiment.py` and `scripts/stream_full.py`
  (the latter fits it in one labelled pass over `--fit-rows` first).
- `fit2082/ucr.py`: the UCR archive. `UCR112` is the bake-off's 112 datasets,
  `load_ucr` reads one on its default train/test split, and `fit_hashboost`,
  `fit_extratrees` and `fit_xgboost` train at fixed settings: there is no
  validation split, so nothing is chosen after training. HashBoost's budget is
  800 rounds, not 50 epochs, because most UCR training sets fit in one batch.
  `scripts/ucr_benchmark.py` runs them, and `notebooks/ucr.ipynb` ranks them.
- `fit2082/results.py`: shared result-file schema
  (`{commit, dataset, device, split, transform, models: {...}}`) and
  GPU/host memory probes. The notebooks read these files.
- `scripts/stream_full.py`: streams a whole training pool from the memmap,
  never holding it all in memory, for HashBoost or XGBoost (external-memory
  ellpack). It includes page-cache management; the README's "Practical notes"
  explain why (sorted batch indices, readahead, periodic `MADV_DONTNEED`).
  HashBoost runs read the next batch in a background thread while the GPU
  trains (`--prefetch`; 0 restores the serial loop). `read_s` is the time spent
  reading, and `read_wait_s` the part the training loop waited for.
- `notebooks/`: plot from `results/*.json` and train nothing
  (`compare.ipynb` is the exception: it produced the off-the-shelf baselines).

Data layout: `data/<Name>/<Name>_X.npy`, `<Name>_y.npy`,
`test_indices_fold_0.txt` for MONSTER, and the unzipped UCR 2018 archive at
`data/UCRArchive_2018/<Name>/<Name>_{TRAIN,TEST}.tsv` (label in the first
column). `data/` and `results/` are gitignored; UCR results go to
`results/UCR/<Name>-<commit>.json`.

## Experimental standards

- Run-to-run sd on Pedestrian is about 0.003, and seeds do not make reruns
  reproducible (leaf values are chaotic). Always rerun `baseline` **in the same
  sweep**. Where a change does not alter training, prefer paired comparisons.
- When Claude runs sweeps, use `--seeds 1` (the user reruns with more seeds
  when a result matters). Say that a number comes from one seed, and do not
  read a gap under ~0.006 on Pedestrian as a result.
- Quote validation error against `X_va`. Anything chosen after training
  (early stopping, readout `lam`) must use the separate `X_tune` slice. UCR is
  the exception: it has only train and test, so UCR runs choose nothing and
  quote test error on the default split.
- Keep train and validation splits byte-identical to earlier runs (fixed seed
  42, fold 0), so that results stay comparable across commits. Seed 42 gives
  the rows the tree baselines and streamed runs use. `experiment.py` sweeps made
  before the `ucr` branch drew seed 123, a different split, so compare them only
  with each other (`--split-seed 123` reproduces them).

## Conventions

- Code is split into sections with `# == name ===...` banner comments. Docstrings
  and comments explain *why*, often with measured numbers.
- Commit messages: `type: lowercase summary` (types seen: feat, perf, fix, docs,
  notebooks, scripts, tests, style, deps, chore). Commits written by Claude end
  the subject with ` (claude)` and have a prose body explaining the reasoning and
  measurements.
- When a change produces results, add or update the matching `README.md` section.
