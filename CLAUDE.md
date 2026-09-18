# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FIT2082 research project: **HashBoost**, gradient boosting over random hash
partitions ("Global Partitions for Boosting Time Series Classification", see
`spec/spec.tex`), GPU-resident in torch, benchmarked on MONSTER time-series
datasets (Pedestrian, InsectSound, LenDB, Tiselac, Traffic) after a QUANT feature
transform, against XGBoost/LightGBM/CatBoost. `README.md` is the lab notebook:
one section per commit or branch, holding measured results, the commands that
produced them and practical notes. Read the relevant section before changing a
component, because it records what has already been tried and what failed.

Hardware budget: one 8 GB RTX 4060 and 7.9 GB host RAM. Memory usage shapes a lot
of the design (chunking, streaming, page-cache eviction).

## Commands

Everything runs through `uv` (Python 3.12).

    uv run pytest                                   # 125 tests; CUDA-only ones skip without a GPU
    uv run pytest tests/test_boost.py::test_name    # single test
    uv run ruff check . && uv run ruff format .
    uv run ty check

    uv run python scripts/fetch_datasets.py ...     # MONSTER data from Hugging Face into data/
    uv run python -m fit2082.boost.experiment --list
    uv run python -m fit2082.boost.experiment --dataset Pedestrian --seeds 3 --compile \
        --variants baseline,capacity_2
    uv run python -m fit2082.boost.benchmark        # torch vs numba reference
    uv run python scripts/stream_full.py --dataset LenDB --model hashboost --epochs 5
    uv run python scripts/xgboost_baseline.py --dataset LenDB

`--compile` (torch.compile the hash encoding) gives identical results and is
faster, but costs a few seconds on first use. Scripts under `scripts/` accept
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
  frozen contribution is cached.
- `tables.py`: the performance core (encode/predict/accumulate/refresh_logits).
  The layout notes in its docstring are load-bearing: codes are round-major
  `(rounds, n)`, and gradient plus hessian are fused into one `stats` tensor.
- `partition.py`: `Partitioner` owns the predicate family, its parameter
  storage and its encoder (axis-aligned or oblique).
- `splits.py`: a `Splitter` chooses the members of that family
  (`HardPairSplitter`, with `sample=True` for Gumbel-sampled pairs).
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
  for torch). `fit2082/demo/utils.py` has `Dataset`, the memmapped `.npy` loader.
- `fit2082/results.py`: shared result-file schema
  (`{commit, dataset, device, split, transform, models: {...}}`) and
  GPU/host memory probes. The notebooks read these files.
- `scripts/stream_full.py`: streams a whole training pool from the memmap,
  never holding it all in memory, for HashBoost or XGBoost (external-memory
  ellpack). It includes page-cache management; the README's "Practical notes"
  explain why (sorted batch indices, readahead, periodic `MADV_DONTNEED`).
- `notebooks/`: plot from `results/*.json` and train nothing
  (`compare.ipynb` is the exception: it produced the off-the-shelf baselines).

Data layout: `data/<Name>/<Name>_X.npy`, `<Name>_y.npy`,
`test_indices_fold_0.txt`. `data/` and `results/` are gitignored.

## Experimental standards

- Run-to-run sd on Pedestrian is about 0.003, and seeds do not make reruns
  reproducible (leaf values are chaotic). Always rerun `baseline` **in the same
  sweep** and report mean +- sd over several seeds. Where a change does not
  alter training, prefer paired comparisons.
- Quote validation error against `X_va`. Anything chosen after training
  (early stopping, readout `lam`) must use the separate `X_tune` slice.
- Keep train and validation splits byte-identical to earlier runs (fixed seed
  123, fold 0), so that results stay comparable across commits.

## Conventions

- Code is split into sections with `# == name ===...` banner comments. Docstrings
  and comments explain *why*, often with measured numbers.
- Commit messages: `type: lowercase summary` (types seen: feat, perf, fix, docs,
  notebooks, scripts, tests, style, deps, chore). Commits written by Claude end
  the subject with ` (claude)` and have a prose body explaining the reasoning and
  measurements.
- When a change produces results, add or update the matching `README.md` section.
