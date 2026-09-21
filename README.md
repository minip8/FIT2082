# FIT2082

## Benchmarks

### pulsar

PULSAR (Cabello & Kulik, ICDM 2025) as a second feature transform, ported to
torch in `fit2082/pulsar/pulsar.py` and selected with `--transform pulsar`.
It is supervised: the Fisher-score selection is fitted on every training batch
and nothing else. Train and validation rows are the same as in every QUANT
sweep, so the two transforms are compared on identical rows. Sweeps for any
transform other than QUANT are written to `results/{dataset}-{transform}-sweep-{commit}.json`.

    uv run python -m fit2082.boost.experiment --dataset InsectSound --num-train 32768 \
        --transform pulsar --seeds 3 --compile --variants baseline,sampled_pairs

3 seeds, compiled. The two transforms ran as back-to-back invocations (one
`load_split` each):

| dataset | variant | QUANT | PULSAR |
| --- | --- | ---: | ---: |
| InsectSound | `baseline` | 0.2699 +- 0.0094 | **0.2262 +- 0.0044** |
| | `sampled_pairs` | 0.2625 +- 0.0030 | **0.2258 +- 0.0028** |
| Pedestrian | `baseline` | **0.2326 +- 0.0059** | 0.2481 +- 0.0034 |
| | `sampled_pairs` | **0.2213 +- 0.0066** | 0.2393 +- 0.0037 |

* **InsectSound: -0.044**, the biggest single gain in this log. It beats
  ANOVA-F weighting of QUANT's features (-0.027 to -0.036 in the screen) and
  closes about half the gap to XGBoost's 0.189. Sampled pairs add nothing on
  top of it.
* **Pedestrian: +0.016 to +0.018**, several times the noise floor. Pedestrian
  is the dataset F weighting did not move either. One untested explanation
  is dilution: HashBoost draws features uniformly, and PULSAR offers 3,000
  columns against QUANT's 212.

Cost. The transform is a one-off setup cost, and features stay cached on the
device:

| dataset | features (QUANT) | transform (QUANT) | peak GPU | fit wall |
| --- | --- | ---: | ---: | ---: |
| InsectSound | 14,811 (5,470) | 48.2s (1.9s) | 2,650 MB | 3.5s |
| Pedestrian | 3,000 (212) | 3.3s (0.9s) | 1,154 MB | 9.4s |

Feature counts at upstream's defaults, with 40% of local features kept:
Tiselac 27,346 and LenDB 44,253. At 65,536 rows these would be 6.7 GB and
10.8 GB, so they will not fit in `experiment.py`'s device cache without a lower
`top_percent`, and neither has been run.

#### Streaming

`scripts/stream_full.py --transform pulsar` fits PULSAR in one labelled pass
over the reference run's 65,536 training rows (`--fit-rows`), then transforms
each streamed batch like QUANT. Only one batch is on the device at a time,
which is also how Tiselac and LenDB can run at all. The cost is recomputed
every epoch. Measured per 4,096-row batch, one timed call each on random
data of each dataset's shape:

| dataset | PULSAR | QUANT | peak GPU |
| --- | ---: | ---: | ---: |
| Pedestrian / Traffic | 98 ms | 34 ms | 235 MB |
| Tiselac | 698 ms | 44 ms | 1,024 MB |
| InsectSound | 2,735 ms | 142 ms | 597 MB |
| LenDB | 7,104 ms | 150 ms | 1,544 MB |

Traffic streams at about 28 s of transform per epoch over its 1,160,582 rows.
LenDB would take about 28 minutes per epoch, roughly 2.4 hours for the 5-epoch
stream that takes 6 minutes with QUANT. The port computes every local feature
and then keeps 40%, so computing only the kept ones (as upstream does at test
time) is the obvious speed-up. It has not been profiled.

    uv run python scripts/stream_full.py --dataset Traffic --transform pulsar \
        --epochs 10 --compile

#### Faithfulness of the port

Checked once in scratch against upstream's own code (numba, statsmodels 0.14);
the repo does not keep that oracle, and `tests/test_pulsar.py` checks against
transcriptions of upstream's loops instead.

* Feature counts are identical for every representation, for lengths 24, 60
  and 150. The batched Burg recursion matches `statsmodels.burg` to 5e-8. The
  histogram median and IQR match upstream exactly on 200k rows.
* Global features agree column-for-column to 1e-4. The exception is
  near-constant partitions, where upstream reports a stdev of ~0.006 that is
  rounding noise: it squares float32 values before its float64 subtraction.
  The port computes centred moments and returns 0 there.
* About 0.5% of local features differ, all at ties: a histogram bin boundary,
  a mean-crossing, or a stdev threshold, reached through last-bit differences
  in the float32 statistics.

Departures from upstream, all deliberate and listed in the module docstring:
multivariate input is handled channel by channel, as QUANT does; the Fisher
score and scaler are accumulated over batches; and the AR order is clamped to
`length - 2`.

### hashboost-screen

A profile of `fit_batch`, three changes it led to, and a screen of about 30
variants on up to five datasets. Pedestrian + QUANT through `experiment.py`
unless stated; every comparison is against a baseline rerun in the same sweep.
`notebooks/screen.ipynb` plots the sweeps and streamed runs made at `157e54d`.

    uv run python -m fit2082.boost.experiment --seeds 3 --compile \
        --variants baseline,capacity_2
    uv run python -m fit2082.boost.experiment --seeds 5 --compile \
        --variants baseline,sampled_pairs
    uv run python -m fit2082.boost.experiment --seeds 3 --compile \
        --variants baseline,frozen_2ep,frozen_10ep,capacity_2,frozen_10ep_capacity_2
    uv run python scripts/stream_full.py --dataset Traffic --epochs 10 --compile \
        --active-epochs 2 --label hashboost_frozen_2ep

#### Where the time goes

Share of one batch, measured stage by stage with synchronisation between stages:

| stage | Pedestrian | InsectSound | LenDB |
| --- | ---: | ---: | ---: |
| features / classes | 212 / 82 | 5,470 / 10 | 14,940 / 2 |
| ms per batch (at rounds) | 27.9 (800) | 13.5 (800) | 18.3 (1,200) |
| accumulate | 57% | 21% | 8% |
| encode | 19% | 43% | 47% |
| predict | 10% | 11% | 10% |
| refresh | 10% | 3% | 1% |
| transpose | 1% | 14% | 28% |
| objective + propose | 4% | 9% | 7% |

Many classes make the scatter dominate; many features make the encoding
dominate. For the streamed runs neither matters much: over the 1,195-round
LenDB streams below, a batch averaged 11 ms in `fit_batch` against 172 ms in
QUANT and 125 ms reading the memmap. QUANT makes 120 separate quantile calls
per representation over only 9-11 distinct interval lengths, so batching
intervals by length is the obvious next step there.

#### Exact kernels: ~1.5x, identical results (cf658bc)

| variant | before | eager | `--compile` | peak GPU |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 11.4s | 8.9s | ~7.5s | 428 -> 284 MB |
| `capacity_2` | 38.5s | 26.7s | 24.9s | 771 -> 491 MB |

* **Encode** built `(chunk, bits, n)` floats plus two int32 copies to weight
  and sum, where the answer is one byte per `(round, example)`. It now ORs one
  bit at a time into a uint8 code: 2-3.5x faster eagerly, and `torch.compile`
  can fuse each bit into a single kernel, 9-18x against the old one.
* **Accumulate** tiled the `(n, 2k)` update over every round of a chunk with
  `repeat` before scattering. `scatter_add_` over `expand`ed views takes the
  same sums without the copy: 1.6-2.4x.
* **Refresh** writes leaf values in place.

`--compile` stays opt-in: it costs ~5.5 s on first use and ~1.5 s with a warm
inductor cache, charged to the first seed.

#### Sampled hard pairs: a small gain that keeps its sign (1a2297c)

`HardPairSplitter(sample=True)` draws the pair examples in proportion to cross
entropy (Gumbel-top-k) instead of taking the hardest first. The strict ranking
rebuilds every hash from the same dozen or so hardest rows each time a batch
comes round -- once memorised, its persistent outliers.

| dataset | seeds | baseline | `sampled_pairs` |
| --- | ---: | ---: | ---: |
| Pedestrian | 5 | 0.2273 +- 0.0019 | **0.2220 +- 0.0034** |
| InsectSound | 5 | 0.2696 +- 0.0056 | 0.2643 +- 0.0034 |

Two screening sweeps before this one read -0.005 and -0.006 on Pedestrian, and
the sweep at `157e54d` read -0.001 there and -0.008 on InsectSound. Four
Pedestrian sweeps agree on the sign and not on the size, so read it as a small
gain rather than a settled number. Screening on Tiselac, Traffic and LenDB (3
seeds) read -0.001, 0.000 and 0.000, and it hurt nowhere it was tried.

#### Frozen rounds: capacity_2's accuracy in 2.4x less time (d39acad)

`active_rounds` freezes every round older than a window, and `fit_batch(...,
rows=...)` caches each row's frozen contribution, so a batch costs O(window +
rounds frozen since the row was last seen) instead of O(rounds). In
`experiment.py` the window is `active_epochs`, so a variant means the same on
every dataset. 3 seeds, compiled:

| variant | val error | wall |
| --- | ---: | ---: |
| `baseline` | 0.2252 +- 0.0023 | 8.4s |
| `frozen_10ep` | 0.2276 +- 0.0037 | 3.6s |
| `frozen_2ep` | 0.2332 +- 0.0042 | 1.8s |
| `capacity_2` | 0.2133 +- 0.0025 | 25.6s |
| **`frozen_10ep_capacity_2`** | **0.2135 +- 0.0032** | **10.8s** |

The sweep at `157e54d` repeats it: `frozen_10ep_capacity_2` at 0.2129 in
10.3 s against `capacity_2` at 0.2137 in 24.7 s.

The window is a real hyperparameter. On Pedestrian a 2-epoch window cost
+0.008, +0.013, +0.023 and +0.027 across four sweeps; a 10-epoch window cost
+0.001 to +0.004. On InsectSound even 2 epochs cost nothing measurable
(0.2726 +- 0.0085 against 0.2703 +- 0.0059, 5 seeds). The saving grows with
the number of rounds: InsectSound's 400-round runs are barely faster.

Practical: pass row ids from the host. The first version read the per-row
state back off the GPU, and those two synchronisations per batch made caching
slower (4.8 s) than freezing without it (4.1 s).

#### Frozen rounds in a streamed run (157e54d)

`stream_full.py --active-epochs` freezes the same way over a whole training
pool, keyed by each row's index into the .npy, and now records synchronised
stage times at every evaluation. Single runs each; validation error is the mean
of the last seven evaluations, because on 4,096 rows consecutive evaluations
differ by as much as 0.01:

| stream | window | val error | `fit_batch` | QUANT | read | wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Traffic, 10 epochs, 1,160,582 rows | none | 0.3922 | 29s | 121s | 2s | 154s |
| | 2 epochs | 0.3877 | 20s | 114s | 2s | 139s |
| | 1 epoch | 0.4052 | 21s | 125s | 2s | 151s |
| LenDB, 5 epochs, 975,291 rows | none | 0.0745 | 13s | 206s | 149s | 370s |
| | 2 epochs | 0.0751 | 14s | 206s | 147s | 368s |
| | 1 epoch | 0.0698 | 13s | 208s | 148s | 371s |

* **A 2-epoch window cost nothing measurable on either pool.** A 1-epoch window
  trailed by 0.013 on Traffic, the one gap bigger than the noise and a single
  run, so a lead rather than a result.
* **Freezing flattens the model's cost per batch.** Unfrozen, Traffic's grows
  linearly, 2 ms to 14.5 ms over 2,840 rounds; frozen, it settles near 7.5 ms.
  LenDB's ~7 ms of fixed per-batch work (14,940 columns to transpose and
  encode) swamps the growth over 1,195 rounds.
* **But in a streamed run the model is the smaller part:** 14-19% of Traffic's
  wall time and under 4% of LenDB's. The 29 s to 20 s that freezing saved on
  Traffic is smaller than QUANT's own spread across the three runs (114-125 s).
  Extrapolating Traffic's slope, the unfrozen model would only match the 42 ms
  per batch of reading plus QUANT at around 11,000 rounds.
* **The whole Traffic pool pays.** On the same validation rows HashBoost scored
  0.4404 from 65,536 training rows, and XGBoost and LightGBM 0.3997 and 0.3950;
  ten streamed epochs over all 1,160,582 rows are level with the tree models.

#### Also measured, with prototype code not in the repo

These came from subclasses written for the screen and are not reproducible
from this branch. Same-sweep baselines, 3-5 seeds.

**Weighting feature draws by ANOVA F is dataset-specific.** Drawing features
from `0.5 * uniform + 0.5 * F / sum(F)`, F computed on the training rows:

| | InsectSound | Pedestrian | LenDB | Tiselac | Traffic |
| --- | ---: | ---: | ---: | ---: | ---: |
| features | 5,470 | 212 | 14,940 | 2,040 | 212 |
| change in val error | **-0.027** | -0.002 | +0.001 | +0.004 | **+0.011** |

On InsectSound -- the largest gap to XGBoost -- error falls steadily with the
F weight (0.265, 0.245, 0.238, 0.233, 0.229 at 0, 0.25, 0.5, 0.75, 1), F from a
single batch keeps most of it (0.2415), and it stacks with `capacity_2` (0.2264)
and bagging (0.2269 against `bagged_4`'s 0.2491). But it hurts Traffic, and
restricting draws to the top quarter of features was catastrophic on Pedestrian
(0.3420, train error 0.20). Dilution is not the general explanation: LenDB has
the most features and did not move.

**`hashes_per_round` counts each batch H times into existing rounds**, as
`test_hashes_per_round_matches_repeated_fit_batch` already pins down. A variant
adding two hashes per batch but updating old rounds once lost most of
`capacity_2`'s Pedestrian gain -- 0.2301 +- 0.0125 against 0.2122 +- 0.0033,
baseline 0.2252 -- while on InsectSound the two were equal (0.2492, 0.2516).
So `cac8e0b`'s "more capacity, same data passes" is only part of the story.

**Pedestrian and InsectSound sit in different regimes.** Halving the batch to
2,048 (twice the rounds) cost +0.021 on Pedestrian and gained -0.015 on
InsectSound. Pedestrian's 82 imbalanced classes make it limited by noisy leaf
estimates -- fewer examples per round hurts, as early freezing and single
updates also did -- while InsectSound is limited by capacity.

**Leaf values are chaotic on Pedestrian**, which is part of why seeds do not
control reruns (`3d25491`). Two models replaying identical hashes end up to 180
logits apart, with 1.4% of validation predictions flipped. Under deterministic
kernels a 1e-6 nudge to one round's statistics grows to 60 logits in 4 epochs.
Extreme leaves are the amplifier: clipping leaves to +-3 holds the nudge at 1e-6,
at no accuracy cost (5 seeds). They come from Newton steps the hessian floor
dominates -- a bucket holding one rare-class example gets roughly
`lr / (count * 1e-3)` -- and reach +-100. Same-seed reruns still diverge once
split selection meets a floating-point near-tie, at round 393 instead of 53, so
clipping alone does not shrink the noise floor. InsectSound showed no such
amplification.

**Dead ends.** Label smoothing (0.05 / 0.1: Pedestrian +0.007 / +0.033), a
hessian floor of 1e-2 or 1e-4 (Pedestrian +0.006 / +0.098: 1e-3 is
load-bearing), thresholds drawn uniformly between the pair, reshuffling batches
every epoch, choosing among 4 candidate features the one that best separates
the pair (-0.007 on InsectSound, dominated by F weighting; -0.001 on
Pedestrian), and leaf clipping as an accuracy lever.

### streaming-baselines

Streaming the whole LenDB training pool -- 975,291 rows -- through both models,
against the 65,536 rows every earlier run used. Validation and test rows are
held byte-identical to those runs, so only the training set size changes.

    uv run python scripts/xgboost_baseline.py --dataset LenDB
    uv run python scripts/stream_full.py --model hashboost --epochs 5
    uv run python scripts/stream_full.py --model xgboost --num-boost-round 3

Neither script materialises the features. The pool is 6.3 GB as raw series
against 7.9 GB of host RAM, and 58 GB once QUANT expands it to 14,940 columns.

| run | rows | rounds | tr | va best | wall | peak GPU | data resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hashboost | 65,536 | 1200 | 0.0049 | **0.0532** | 230s | 2,572 MB | one batch |
| hashboost | 975,291 | 1195 | 0.0608 | 0.0691 | 560s | 2,572 MB | one batch |
| xgboost | 65,536 | 179 | 0.0236 | **0.0459** | 131s | 6,223 MB | whole ellpack in RAM |
| xgboost | 975,291 | 3 | 0.0566 | 0.0627 | 702s | 6,021 MB | 14.6 GB paged from disk |

#### Peak memory is flat in HashBoost and linear in XGBoost

HashBoost took **2,572 MB for 65,536 rows and 2,572 MB for 975,291** -- the same
number, because `fit_batch` folds a batch in and frees it, so the pool size never
enters. That is the property that makes the full dataset reachable at all.

XGBoost needs the whole binned matrix available for every boosting round, so
streaming only moves where it lives. The ellpack is almost exactly one byte per
element, which is worth knowing because a global symbol space would imply ~22
bits and a 40 GB matrix:

| rows | ellpack | s/round |
| ---: | ---: | ---: |
| 65,536 | in RAM | 0.68 |
| 122,880 | 1.8 GB | 8.9 |
| 245,760 | 3.7 GB | 16.2 |
| 975,291 | 14.6 GB | 116.0 |

Linear in rows, and 171x the in-memory cost per round at the full pool: 353s to
build the ellpack, then every round pages 14.6 GB back off disk. Reaching the
65,536-row run's 179 rounds would take about 6 hours.

#### More data made HashBoost worse, because it is capacity-bound not data-bound

At matched rounds the small run wins -- 0.0532 against 0.0691 -- on the same
gradient steps, the same samples processed and identical capacity. The train
errors say why: the 65,536-row model reaches **tr=0.0049**, having essentially
memorised its training set, while the full-pool model sits at tr=0.0608 against
va=0.0691. Train ~= validation is underfitting. 1,200 hashes is enough to
memorise 65k rows and nowhere near enough for 975k, and the full-pool curve was
still descending when it stopped.

So this is not "more data hurts". It is that more data needs more rounds, and
per-batch cost grows linearly with the number of rounds -- 50 passes over the
full pool is ~9,000 rounds, roughly 60x the compute spent here. That quadratic
is where HashBoost's scaling actually binds.

#### Practical notes

* Sort each batch's indices after the global shuffle. Same rows, but the memmap
  reads go from scattered to near-sequential: 593 ms/batch to 64 ms.
* Readahead reads 38x the bytes these scattered rows need. Leave it on anyway --
  `MADV_RANDOM` cuts that to 1.6x but is 20x *slower* (1571 ms/batch against 78),
  because sorting is what makes the kernel's bulk fetches pay off. Advise the
  *mapping*: `posix_fadvise` on the fd governs `read()`, not page faults through
  a memmap, and measured byte-for-byte identical.
* Pay for the amplification by dropping the page cache periodically instead --
  but not too often. Every 20 batches is fine; every 4 is pathological, because
  `MADV_DONTNEED` resets the kernel's readahead state and it re-ramps: 3.7 GB
  read per batch instead of 202 MB, and 16x slower.
* Drop the input's cache at *startup* too. The 8 GB .npy leaves ~5 GB of itself
  cached, so the second run of anything starts with MemFree at 195 MB.
* `POSIX_FADV_DONTNEED` cannot evict a dirty page. XGBoost's ~15 GB of ellpack
  writes need an fsync first or they sit in the page cache until the run dies.
* Watch MemFree, not just MemAvailable. They disagree by 6 GB during a streaming
  run, and it is the pessimistic one that gets runs killed.

### 5a36857

Baseline with off-the-shelf models (XGBoost, LightGBM, CatBoost, DecisionTreeClassifier).

### 4eb8d6b

* QUANT feature transform
* Train off-the-shelf models with the entire dataset instead of a slice

### 30c956f

GPU hash boosting (`fit2082.boost`), replacing the numba implementation in
`fit2082/demo/boost.py` (kept as the reference oracle).

* Pedestrian, 82 classes, 65,536 training rows, batch 4,096, 50 epochs
  (800 rounds): **101.5s -> 10.6s**, at equal validation error and ~369 MB
  of GPU memory. The CPU is no longer saturated.
* Verified against the numba implementation: given the same hashes, logits
  agree to `4e-7` with 100% argmax agreement.
* Run it with `uv run python -m fit2082.boost.benchmark`.

### cac8e0b

Accuracy work on HashBoost. Variants are run with
`uv run python -m fit2082.boost.experiment --variants <names> --seeds 3`
(`--list` shows them); results land in `results/{dataset}-sweep-{commit}.json`.

Pedestrian + QUANT, 800 rounds unless stated, 3 seeds, validation error:

| variant | val error | note |
|---|---|---|
| baseline | 0.2240 ± 0.0004 | |
| **`hashes_per_round=2`** (1600 rounds) | **0.2150 ± 0.0032** | more capacity, same data passes |
| `BaggedHashBoost(2)` (2 × 800 rounds) | 0.2204 ± 0.0010 | control for the above |
| `neighbour_shrinkage` 0.1 / 0.3 / 0.6 | 0.2233 / 0.2250 / 0.2318 | no gain |

**Run multiple seeds.** The run-to-run standard deviation is about 0.003, so a
single run can easily show a 0.006 "improvement" that is pure noise.

Two findings worth keeping:

* **The randomness is load-bearing.** Gain-based split selection (best-of-K by
  boosting gain) raises occupied buckets from 79/256 to 119/256 and makes the
  *ensemble worse* (0.2241–0.2339). So do statistic decay (0.2227/0.2383/0.4128
  at γ = 0.999/0.99/0.95), row subsampling (0.2585 at 0.5), richer feature
  spaces (QUANT `div=2`/`div=1`: 0.2275/0.2434; +848 random feature
  differences: 0.2239), and leaf L2. `num_bits=8` and `lr=0.1` are already at
  their optimum. Making individual hashes smarter makes the ensemble worse.
* **Capacity beats averaging at 1600 hashes, but not per unit compute.**
  `hashes_per_round=2` and `BaggedHashBoost(2)` both fit 1600 hashes and the
  boosted one is better (0.2150 vs 0.2204). That advantage does not survive
  scaling up -- see `b21d3e8`, where bagging wins on cost by a wide margin.

Cost is quadratic in total rounds — 1600 rounds is 37.5s against 800 rounds'
10.5s — and GPU memory scales as `max_num_hashes × 2**num_bits × num_classes`.

For reference, XGBoost on the same split reaches 0.2041 (see `4eb8d6b`).

### b21d3e8

Combining capacity with bagging closes the gap to XGBoost. Pedestrian + QUANT,
3 seeds, validation error:

| variant | val error | wall | total hashes |
|---|---|---|---|
| baseline | 0.2240 ± 0.0004 | 10.5s | 800 |
| `bagged_4` | 0.2128 ± 0.0024 | 43.5s | 3200 |
| `capacity_4` | 0.2103 ± 0.0012 | 818.0s | 3200 |
| **`bagged_4_capacity_2`** | **0.2059 ± 0.0017** | 148.1s | 6400 |
| `smooth_0.3_bagged_4_capacity_2` | 0.2049 ± 0.0004 | 304.7s | 6400 |
| *XGBoost (`4eb8d6b`)* | *0.2041* | *332.7s* | |

`bagged_4_capacity_2` matches XGBoost at less than half the training time. Note
the XGBoost figure is the best point on its boosting curve from a single run,
while these are the final error averaged over three seeds, so if anything the
comparison flatters XGBoost.

**Bagging divides the quadratic cost.** Training cost is quadratic in rounds
*per model*, so splitting H hashes across E models costs `H**2 / E` rather than
`H**2`. At 3200 hashes, four 800-round models take 43.5s where one 3200-round
model takes 818s -- far more than the 4x the algebra predicts, the rest being
memory pressure (the deep model peaked near the 8 GB card limit). Prefer more
estimators over more rounds when buying capacity.

**`neighbour_shrinkage` has no consistent effect** and should be treated as a
dead end without further evidence. Across four paired comparisons it helps
twice and hurts twice, averaging to roughly zero:

| pair | without | with | effect |
|---|---|---|---|
| baseline | 0.2240 | 0.2250 | -0.0010 |
| `capacity_2` | 0.2150 | 0.2107 | +0.0043 |
| `bagged_4` | 0.2128 | 0.2188 | -0.0060 |
| `bagged_4_capacity_2` | 0.2059 | 0.2049 | +0.0010 |

The mechanism is real -- it takes non-zero leaves from 31% to 78% of buckets --
but it does not turn into accuracy. This is a good illustration of the ±0.003
noise floor: the `capacity_2` row alone looks like a 1.5σ win and is not one.

### 3d25491

Three accuracy extensions, and a methodology correction that reframes every
number above. Results in `results/Pedestrian-sweep-3d25491.json`, where each
variant carries the `sweep` it was measured in -- **only variants sharing a
sweep are directly comparable**, for the reason in the next section.

Headline: `readout_round_class_bagged_4_capacity_2` reaches **0.2006 +- 0.0025**
against XGBoost's 0.2041, the first result here that is below XGBoost rather
than level with it.

#### Seeds do not control the run-to-run noise

Baseline, identical configuration, three separate sweeps:

| sweep | baseline val error |
|---|---|
| `b21d3e8` | 0.2240 +- 0.0004 |
| 1 | 0.2270 +- 0.0090 |
| 2 | 0.2214 +- 0.0011 |

A 0.0056 spread, and the three-sample sd itself ranges 0.0004 to 0.0090 for the
same settings. `bagged_4` drifts the same way (0.2128 / 0.2132 / 0.2149), and so
does `estimator_agreement` (0.876 / 0.865).

`torch.manual_seed(seed)` does not make a run reproducible. Float32 scatter-adds
on CUDA accumulate in nondeterministic order; that perturbs the logits; the
perturbed logits reorder the cross-entropy ranking in `HardPairSplitter`; and
every hash chosen after that point differs. A "seed" here is just a rerun, and
three seeds measure repeatability, not seed-sensitivity.

So the +-0.003 noise floor quoted above understates it. **Unpaired differences
below about 0.006 are not interpretable, and a variant needs a control rerun in
its own sweep.** Several conclusions below reversed when their control was rerun
beside them -- `bagged_4_mixed_family` read as neutral against a stale control
and as a win against a fresh one; `readout_round` read as +0.003 in one sweep and
-0.003 in the next.

Where a change does not alter training, the comparison can instead be *paired*
against the very same fitted model, which removes all of this. `run_once` records
`boosted_final` per run for exactly that.

#### Refitting the readout: +0.0078 paired, and rounds are not the problem

`predict` sums one table row per round, each row being `lr * G / (H + eps)` -- a
Newton step computed for that round in isolation, assuming every other round is
held fixed. Correct *while* boosting; not the best leaf values for the finished
ensemble. Nothing had measured the gap.

`tables.logits` is `(rounds, 2**num_bits, k)` = 16.8M floats, which is exactly
the parameter count of a multiclass linear model on the one-hot codes -- because
it *is* that tensor. The refit is therefore not a head bolted on top: it is the
leaf tables, fit jointly against cross entropy instead of round by round,
warm-started at the boosted solution and penalised back toward it. The forward
pass already existed and was already differentiable (`F.embedding_bag`).
`fit2082/boost/readout.py`, run via the `readout` key in a variant.

Paired against the same model and seed (positive = the refit won), 800 rounds:

| rung | parameters | paired delta | extra wall |
|---|---|---|---|
| `round` -- one gain per round | 800 | **+0.0000 +- 0.0028** | +4s |
| `round_class` -- one per (round, class) | 65,600 | +0.0061 +- 0.0034 | +5s |
| `table` -- the full table | 16,793,600 | **+0.0078 +- 0.0002** | +7s |

The premise holds, but not in the expected shape. **Rescaling whole rounds is
worth exactly nothing** (+0.0010, +0.0022, -0.0032 across seeds). The gain comes
from changing leaf values *within* a round, most of it from the per-class rung.
The boosted leaves are not mis-weighted between rounds; they are wrong per class.

The `table` rung moves every seed by nearly the same amount (+0.0080, +0.0075,
+0.0078), so its paired sd is 0.0002 where the unpaired sd over those same three
runs is 0.0084 -- a 40x difference, and the sharpest illustration of the section
above. About 40% of the tune-slice gain reaches validation; the tune figure is
optimistically biased by early stopping, the validation figure is not.

**It stacks on the best ensemble.** `round_class` on `bagged_4_capacity_2`,
paired **+0.0049 +- 0.0025** (per seed +0.0063, +0.0020, +0.0063):

| model | val error | wall |
|---|---|---|
| `bagged_4_capacity_2` | 0.2052 +- 0.0015 | 150s |
| **`readout_round_class_bagged_4_capacity_2`** | **0.2006 +- 0.0025** | 375s |
| *XGBoost (`4eb8d6b`)* | *0.2041* | *333s* |

One seed reached 0.1985. Note this comparison is the honest way round: XGBoost's
figure is the best point on one run's curve, these are final error over 3 seeds.

**Which rung wins depends on model size.** `table` beat `round_class` on a plain
800-round model and collapses to +0.0008 +- 0.0004 on the 6400-round ensemble --
*not* from overfitting: its tune error moved 0.2070 -> 0.2070, i.e. it barely
trained. 300 Adam steps at lr=0.01 on 134M parameters warm-started at the prior
is not enough optimisation when each parameter only sees gradient from the ~16
examples per batch landing in its bucket. It also peaked at 8424 MB on an 8 GB
card and took 1531s. Treat `table` as untuned at that scale, not beaten.

Methodology: the readout early-stops on a tune slice taken from *after* the
validation slice, so training and validation sets stay byte-identical to every
earlier sweep and nothing is ever selected on `X_va`. `best_error` starts at the
warm start, so a refit can never return something worse than the model it was
given.

#### Mass-adaptive leaf smoothing revives a dead end

Fixed `neighbour_shrinkage` is recorded above as noise -- it helps in two paired
comparisons and hurts in two, despite the mechanism demonstrably working. That
signature says one global alpha is helping sparse buckets and damaging dense
ones. `shrinkage_tau` makes the borrowing empirical-Bayes instead:

    alpha_s = tau / (tau + mass_s),   mass_s = sum_k hessian[s, k]

An empty bucket takes its neighbours' evidence in full, a well-populated one is
left alone. `tau` reads as "the bucket mass at which a bucket trusts itself and
its neighbours equally".

800 rounds, 3 seeds, same-sweep baseline 0.2214 +- 0.0011:

| tau | val error |
|---|---|
| 100 | 0.2197 |
| 300 | 0.2179 |
| 1000 | 0.2178 |
| **3000** | **0.2149** |
| 10000 | 0.2199 |
| 30000 | 0.2261 |
| 100000 | 0.2467 |
| 1000000 | 0.3803 |

A clean inverted U with an interior optimum -- a shape baseline drift cannot
manufacture, which is what makes this more convincing than the single-point
comparisons elsewhere in this log. The optimum sits where the mechanism predicts:
typical bucket mass is ~1e4 here (65,536 rows x 50 epochs / 256 buckets), so
tau = 3000 mixes about a quarter of the neighbours into a well-populated bucket
while replacing an empty one wholesale.

**It gains least where bagging already is.** At tau = 3000:

| base | without | with | gain | |
|---|---|---|---|---|
| `capacity_2` | 0.2109 | 0.2062 | **+0.0047** | same sweep |
| `bagged_4` | 0.2149 | 0.2115 | +0.0034 | across sweeps |
| `bagged_4_capacity_2` | 0.2052 | 0.2023 | +0.0029 | across sweeps |

Only the first row is a same-sweep comparison and so the only one to lean on;
the other two are quoted across sweeps and could each be off by ~0.005.

At tau = 100 the middle row reads +0.0000 and the bottom +0.0009: a combination
measured at a badly chosen hyperparameter looks like a null, which is worth
remembering before writing one off.

`estimator_agreement` suggests why the gain shrinks under bagging: smoothing
pushes agreement between bagged estimators *up*, 0.865 -> 0.910 -> 0.924. Part of
what decorrelates the estimators is exactly the idiosyncratic empty-bucket noise
smoothing removes, so the two overlap -- though only partly, since smoothing
still pays on top of bagging.

Practical: `adaptive_smooth_3000_capacity_2` reaches 0.2062 in 78s against
`bagged_4_capacity_2`'s 0.2052 in 150s -- near-equal accuracy from a single model
at half the compute. Smoothing roughly doubles wall time (the pooled copy in
`refresh_logits`), so it is worth it against capacity and marginal against
bagging.

#### Heterogeneous bagging: real decorrelation, no reliable payoff

`BaggedHashBoost` now takes per-estimator `overrides` (cycled kwarg patches), so
members can differ in bit width, learning rate and partition family rather than
only in their random draws. The variance of an average falls with the
*correlation* between members, and that lever had never been touched. To vary the
family, pass the partitioner *class*: `HashBoost` builds one per estimator, where
a shared instance would give all four members the same split tables.

Same sweep, 800 rounds:

| variant | val error | agreement |
|---|---|---|
| `bagged_4` (homogeneous) | 0.2149 +- 0.0020 | 0.865 |
| `bagged_4_mixed_bits` | 0.2177 +- 0.0018 | 0.855 |
| `bagged_4_mixed_family` | 0.2122 +- 0.0015 | 0.854 |
| `bagged_4_mixed_all` | 0.2171 +- 0.0004 | 0.850 |

Every mixture decorrelates -- agreement drops 0.010 to 0.015, reproducibly across
two sweeps -- and only one converts. `mixed_family` (axis-aligned alternating
with oblique) is 0.0027 *better* than homogeneous bagging despite oblique being
individually the worst member available: the textbook diversity effect, a weaker
but differently-wrong member improving the average. Mixing bit widths is
consistently worse, because `num_bits` 6 and 7 lower member quality without
buying more decorrelation than the family mix does.

Read this cautiously: every effect here is <= 0.003 against sds of ~0.002 and
observed drift of ~0.005. What reproduces across sweeps is the *ordering*, not
the magnitudes. `mixed_family` is worth another look at more seeds; `mixed_bits`
and `mixed_all` are not.

`bagged_4_mixed_all_capacity_2` OOMs on an 8 GB card -- a 9-bit member at 1600
rounds is 537 MB of statistics by itself.

#### Oblique partitions, measured at last

`ObliquePartitioner` has been implemented and unit-tested since `c1b64fe` and
never measured. Alone: **0.2277 +- 0.0009** (and 0.2306 +- 0.0023 in another
sweep) against baselines of 0.2214 to 0.2270 -- consistently a little worse,
matching the "+848 random feature differences: 0.2239" screening above. Its value
is as an ensemble member, not a replacement. Thread closed.

#### Practical notes

* `experiment.py` writes its results file after *every* variant. A sweep is tens
  of minutes and the expensive variants run last; an OOM there used to discard
  everything before it.
* Rerun the control in the same sweep. Seeds alone are not enough.
* Prefer a paired comparison wherever a change does not alter training. It turned
  an unreadable +-0.0084 into +-0.0002 on the same three runs.
