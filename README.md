# FIT2082

## Benchmarks

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
