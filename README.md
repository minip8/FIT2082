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
