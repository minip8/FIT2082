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
* **Capacity beats averaging at matched hash count.** `hashes_per_round=2` and
  `BaggedHashBoost(2)` both fit 1600 hashes; the boosted one is better
  (0.2150 vs 0.2204), so the gain comes from fitting more residuals in
  sequence, not from averaging more models.

Cost is quadratic in total rounds — 1600 rounds is 37.5s against 800 rounds'
10.5s — and GPU memory scales as `max_num_hashes × 2**num_bits × num_classes`.

For reference, XGBoost on the same split reaches 0.2041 (see `4eb8d6b`).
