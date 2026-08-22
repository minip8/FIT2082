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
