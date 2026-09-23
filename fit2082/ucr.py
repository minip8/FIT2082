"""The UCR archive: its 112 standard datasets, and the models run on them.

The 2018 archive (`data/UCRArchive_2018/<Name>/<Name>_{TRAIN,TEST}.tsv`, the
label in the first column) holds 128 univariate datasets. Fifteen have missing
values or variable lengths -- the ones the archive ships adjusted copies of, in
`Missing_value_and_variable_length_datasets_adjusted/` -- and Fungi has one
training series per class. The remaining 112 are the set the TSC bake-off
(Middlehurst et al. 2024) reports on. Published default-split accuracies for
QUANT, HC2 and others exist for exactly these names, so our numbers can be
compared with theirs directly.

Everything here uses the archive's own train/test split and chooses nothing
after training, so the test error is the only number reported. Unlike MONSTER
there is no validation slice to choose on, which is why every model runs at
fixed settings: HashBoost at the MONSTER default, the baselines at their
library defaults.

Most UCR training sets are smaller than one 4,096-row batch -- 27 of the 112
have 50 rows or fewer -- so HashBoost's budget is set in rounds, not epochs:
50 epochs would be 50 rounds for any set that fits in one batch.
"""

import time
from typing import Any, NamedTuple, Self

import numpy as np
import numpy.typing as npt
import torch

from fit2082.boost import HashBoost
from fit2082.quant.quant import Quant
from fit2082.results import curve

# == datasets ==================================================================

# The 128 dataset directories of UCRArchive_2018, less the 15 in its
# Missing_value_and_variable_length_datasets_adjusted/ folder, less Fungi. The
# 112 names match the published QUANT results file exactly.
UCR112: tuple[str, ...] = (
    "ACSF1", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "BME", "Car",
    "CBF", "Chinatown", "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers",
    "CricketX", "CricketY", "CricketZ", "Crop", "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW",
    "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices",
    "EOGHorizontalSignal", "EOGVerticalSignal", "EthanolLevel", "FaceAll", "FaceFour",
    "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB", "FreezerRegularTrain",
    "FreezerSmallTrain", "GunPoint", "GunPointAgeSpan", "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung", "Ham", "HandOutlines", "Haptics", "Herring",
    "HouseTwenty", "InlineSkate", "InsectEPGRegularTrain", "InsectEPGSmallTrain",
    "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2",
    "Lightning7", "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MixedShapesRegularTrain",
    "MixedShapesSmallTrain", "MoteStrain", "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect",
    "Phoneme", "PigAirwayPressure", "PigArtPressure", "PigCVP", "Plane", "PowerCons",
    "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW", "RefrigerationDevices", "Rock", "ScreenType",
    "SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2", "ShapeletSim",
    "ShapesAll", "SmallKitchenAppliances", "SmoothSubspace", "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols",
    "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG",
    "TwoPatterns", "UMD", "UWaveGestureLibraryAll", "UWaveGestureLibraryX",
    "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine", "WordSynonyms",
    "Worms", "WormsTwoClass", "Yoga",
)  # fmt: skip

DEFAULT_PATH = "data/UCRArchive_2018"

SEED = 42


class UCRDataset(NamedTuple):
    """One dataset on its default split, labels encoded to 0..k-1."""

    X_tr: npt.NDArray[np.float32]  # (n_tr, 1, length)
    y_tr: npt.NDArray[np.int64]
    X_te: npt.NDArray[np.float32]  # (n_te, 1, length)
    y_te: npt.NDArray[np.int64]
    classes: npt.NDArray[Any]  # the original labels, in encoded order


def _read(path: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    table = np.loadtxt(path, delimiter="\t", dtype=np.float64, ndmin=2)

    return table[:, 1:], table[:, 0]


def load_ucr(name: str, path: str = DEFAULT_PATH) -> UCRDataset:
    """Read one dataset's TRAIN and TEST files.

    Labels are encoded over train and test together, so a label means the same
    class in both. Every test class occurs in training for all of `UCR112`.
    NaN raises: in this archive it marks a missing value or the padding of a
    variable-length series, which is what `UCR112` leaves out.
    """

    X_tr, y_tr = _read(f"{path}/{name}/{name}_TRAIN.tsv")
    X_te, y_te = _read(f"{path}/{name}/{name}_TEST.tsv")

    if np.isnan(X_tr).any() or np.isnan(X_te).any():
        raise ValueError(
            f"{name} has missing values or variable lengths; UCR112 excludes "
            "those datasets"
        )

    classes, encoded = np.unique(np.concatenate([y_tr, y_te]), return_inverse=True)

    return UCRDataset(
        X_tr=X_tr.astype(np.float32)[:, None, :],
        y_tr=encoded[: y_tr.shape[0]].astype(np.int64),
        X_te=X_te.astype(np.float32)[:, None, :],
        y_te=encoded[y_tr.shape[0] :].astype(np.int64),
        classes=classes,
    )


# == features ==================================================================


def quant_features(
    X_tr: npt.NDArray[np.float32],
    X_te: npt.NDArray[np.float32],
    device: str,
    chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """QUANT features for both halves of a split, left on `device`.

    QUANT has no data-dependent state -- its intervals follow from the series
    length alone -- so fitting it on one row fixes the transform, as
    `scripts/xgboost_baseline.py` also relies on. The rows go through in chunks
    so a long test set (StarLightCurves: 8,236 x 1,024) never builds every
    intermediate at once.
    """

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    began = time.perf_counter()

    quant = Quant()
    quant.fit_transform(torch.as_tensor(X_tr[:1], device=device))

    def transform(X: npt.NDArray[np.float32]) -> torch.Tensor:
        return torch.cat(
            [
                quant.transform(torch.as_tensor(X[a : a + chunk], device=device))
                for a in range(0, X.shape[0], chunk)
            ]
        )

    Z_tr, Z_te = transform(X_tr), transform(X_te)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    info = {
        "name": "quant",
        "depth": quant.depth,
        "div": quant.div,
        "num_features": int(Z_tr.shape[1]),
        "transform_s": time.perf_counter() - began,
    }

    return Z_tr, Z_te, info


# == models ====================================================================

# Every fit_* takes device-resident features and integer labels, and returns a
# model entry in the shared result schema (`fit2082.results`): params, timings,
# and misclassification curves under results[split]["merror"]. The reported
# numbers are under "final" -- the error of the finished model, the same in
# every entry however its curve was sampled.


class _Timer:
    """Wall and CPU time of a block, synchronising the GPU at both ends."""

    def __init__(self, device: str, phase: str) -> None:

        self.device = device
        self.phase = phase

    def _sync(self) -> None:

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def __enter__(self) -> Self:

        self._sync()
        self.wall, self.cpu = time.perf_counter(), time.process_time()

        return self

    def __exit__(self, *exc: object) -> None:

        self._sync()
        self.record = {
            "phase": self.phase,
            "wall_s": time.perf_counter() - self.wall,
            "cpu_s": time.process_time() - self.cpu,
        }


def _error(predicted: Any, y: npt.NDArray[np.int64]) -> float:

    return float((np.asarray(predicted).reshape(-1) != y).mean())


def hashboost_batches(
    Z: torch.Tensor, y: torch.Tensor, batch_size: int, seed: int = SEED
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Fixed batches of at most `batch_size` rows, from one seeded permutation.

    Every batch must hold two classes: `HardPairSplitter` pairs examples of
    different classes, and on a batch of one class it would never finish.
    """

    order = torch.as_tensor(np.random.default_rng(seed).permutation(Z.shape[0]))
    order = order.to(Z.device)

    batches = [
        (Z[order[a : a + batch_size]], y[order[a : a + batch_size]])
        for a in range(0, Z.shape[0], batch_size)
    ]

    for _, Y in batches:
        if torch.unique(Y).numel() < 2:
            raise ValueError(
                "a batch holds a single class, which HardPairSplitter cannot "
                "pair; use a larger batch_size"
            )

    return batches


def fit_hashboost(
    Z_tr: torch.Tensor,
    y_tr: npt.NDArray[np.int64],
    Z_te: torch.Tensor,
    y_te: npt.NDArray[np.int64],
    num_classes: int,
    rounds: int = 800,
    batch_size: int = 4096,
    device: str = "cuda",
    compile: bool = False,
    curve_every: int = 50,
    seed: int = SEED,
    num_bits: int = 8,
) -> dict[str, Any]:
    """HashBoost at the default lr, trained for exactly `rounds` rounds.

    The fixed batches are cycled until the budget is spent, so a set that fits
    in one batch is re-accumulated every round, and one of 8,926 rows (three
    batches) stops part-way through its last pass. The test curve is recorded
    for plotting only; the reported error is the one at `rounds`.

    `num_bits` is the one model setting exposed: at the default 8, a round has
    256 buckets, far more than a UCR training set of a few dozen rows can
    fill.
    """

    y_tr_d = torch.as_tensor(y_tr, device=device)
    y_te_d = torch.as_tensor(y_te, device=device)

    batches = hashboost_batches(Z_tr, y_tr_d, batch_size, seed)

    torch.manual_seed(seed)

    model = HashBoost(
        num_classes=num_classes,
        num_bits=num_bits,
        max_num_hashes=rounds,
        device=device,
        compile=compile,
    )

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with _Timer(device, "fit") as fit:
        step = 0

        while model.num_rounds < rounds:
            X, Y = batches[step % len(batches)]
            model.fit_batch(X, Y)
            step += 1

    with _Timer(device, "predict") as predict:
        predicted = model.predict(Z_te).argmax(-1).cpu().numpy()

    peak_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if device.startswith("cuda") else 0.0
    )

    staged = model.staged_error(Z_te, y_te_d).cpu().numpy()
    x = list(range(0, rounds + 1, curve_every))
    if x[-1] != rounds:
        x.append(rounds)

    test_error = _error(predicted, y_te)
    train_error = _error(model.predict(Z_tr).argmax(-1).cpu().numpy(), y_tr)

    return {
        "x_name": "round",
        "params": {
            "num_classes": num_classes,
            "num_bits": model.num_bits,
            "lr": model.lr,
            "rounds": model.num_rounds,
            "batch_size": min(batch_size, Z_tr.shape[0]),
            "num_batches": len(batches),
            "passes": step / len(batches),
            "compile": compile,
            "seed": seed,
        },
        "timings": [fit.record, predict.record],
        "peak_mb": peak_mb,
        "final": {"te": test_error, "tr": train_error},
        "results": {
            "tr": {"merror": curve([train_error], [rounds])},
            "te": {"merror": curve([float(staged[i]) for i in x], x)},
        },
    }


# QUANT's own classifier: the settings of `QuantClassifier` in
# fit2082/quant/quant.py, trained in one piece rather than batch by batch
EXTRATREES: dict[str, Any] = {
    "n_estimators": 200,
    "criterion": "entropy",
    "max_features": 0.1,
    "n_jobs": -1,
    "random_state": SEED,
}


def fit_extratrees(
    Z_tr: torch.Tensor,
    y_tr: npt.NDArray[np.int64],
    Z_te: torch.Tensor,
    y_te: npt.NDArray[np.int64],
) -> dict[str, Any]:
    """ExtraTrees on the host: sklearn has no GPU path."""

    from sklearn.ensemble import ExtraTreesClassifier

    A_tr, A_te = Z_tr.cpu().numpy(), Z_te.cpu().numpy()

    model = ExtraTreesClassifier(**EXTRATREES)

    with _Timer("cpu", "fit") as fit:
        model.fit(A_tr, y_tr)

    with _Timer("cpu", "predict") as predict:
        predicted = model.predict(A_te)

    trees = EXTRATREES["n_estimators"]
    test_error = _error(predicted, y_te)
    train_error = _error(model.predict(A_tr), y_tr)

    return {
        "x_name": "trees",
        "params": {**EXTRATREES, "device": "cpu"},
        "timings": [fit.record, predict.record],
        "final": {"te": test_error, "tr": train_error},
        "results": {
            "tr": {"merror": curve([train_error], [trees])},
            "te": {"merror": curve([test_error], [trees])},
        },
    }


def _to_xgb(Z: torch.Tensor) -> Any:
    """Features as XGBoost takes them: CuPy on the GPU, with no host copy."""

    if Z.is_cuda:
        import cupy

        return cupy.from_dlpack(Z.contiguous())

    return Z.cpu().numpy()


def fit_xgboost(
    Z_tr: torch.Tensor,
    y_tr: npt.NDArray[np.int64],
    Z_te: torch.Tensor,
    y_te: npt.NDArray[np.int64],
    device: str = "cuda",
    curve_every: int = 10,
) -> dict[str, Any]:
    """XGBoost at its library defaults, on the device the features are on.

    The scikit-learn wrapper, not `xgb.train`: its defaults are the documented
    ones (100 rounds, eta 0.3, depth 6), where `xgb.train` stops at 10. The fit
    is timed with no eval sets attached, so its time is training alone, like
    HashBoost's; the test curve is computed afterwards from truncated
    ensembles, for plotting only.
    """

    import xgboost as xgb

    if device.startswith("cuda"):
        # torch's caching allocator holds blocks XGBoost's own allocator cannot
        # reuse; see scripts/xgboost_baseline.py
        torch.cuda.empty_cache()

    A_tr, A_te = _to_xgb(Z_tr), _to_xgb(Z_te)

    model = xgb.XGBClassifier(device=device, random_state=SEED)

    with _Timer(device, "fit") as fit:
        model.fit(A_tr, y_tr)

    with _Timer(device, "predict") as predict:
        predicted = model.predict(A_te)

    def error(A: Any, y: npt.NDArray[np.int64], rounds: int | None = None) -> float:
        kwargs = {} if rounds is None else {"iteration_range": (0, rounds)}
        labels = model.predict(A, **kwargs)
        return _error(labels.get() if hasattr(labels, "get") else labels, y)

    rounds = model.get_booster().num_boosted_rounds()

    x = list(range(curve_every, rounds + 1, curve_every))
    if not x or x[-1] != rounds:
        x.append(rounds)

    predicted = predicted.get() if hasattr(predicted, "get") else predicted
    test_error = _error(predicted, y_te)
    train_error = error(A_tr, y_tr)

    return {
        "x_name": "round",
        "params": {
            "library_defaults": True,
            "num_boosted_rounds": rounds,
            "device": device,
            "random_state": SEED,
        },
        "timings": [fit.record, predict.record],
        "final": {"te": test_error, "tr": train_error},
        "results": {
            "tr": {"merror": curve([train_error], [rounds])},
            "te": {"merror": curve([error(A_te, y_te, r) for r in x], x)},
        },
    }
