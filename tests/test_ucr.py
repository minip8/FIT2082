"""The UCR loader, the UCR 112 list, and the benchmark's fixed round budget."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fit2082.ucr import (
    DEFAULT_PATH,
    SEED,
    UCR112,
    fit_extratrees,
    fit_hashboost,
    fit_xgboost,
    hashboost_batches,
    load_ucr,
)

# the archive's Missing_value_and_variable_length_datasets_adjusted/ folder
ADJUSTED = {
    "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ",
    "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend",
    "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "GesturePebbleZ1", "GesturePebbleZ2", "MelbournePedestrian",
    "PickupGestureWiimoteZ", "PLAID", "ShakeGestureWiimoteZ",
}  # fmt: skip


def _write(root: Path, name: str, split: str, labels, X) -> None:

    (root / name).mkdir(exist_ok=True)
    np.savetxt(
        root / name / f"{name}_{split}.tsv",
        np.column_stack([labels, X]),
        delimiter="\t",
    )


# == loading ===================================================================


def test_labels_are_encoded_once_over_train_and_test(tmp_path):

    rng = np.random.default_rng(0)

    # numeric labels, so 10 must sort after 2 -- and -1 first
    _write(tmp_path, "Toy", "TRAIN", [2, 10, -1, 2], rng.normal(size=(4, 6)))
    _write(tmp_path, "Toy", "TEST", [10, -1, 2], rng.normal(size=(3, 6)))

    data = load_ucr("Toy", str(tmp_path))

    assert data.X_tr.shape == (4, 1, 6) and data.X_tr.dtype == np.float32
    assert data.X_te.shape == (3, 1, 6) and data.X_te.dtype == np.float32
    assert data.classes.tolist() == [-1, 2, 10]
    assert data.y_tr.tolist() == [1, 2, 0, 1]
    assert data.y_te.tolist() == [2, 0, 1]


def test_series_values_survive_the_round_trip(tmp_path):

    X = np.arange(12, dtype=np.float64).reshape(3, 4) / 7

    _write(tmp_path, "Toy", "TRAIN", [0, 1, 0], X)
    _write(tmp_path, "Toy", "TEST", [1, 0, 1], X)

    data = load_ucr("Toy", str(tmp_path))

    assert np.allclose(data.X_tr[:, 0], X)


def test_missing_values_raise(tmp_path):

    X = np.ones((3, 5))
    X[1, 3:] = np.nan  # how the archive pads a variable-length series

    _write(tmp_path, "Toy", "TRAIN", [0, 1, 0], X)
    _write(tmp_path, "Toy", "TEST", [0, 1, 0], np.ones((3, 5)))

    with pytest.raises(ValueError, match="missing values or variable lengths"):
        load_ucr("Toy", str(tmp_path))


# == the 112 ===================================================================


def test_ucr112_is_the_bakeoff_set():

    assert len(UCR112) == len(set(UCR112)) == 112
    assert "Fungi" not in UCR112
    assert not ADJUSTED & set(UCR112)


@pytest.mark.skipif(not Path(DEFAULT_PATH).is_dir(), reason="UCR archive not in data/")
def test_ucr112_is_the_archive_less_the_adjusted_and_fungi():

    folders = {
        entry
        for entry in os.listdir(DEFAULT_PATH)
        if (Path(DEFAULT_PATH) / entry).is_dir() and not entry.startswith("Missing")
    }

    assert set(UCR112) == folders - ADJUSTED - {"Fungi"}

    for name in UCR112:
        for split in ("TRAIN", "TEST"):
            assert (Path(DEFAULT_PATH) / name / f"{name}_{split}.tsv").is_file()


# == the round budget ==========================================================


def _two_classes_per_batch(n: int, batch_size: int) -> np.ndarray:
    """Labels that alternate along the seeded batch order, so every batch of
    two or more rows holds both classes."""

    order = np.random.default_rng(SEED).permutation(n)

    y = np.empty(n, dtype=np.int64)
    y[order] = np.arange(n) % 2

    return y


def test_hashboost_cycles_its_batches_to_exactly_the_budget():

    Z = torch.randn(10, 5)
    y = _two_classes_per_batch(10, 4)

    entry = fit_hashboost(
        Z, y, Z, y, num_classes=2, rounds=7, batch_size=4, device="cpu", curve_every=5
    )

    # batches of 4, 4 and 2 rows, cycled for 7 rounds: two passes and a third
    assert entry["params"]["num_batches"] == 3
    assert entry["params"]["rounds"] == 7
    assert entry["params"]["passes"] == pytest.approx(7 / 3)

    assert entry["results"]["te"]["merror"]["x"] == [0, 5, 7]
    assert 0.0 <= entry["final"]["te"] <= 1.0

    # the curve's last point is the finished model, as "final" reports it
    assert entry["results"]["te"]["merror"]["y"][-1] == pytest.approx(
        entry["final"]["te"]
    )


@pytest.mark.parametrize("num_bits", [2, 4])
def test_hashboost_takes_its_bit_width(num_bits):

    Z = torch.randn(12, 5)
    y = _two_classes_per_batch(12, 12)

    entry = fit_hashboost(
        Z, y, Z, y, num_classes=2, rounds=3, device="cpu", num_bits=num_bits
    )

    assert entry["params"]["num_bits"] == num_bits
    assert entry["params"]["rounds"] == 3


def test_a_single_class_batch_raises_rather_than_hanging():

    # one row per batch is one class per batch, which HardPairSplitter could
    # never pair
    with pytest.raises(ValueError, match="single class"):
        hashboost_batches(torch.randn(6, 3), torch.tensor([0, 1] * 3), batch_size=1)


@pytest.mark.parametrize("fit", [fit_extratrees, fit_xgboost])
def test_baselines_report_their_finished_error(fit):

    Z = torch.randn(40, 6)
    y = (Z[:, 0] > 0).long().numpy()

    entry = fit(Z, y, Z, y) if fit is fit_extratrees else fit(Z, y, Z, y, "cpu")

    assert 0.0 <= entry["final"]["te"] <= 1.0
    assert entry["results"]["te"]["merror"]["y"][-1] == pytest.approx(
        entry["final"]["te"]
    )
