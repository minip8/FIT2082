"""Checks that the torch implementation matches the numba reference.

`fit2082.demo.boost` is kept as the oracle: it is the original, independently
written implementation, so agreeing with it is real evidence and not a tautology.

Note that float32 scatter-adds on CUDA accumulate in nondeterministic order, so
results wobble in the last bits between runs. Everything here compares with a
tolerance. Set `torch.use_deterministic_algorithms(True)` if you need bitwise
repeatability (at a cost in speed).
"""

import numpy as np
import pytest
import torch

from fit2082.boost import HashBoost
from fit2082.demo.boost import NewHashBoost, _predict_multi0

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class ReplaySplitter:
    """Splitter that replays a recorded sequence of hashes.

    Doubles as a demonstration of the `Splitter` seam.
    """

    def __init__(self, feature_indices, midpoints, device):

        self.feature_indices = torch.as_tensor(
            feature_indices.astype(np.int64), device=device
        )
        self.midpoints = torch.as_tensor(midpoints.astype(np.float32), device=device)
        self.index = 0

    def propose(self, X, Y, probabilities, num_bits):

        i = self.index
        self.index += 1

        return self.feature_indices[i], self.midpoints[i]


def _data(n=512, p=16, k=5, seed=0):

    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, p), dtype=np.float32)
    Y = rng.integers(0, k, n).astype(np.int32)

    return X, Y, k


def _reference(X, Y, k, num_rounds, num_bits=8, lr=0.1):
    """Fit the numba model and return it plus its logits."""

    model = NewHashBoost(
        num_classes=np.int32(k),
        num_pairs_per_hash=np.int32(num_bits),
        lr=np.float32(lr),
        max_num_hashes=num_rounds + 1,
    )

    for _ in range(num_rounds):
        model.fit_batch(X, Y)

    logits = _predict_multi0(
        X=X,
        feature_indices=model.feature_indices,
        midpoints=model.midpoints,
        num_rounds=np.int32(num_rounds),
        num_classes=np.int32(k),
        logits0=model.logits0,
    )

    return model, logits


def _replay(
    X, Y, k, reference, num_rounds, device, round_chunk=None, num_bits=8, lr=0.1
):
    """Fit HashBoost on the reference model's hash sequence."""

    model = HashBoost(
        num_classes=k,
        num_bits=num_bits,
        lr=lr,
        max_num_hashes=num_rounds + 1,
        device=device,
        round_chunk=round_chunk,
        splitter=ReplaySplitter(
            reference.feature_indices[:num_rounds],
            reference.midpoints[:num_rounds],
            device,
        ),
    )

    for _ in range(num_rounds):
        model.fit_batch(X, Y)

    return model


# == equivalence ===============================================================


@pytest.mark.parametrize("device", DEVICES)
def test_matches_numba_reference(device):

    X, Y, k = _data()
    num_rounds = 25

    reference, expected = _reference(X, Y, k, num_rounds)
    model = _replay(X, Y, k, reference, num_rounds, device)

    actual = model.predict(X).cpu().numpy()

    assert np.abs(expected - actual).max() < 1e-4
    assert (expected.argmax(-1) == actual.argmax(-1)).all()

    # the leaf tables themselves, not just their sum
    tables = model.tables.logits[:num_rounds].cpu().numpy()
    assert np.abs(reference.logits0[:num_rounds] - tables).max() < 1e-4


@pytest.mark.parametrize("device", DEVICES)
def test_predict_all_matches_numba_reference(device):

    X, Y, k = _data()
    num_rounds = 12

    reference, _ = _reference(X, Y, k, num_rounds)
    model = _replay(X, Y, k, reference, num_rounds, device)

    expected = reference.predict_all(X)
    actual = model.predict_all(X).cpu().numpy()

    assert actual.shape == expected.shape
    assert np.abs(expected - actual).max() < 1e-4


# == internal consistency ======================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("round_chunk", [8, 32, 128])
def test_round_chunk_invariance(device, round_chunk):
    """Chunking is a memory/throughput knob and must not change results.

    25 rounds is deliberately not a multiple of any chunk size here, so the
    short tail chunk is exercised in every case.
    """

    X, Y, k = _data()
    num_rounds = 25

    reference, _ = _reference(X, Y, k, num_rounds)

    baseline = _replay(X, Y, k, reference, num_rounds, device, round_chunk=None)
    chunked = _replay(X, Y, k, reference, num_rounds, device, round_chunk=round_chunk)

    assert torch.allclose(baseline.predict(X), chunked.predict(X), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(len(DEVICES) < 2, reason="needs both cpu and cuda")
def test_device_parity():

    X, Y, k = _data()
    num_rounds = 20

    reference, _ = _reference(X, Y, k, num_rounds)

    on_cpu = _replay(X, Y, k, reference, num_rounds, "cpu")
    on_gpu = _replay(X, Y, k, reference, num_rounds, "cuda")

    assert torch.allclose(
        on_cpu.predict(X), on_gpu.predict(X).cpu(), atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("device", DEVICES)
def test_staged_error_matches_predict_all(device):

    X, Y, k = _data()
    num_rounds = 15

    reference, _ = _reference(X, Y, k, num_rounds)
    model = _replay(X, Y, k, reference, num_rounds, device)

    expected = (
        (model.predict_all(X).cumsum(0).argmax(-1) != model._Y(Y))
        .to(torch.float32)
        .mean(-1)
    )

    assert torch.allclose(expected, model.staged_error(X, Y), atol=1e-6)


# == api =======================================================================


@pytest.mark.parametrize("device", DEVICES)
def test_accepts_numpy_and_torch(device):

    X, Y, k = _data()

    model = HashBoost(num_classes=k, max_num_hashes=4, device=device)

    model.fit_batch(X, Y)
    model.fit_batch(
        torch.as_tensor(X, device=device),
        torch.as_tensor(Y.astype(np.int64), device=device),
    )

    assert model.num_rounds == 2

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (X.shape[0], k)
    assert torch.allclose(
        probabilities.sum(-1), torch.ones(X.shape[0], device=probabilities.device)
    )


def test_raises_past_max_num_hashes():

    X, Y, k = _data()

    model = HashBoost(num_classes=k, max_num_hashes=2, device=DEVICES[-1])

    model.fit_batch(X, Y)
    model.fit_batch(X, Y)

    with pytest.raises(RuntimeError, match="max_num_hashes"):
        model.fit_batch(X, Y)
