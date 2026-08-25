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

from fit2082.boost import (
    BaggedHashBoost,
    HashBoost,
    HashTables,
    ObliquePartitioner,
    Readout,
    fit_readout,
)
from fit2082.boost.readout import RUNGS
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

    def propose(self, X, Y, probabilities, gradient, hessian, num_bits):

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


# == capacity, smoothing, bagging ==============================================


def _random_splitter(num_rounds, num_features, device, num_bits=8, seed=0):
    """Factory of ReplaySplitters over one fixed random split sequence.

    The splits are drawn once and shared, so two models built from the same
    factory see identical hashes and any difference between them comes from the
    feature under test rather than from the RNG.
    """

    rng = np.random.default_rng(seed)

    feature_indices = rng.integers(0, num_features, (num_rounds, num_bits))
    midpoints = rng.standard_normal((num_rounds, num_bits))

    return lambda: ReplaySplitter(feature_indices, midpoints, device)


@pytest.mark.parametrize("device", DEVICES)
def test_hashes_per_round_matches_repeated_fit_batch(device):
    """H hashes per batch must equal calling fit_batch H times."""

    X, Y, k = _data()
    make = _random_splitter(12, X.shape[1], device)

    grouped = HashBoost(
        num_classes=k,
        max_num_hashes=13,
        device=device,
        hashes_per_round=3,
        splitter=make(),
    )
    singly = HashBoost(num_classes=k, max_num_hashes=13, device=device, splitter=make())

    for _ in range(4):
        grouped.fit_batch(X, Y)
        for _ in range(3):
            singly.fit_batch(X, Y)

    assert grouped.num_rounds == singly.num_rounds == 12
    assert torch.allclose(grouped.predict(X), singly.predict(X), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_hashes_per_round_respects_max(device):

    X, Y, k = _data()

    model = HashBoost(
        num_classes=k, max_num_hashes=3, device=device, hashes_per_round=2
    )
    model.fit_batch(X, Y)

    with pytest.raises(RuntimeError, match="max_num_hashes"):
        model.fit_batch(X, Y)

    assert model.num_rounds == 3


@pytest.mark.parametrize("device", DEVICES)
def test_zero_shrinkage_is_unsmoothed(device):

    X, Y, k = _data()
    make = _random_splitter(10, X.shape[1], device)

    plain = HashBoost(num_classes=k, max_num_hashes=11, device=device, splitter=make())
    zero = HashBoost(
        num_classes=k,
        max_num_hashes=11,
        device=device,
        neighbour_shrinkage=0.0,
        splitter=make(),
    )

    for _ in range(10):
        plain.fit_batch(X, Y)
        zero.fit_batch(X, Y)

    assert torch.allclose(plain.tables.logits, zero.tables.logits, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_shrinkage_fills_unoccupied_buckets(device):
    """Smoothing spreads evidence to buckets the data never populated.

    Most buckets are empty -- on real data typically ~203 of 256 -- and an empty
    bucket contributes exactly zero. Borrowing from its one-bit-flip neighbours
    gives it a value, so an example whose code was never seen still gets a
    prediction. Note this does *not* shrink leaf magnitudes; it increases the
    fraction of buckets that say anything at all.
    """

    X, Y, k = _data()
    make = _random_splitter(10, X.shape[1], device)

    plain = HashBoost(num_classes=k, max_num_hashes=11, device=device, splitter=make())
    smoothed = HashBoost(
        num_classes=k,
        max_num_hashes=11,
        device=device,
        neighbour_shrinkage=0.5,
        splitter=make(),
    )

    for _ in range(10):
        plain.fit_batch(X, Y)
        smoothed.fit_batch(X, Y)

    a = plain.tables.logits[: plain.num_rounds]
    b = smoothed.tables.logits[: smoothed.num_rounds]

    assert torch.isfinite(b).all()
    assert not torch.allclose(a, b)
    assert (b != 0).float().mean() > (a != 0).float().mean() * 1.5


@pytest.mark.parametrize("device", DEVICES)
def test_bagging_with_one_estimator_matches_bare_model(device):

    X, Y, k = _data()
    make = _random_splitter(8, X.shape[1], device)

    bare = HashBoost(num_classes=k, max_num_hashes=9, device=device, splitter=make())
    bagged = BaggedHashBoost(
        num_estimators=1,
        num_classes=k,
        max_num_hashes=9,
        device=device,
        splitter=make(),
    )

    for _ in range(8):
        bare.fit_batch(X, Y)
        bagged.fit_batch(X, Y)

    assert bagged.num_rounds == bare.num_rounds
    assert torch.allclose(
        bagged.predict_proba(X), bare.predict_proba(X), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", DEVICES)
def test_bagging_averages_estimators(device):

    X, Y, k = _data()

    bagged = BaggedHashBoost(
        num_estimators=3, num_classes=k, max_num_hashes=6, device=device
    )

    for _ in range(5):
        bagged.fit_batch(X, Y)

    probabilities = bagged.predict_proba(X)

    assert bagged.num_estimators == 3
    assert torch.allclose(
        probabilities.sum(-1), torch.ones(X.shape[0], device=probabilities.device)
    )

    expected = torch.stack([e.predict_proba(X) for e in bagged.estimators]).mean(0)
    assert torch.allclose(probabilities, expected, atol=1e-6)


# == partition families ========================================================


@pytest.mark.parametrize("device", DEVICES)
def test_oblique_partitioner_trains(device):
    """The oblique family must be usable without touching the boosting core."""

    X, Y, k = _data()

    model = HashBoost(
        num_classes=k,
        max_num_hashes=21,
        device=device,
        partitioner=ObliquePartitioner(
            num_bits=8, max_num_hashes=21, device=torch.device(device)
        ),
    )

    for _ in range(20):
        model.fit_batch(X, Y)

    logits = model.predict(X)

    assert model.num_rounds == 20
    assert logits.shape == (X.shape[0], k)
    assert torch.isfinite(logits).all()

    # it must actually learn something, not just run
    assert (logits.argmax(-1) != model._Y(Y)).float().mean() < 0.9


@pytest.mark.parametrize("device", DEVICES)
def test_oblique_codes_use_feature_differences(device):
    """x[i] - x[j] <= m, which no single-feature threshold can express."""

    partitioner = ObliquePartitioner(
        num_bits=2, max_num_hashes=1, device=torch.device(device)
    )

    partitioner.left[0] = torch.tensor([0, 0], device=device)
    partitioner.right[0] = torch.tensor([1, 1], device=device)
    partitioner.midpoints[0] = torch.tensor([0.0, 5.0], device=device)

    # x[0] - x[1] is -1, 3 and 10 for the three rows
    X = torch.tensor([[1.0, 2.0], [5.0, 2.0], [12.0, 2.0]], device=device)

    codes = partitioner.encode(X.t().contiguous(), 0, 1)[0]

    # bit0 = (d <= 0), bit1 = (d <= 5)
    assert codes.tolist() == [0b11, 0b10, 0b00]


@pytest.mark.parametrize("device", DEVICES)
def test_state_dict_round_trip(device):

    X, Y, k = _data()

    model = HashBoost(num_classes=k, max_num_hashes=9, device=device)
    for _ in range(8):
        model.fit_batch(X, Y)

    restored = HashBoost(num_classes=k, max_num_hashes=9, device=device)
    restored.load_state_dict(model.state_dict())

    assert restored.num_rounds == model.num_rounds
    assert torch.allclose(restored.predict(X), model.predict(X), atol=1e-6)


# == mass-adaptive leaf smoothing ==============================================


@pytest.mark.parametrize("device", DEVICES)
def test_zero_tau_is_unsmoothed(device):

    X, Y, k = _data()
    make = _random_splitter(10, X.shape[1], device)

    plain = HashBoost(num_classes=k, max_num_hashes=11, device=device, splitter=make())
    zero = HashBoost(
        num_classes=k,
        max_num_hashes=11,
        device=device,
        shrinkage_tau=0.0,
        splitter=make(),
    )

    for _ in range(10):
        plain.fit_batch(X, Y)
        zero.fit_batch(X, Y)

    assert torch.allclose(plain.tables.logits, zero.tables.logits, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_adaptive_shrinkage_scales_with_bucket_mass(device):
    """The whole point: borrowing is inverse to the evidence a bucket already has.

    Two buckets with the same leaf value, one holding mass 200 and one holding
    mass 0.02, each with a neighbour pulling towards the other class. Under a
    fixed alpha both move by the same fraction, which is the likeliest reason
    the fixed version measures as noise -- it pays for the sparse buckets by
    damaging the dense ones. Under `shrinkage_tau` only the sparse one moves.
    """

    def build(tau):

        tables = HashTables(
            num_classes=2,
            num_bits=3,
            max_num_hashes=1,
            lr=1.0,
            device=torch.device(device),
            shrinkage_tau=tau,
        )

        def put(bucket, values):
            tables.stats[0, bucket] = torch.tensor(
                values, dtype=torch.float32, device=tables.stats.device
            )

        # [numerator (2 classes), denominator (2 classes)]
        put(0, [100.0, 0.0, 100.0, 100.0])  # heavy, class 0
        put(1, [0.01, 0.0, 0.01, 0.01])  # light, class 0, same leaf value
        put(2, [0.0, 60.0, 60.0, 60.0])  # neighbour of 0 (0^2), class 1
        put(3, [0.0, 60.0, 60.0, 60.0])  # neighbour of 1 (1^2), class 1

        tables.refresh_logits(1)

        return tables.logits[0].clone()

    plain = build(0.0)
    adaptive = build(10.0)

    # both buckets start at the same leaf value, so the shifts are comparable
    assert torch.allclose(plain[0], plain[1], atol=1e-4)

    heavy = (adaptive[0] - plain[0]).abs().max()
    light = (adaptive[1] - plain[1]).abs().max()

    assert light > heavy * 10


@pytest.mark.parametrize("device", DEVICES)
def test_adaptive_shrinkage_fills_unoccupied_buckets(device):

    X, Y, k = _data()
    make = _random_splitter(10, X.shape[1], device)

    plain = HashBoost(num_classes=k, max_num_hashes=11, device=device, splitter=make())
    smoothed = HashBoost(
        num_classes=k,
        max_num_hashes=11,
        device=device,
        shrinkage_tau=10.0,
        splitter=make(),
    )

    for _ in range(10):
        plain.fit_batch(X, Y)
        smoothed.fit_batch(X, Y)

    a = plain.tables.logits[: plain.num_rounds]
    b = smoothed.tables.logits[: smoothed.num_rounds]

    assert torch.isfinite(b).all()
    assert (b != 0).float().mean() > (a != 0).float().mean() * 1.5


# == heterogeneous bagging =====================================================


@pytest.mark.parametrize("device", DEVICES)
def test_bagging_without_overrides_is_unchanged(device):

    X, Y, k = _data()

    # one replay sequence per model, shared by its three estimators, so the two
    # models see identical hashes and only the `overrides` path differs
    make = _random_splitter(3 * 5, X.shape[1], device)

    def build(overrides):
        return BaggedHashBoost(
            num_estimators=3,
            overrides=overrides,
            num_classes=k,
            max_num_hashes=6,
            device=device,
            splitter=make(),
        )

    uniform = build(None)
    empty = build([{}])

    for _ in range(5):
        uniform.fit_batch(X, Y)
        empty.fit_batch(X, Y)

    assert torch.allclose(uniform.predict_proba(X), empty.predict_proba(X), atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_bagging_overrides_cycle_and_build_separate_partitioners(device):
    """Two patches over four estimators is ABAB, and no partitioner is shared.

    Sharing matters: a `Partitioner` *instance* owns the split tables for every
    round, so passing one would silently give all four estimators the same
    partitions -- an ensemble of identical members.
    """

    X, Y, k = _data()

    bagged = BaggedHashBoost(
        num_estimators=4,
        overrides=[{"num_bits": 6}, {"num_bits": 4, "partitioner": ObliquePartitioner}],
        num_classes=k,
        max_num_hashes=6,
        device=device,
    )

    for _ in range(5):
        bagged.fit_batch(X, Y)

    assert [e.num_bits for e in bagged.estimators] == [6, 4, 6, 4]
    assert [type(e.partitioner).__name__ for e in bagged.estimators] == [
        "AxisAlignedPartitioner",
        "ObliquePartitioner",
        "AxisAlignedPartitioner",
        "ObliquePartitioner",
    ]

    identities = {id(e.partitioner) for e in bagged.estimators}
    assert len(identities) == 4

    assert torch.isfinite(bagged.predict_proba(X)).all()


@pytest.mark.parametrize("device", DEVICES)
def test_estimator_agreement_is_a_fraction(device):

    X, Y, k = _data()

    bagged = BaggedHashBoost(
        num_estimators=3, num_classes=k, max_num_hashes=6, device=device
    )
    lone = BaggedHashBoost(
        num_estimators=1, num_classes=k, max_num_hashes=6, device=device
    )

    for _ in range(5):
        bagged.fit_batch(X, Y)
        lone.fit_batch(X, Y)

    agreement = bagged.estimator_agreement(X)

    assert 0.0 <= agreement <= 1.0
    assert lone.estimator_agreement(X) == 1.0

    # an ensemble of copies of one estimator agrees with itself perfectly
    for estimator in bagged.estimators[1:]:
        estimator.load_state_dict(bagged.estimators[0].state_dict())

    assert bagged.estimator_agreement(X) == pytest.approx(1.0)


# == refit readout =============================================================


def _readout_data(device, rounds=8, num_bits=4, n=2048, k=8):

    X, Y, _ = _data(n=n, p=16, k=k, seed=3)

    model = HashBoost(
        num_classes=k,
        num_bits=num_bits,
        max_num_hashes=rounds + 1,
        device=device,
    )

    for _ in range(rounds):
        model.fit_batch(X, Y)

    return model, X, Y


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rung", RUNGS)
def test_readout_warm_start_reproduces_the_model(device, rung):
    """Step zero must be the source model, or the fit is not a refit of it.

    This is what makes the comparison meaningful: any difference the readout
    shows is the joint fit, not a different parameterisation of the same table.
    """

    model, X, Y = _readout_data(device)

    readout, _ = fit_readout(model, X, Y, X, Y, rung=rung, steps=0)

    assert torch.allclose(readout.predict(X), model.predict(X), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_readout_forward_matches_predict_from_codes(device):
    """The readout's `embedding_bag` and the model's must agree on one table."""

    model, X, _ = _readout_data(device)

    readout = Readout(
        [model], model.tables.logits[: model.num_rounds].clone(), rung="table"
    )

    assert torch.allclose(readout.predict(X), model.predict(X), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rung", RUNGS)
def test_readout_lowers_training_loss(device, rung):
    """Warm-started and fit against the same objective, it can only improve."""

    model, X, Y = _readout_data(device)

    readout, _ = fit_readout(
        model, X, Y, rung=rung, steps=60, lr=0.05, lam=0.0, batch_size=512
    )

    Yd = torch.as_tensor(Y.astype(np.int64), device=readout.device)

    before = torch.nn.functional.cross_entropy(model.predict(X), Yd)
    after = torch.nn.functional.cross_entropy(readout.predict(X), Yd)

    assert after < before


@pytest.mark.parametrize("device", DEVICES)
def test_readout_spans_a_heterogeneous_ensemble(device):
    """Members with different bit widths stack into one padded table."""

    X, Y, k = _data(n=1024, k=6, seed=4)

    bagged = BaggedHashBoost(
        num_estimators=3,
        overrides=[
            {"num_bits": 4},
            {"num_bits": 6},
            {"partitioner": ObliquePartitioner},
        ],
        num_classes=k,
        num_bits=5,
        max_num_hashes=7,
        device=device,
    )

    for _ in range(6):
        bagged.fit_batch(X, Y)

    readout, info = fit_readout(bagged, X, Y, X, Y, rung="round", steps=20, lr=0.1)

    # every estimator's rounds, and the widest estimator's bucket count
    assert info["rounds"] == 3 * bagged.num_rounds
    assert readout.weights.shape[1] == 2**6

    assert torch.isfinite(readout.predict(X)).all()
