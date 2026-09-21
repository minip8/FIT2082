"""The PULSAR transform: its statistics, selection and device parity.

There is no upstream oracle in the tree. The statistics are checked against
direct numpy transcriptions of upstream's loops, and the Burg recursion
against a simulated AR process with known coefficients. The port was checked
once against upstream itself: identical feature counts, global features
within 1e-4 except near-constant partitions, where upstream's float32 moments
are rounding noise (see the README section).
"""

import numpy as np
import pytest
import torch

from fit2082.pulsar.pulsar import (
    Pulsar,
    burg,
    fisher_scores,
    hist_quantiles,
    local_stats,
    pool,
    represent,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def upstream_hist_quantile(row: np.ndarray, q: float, bins: int = 64) -> float:
    """`approx_median_iqr`'s loop from upstream, for one row and quantile."""

    lo, hi = row.min(), row.max()
    if hi == lo:
        return float(lo)
    width = (hi - lo) / np.float32(bins)
    hist = np.zeros(bins, dtype=np.int64)
    for value in row:
        hist[min(int((value - lo) / width), bins - 1)] += 1
    cum = 0
    for b in range(bins):
        cum += hist[b]
        if cum >= len(row) * q:
            return float(lo + (b + 0.5) * width)
    raise AssertionError


# == statistics ================================================================


@pytest.mark.parametrize("n", [1, 2, 7, 11, 64, 301, 600])
def test_hist_quantiles_match_upstream_loop(n):

    rng = np.random.default_rng(n)
    X = rng.standard_normal((200, n)).astype(np.float32)
    X[0] = 3.0  # constant row

    got = hist_quantiles(torch.tensor(X), (0.25, 0.5, 0.75))

    for q, column in zip((0.25, 0.5, 0.75), got, strict=True):
        want = [upstream_hist_quantile(row, q) for row in X]
        np.testing.assert_allclose(column.numpy(), want, rtol=1e-5, atol=1e-6)


def test_local_stats_by_hand():

    rng = np.random.default_rng(0)
    S = rng.standard_normal((5, 3, 9)).astype(np.float32)

    T = local_stats(torch.tensor(S)).numpy()

    assert T.shape == (5, 7, 3)
    np.testing.assert_allclose(T[:, 0], S.mean(-1), atol=1e-6)
    np.testing.assert_allclose(T[:, 1], S.std(-1), atol=1e-5)
    slope = np.polyfit(np.arange(9), S.reshape(-1, 9).T, 1)[0].reshape(5, 3)
    np.testing.assert_allclose(T[:, 2], slope, atol=1e-5)
    np.testing.assert_allclose(T[:, 3], S.min(-1))
    np.testing.assert_allclose(T[:, 4], S.max(-1))


def test_pool_crossings_and_above_mean():

    Y = torch.tensor([[0.0, 2.0, 0.0, 2.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])

    pooled = pool(Y)
    crossing, above = pooled[:, 6], pooled[:, 7]

    # mean 0.8: every step crosses it, 2 of 5 values are above it
    assert crossing.tolist() == [1.0, 0.0]
    assert above.tolist() == pytest.approx([0.4, 0.0])
    # a constant row has no spread and no slope
    assert pooled[1, 4] == 0 and pooled[1, 5] == 0 and pooled[1, 8] == 0


def test_burg_recovers_ar2():

    torch.manual_seed(0)
    phi = (0.6, -0.3)
    n = 4096
    e = torch.randn(64, n, dtype=torch.float64)
    x = torch.zeros_like(e)
    for t in range(2, n):
        x[:, t] = phi[0] * x[:, t - 1] + phi[1] * x[:, t - 2] + e[:, t]

    ar = burg(x.float(), 2).mean(0)

    assert ar.tolist() == pytest.approx(phi, abs=0.02)


def test_burg_of_a_constant_series_is_zero():

    assert (burg(torch.ones(3, 24), 8) == 0).all()


def test_periodogram_keeps_half_the_spectrum():

    for n in (23, 24):
        assert represent(torch.randn(2, n), "periodogram").shape[-1] == n // 2


# == selection =================================================================


def test_fisher_scores_match_a_direct_computation():

    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 6))
    y = rng.integers(0, 4, 300)
    X[:, 2] += 3 * y  # informative
    X[:, 5] = 1.0  # constant

    count = torch.tensor(np.bincount(y, minlength=4), dtype=torch.float64)
    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    total = torch.zeros(4, 6, dtype=torch.float64).index_add_(0, yt, Xt)
    squares = torch.zeros(4, 6, dtype=torch.float64).index_add_(0, yt, Xt**2)

    got = fisher_scores(count, total, squares).numpy()

    mu = X.mean(0)
    want = []
    for j in range(6):
        num = den = 0.0
        for k in range(4):
            sub = X[y == k, j]
            sd = max(sub.std(ddof=1), 1e-4)
            num += len(sub) * (sub.mean() - mu[j]) ** 2
            den += len(sub) * sd**2
        want.append(0.0 if num == 0 or den == 0 else num / den)

    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-12)
    assert got.argmax() == 2 and got[5] == 0


def labelled(num: int, channels: int, length: int, seed: int = 0):

    generator = torch.Generator().manual_seed(seed)
    Y = torch.randint(0, 3, (num,), generator=generator)
    X = torch.randn(num, channels, length, generator=generator)

    return X, Y


def test_fit_over_batches_equals_fit_on_their_concatenation():

    X, Y = labelled(96, 1, 24)

    split = Pulsar().fit([(X[:40], Y[:40]), (X[40:], Y[40:])])
    whole = Pulsar().fit([(X, Y)])

    assert torch.equal(split.selected, whole.selected)
    torch.testing.assert_close(split.transform(X), whole.transform(X))


def test_selection_keeps_an_informative_series():

    X, Y = labelled(120, 1, 24)
    X[..., :12] += 4 * Y.view(-1, 1, 1)

    pulsar = Pulsar().fit([(X, Y)])
    Z = pulsar.transform(X)

    # the most informative kept column separates the classes almost perfectly
    varying = Z[:, Z.std(0) > 0].numpy()
    best = max(abs(np.corrcoef(z, Y.numpy())[0, 1]) for z in varying.T)
    assert best > 0.9


def test_chunking_does_not_change_the_features():

    X, Y = labelled(64, 1, 24)

    whole = Pulsar().fit([(X, Y)])
    chunked = Pulsar(chunk_elements=1).fit([(X, Y)])

    assert torch.equal(whole.selected, chunked.selected)
    torch.testing.assert_close(whole.transform(X), chunked.transform(X))


# == shapes and devices ========================================================


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("channels", "length"), [(1, 24), (10, 23), (3, 60), (1, 5), (1, 2)]
)
def test_transform_shape_and_device(device, channels, length):

    X, Y = labelled(32, channels, length)
    X, Y = X.to(device), Y.to(device)

    pulsar = Pulsar()
    Z = pulsar.fit_transform(X, Y)

    local = channels * pulsar.num_local
    assert Z.shape == (32, channels * pulsar.num_global + int(0.4 * local))
    assert Z.shape[1] == pulsar.num_features
    assert Z.device.type == device
    assert torch.isfinite(Z).all()

    # fitted scaling: training columns come out centred
    assert Z.mean(0).abs().max() < 1e-3


def test_transform_rejects_a_different_length():

    X, Y = labelled(16, 1, 24)
    pulsar = Pulsar().fit([(X, Y)])

    with pytest.raises(AssertionError):
        pulsar.transform(torch.randn(4, 1, 25))


def test_operator_draws_follow_the_seed():

    X, Y = labelled(32, 1, 24)

    a = Pulsar(seed=1).fit([(X, Y)]).transform(X)
    b = Pulsar(seed=1).fit([(X, Y)]).transform(X)
    c = Pulsar(seed=2).fit([(X, Y)]).transform(X)

    assert torch.equal(a, b)
    assert a.shape != c.shape or not torch.equal(a, c)


@pytest.mark.skipif(len(DEVICES) < 2, reason="needs both cpu and cuda")
def test_pulsar_cpu_and_cuda_agree():

    X, Y = labelled(64, 1, 60)

    on_cpu = Pulsar().fit([(X, Y)])
    on_gpu = Pulsar().fit([(X.cuda(), Y.cuda())])

    # near-ties in the Fisher ranking may swap a column or two
    shared = np.intersect1d(on_cpu.selected.numpy(), on_gpu.selected.cpu().numpy())
    assert len(shared) >= 0.99 * len(on_cpu.selected)
    assert len(on_cpu.selected) == len(on_gpu.selected)

    G = on_cpu.channels * on_cpu.num_global
    Zc = on_cpu.transform(X)[:, :G]
    Zg = on_gpu.transform(X.cuda())[:, :G].cpu()
    # histogram quantiles can land a bin apart under different rounding
    close = torch.isclose(Zc, Zg, atol=1e-3, rtol=1e-3).float().mean()
    assert close > 0.995


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_compiled_pulsar_agrees_with_eager_at_the_tie_level():
    """`compile=True` reorders float ops, so it may tip a value across a
    histogram bin or threshold, but must otherwise give the same features."""

    generator = torch.Generator().manual_seed(0)
    Y = torch.randint(0, 3, (256,), generator=generator)
    X = torch.randn(256, 1, 60, generator=generator).cumsum(-1)
    X = X + Y.view(-1, 1, 1)  # informative, so the Fisher ranking has few ties
    X, Y = X.cuda(), Y.cuda()

    eager = Pulsar().fit([(X, Y)])
    compiled = Pulsar(compile=True).fit([(X, Y)])

    shared = np.intersect1d(eager.selected.cpu(), compiled.selected.cpu())
    assert len(shared) >= 0.99 * len(eager.selected)

    G = eager.channels * eager.num_global
    close = torch.isclose(
        eager.transform(X)[:, :G], compiled.transform(X)[:, :G], atol=1e-3, rtol=1e-3
    )
    assert close.float().mean() > 0.97
