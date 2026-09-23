"""Device parity and faithfulness for the QUANT transform.

`f_quantile` used to build its quantile positions on the default (CPU) device,
which made `Quant` unusable on GPU input -- and so unusable as a front end for
`fit2082.boost`. The first tests pin that fix. `IntervalModel` now takes the
quantiles of every interval of one length together, and the rest hold it to
upstream's interval-at-a-time `f_quantile`.
"""

import pytest
import torch

from fit2082.quant.quant import IntervalModel, Quant, f_quantile

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length", [1, 4, 24, 64])
def test_f_quantile_runs_on_device(device, length):

    X = torch.randn(8, 1, length, device=device)

    Z = f_quantile(X)

    assert Z.device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_quant_transform_runs_on_device(device):

    X = torch.randn(16, 1, 24, device=device)

    transform = Quant()
    Z = transform.fit_transform(X, None)

    assert Z.device.type == device
    assert Z.shape[0] == X.shape[0]
    assert torch.isfinite(Z).all()


@pytest.mark.skipif(len(DEVICES) < 2, reason="needs both cpu and cuda")
def test_quant_cpu_and_cuda_agree():

    X = torch.randn(16, 1, 24)

    on_cpu = Quant().fit_transform(X, None)
    on_gpu = Quant().fit_transform(X.cuda(), None)

    assert torch.allclose(on_cpu, on_gpu.cpu(), atol=1e-5)


# Lengths 1 and 4 take only medians, 15 and 24 mix medians with short quantile
# runs, and 150 and 540 have LenDB-sized windows. Three channels check that each
# interval's columns stay channel by channel, as upstream lays them out.
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("length", [1, 4, 15, 24, 150, 540])
def test_interval_model_matches_upstream(device, channels, length):

    X = torch.randn(32, channels, length, device=device)

    model = IntervalModel(length)
    Z = model.fit_transform(X)

    upstream = torch.cat(
        [f_quantile(X[..., a:b], div=model.div).squeeze(1) for a, b in model.intervals],
        -1,
    )

    torch.testing.assert_close(Z, upstream)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_quant_transform_does_not_sync():

    X = torch.randn(16, 3, 150, device="cuda")

    transform = Quant()
    transform.fit_transform(X)

    torch.cuda.set_sync_debug_mode("error")
    try:
        transform.transform(X)
    finally:
        torch.cuda.set_sync_debug_mode("default")
