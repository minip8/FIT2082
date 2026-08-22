"""Device parity for the QUANT transform.

`f_quantile` used to build its quantile positions on the default (CPU) device,
which made `Quant` unusable on GPU input -- and so unusable as a front end for
`fit2082.boost`. These tests pin the fix.
"""

import pytest
import torch

from fit2082.quant.quant import Quant, f_quantile

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
