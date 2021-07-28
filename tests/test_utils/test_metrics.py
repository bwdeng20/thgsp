import torch
import pytest
from ..utils4t import float_dtypes, devices, snr, mse


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", float_dtypes)
def test_snr(device, dtype):
    N = 5
    M = 2
    sigma = 0.1
    signal1 = torch.rand(N, M, dtype=dtype, device=device)
    noise = sigma * torch.rand(N, M, dtype=dtype, device=device)
    fn = signal1 + noise
    s = snr(fn, signal1)
    e = mse(fn, signal1)

    assert s > 1 / sigma
    assert e < 0.01

    with pytest.raises(AssertionError):
        snr(torch.rand(2, M), signal1)

    with pytest.raises(AssertionError):
        mse(torch.rand(2, M), signal1)

    signal2 = torch.rand(N, 1, dtype=dtype, device=device)
    fn2 = noise + signal2
    s1 = snr(fn2, signal2)
    e2 = mse(fn2, signal2)

    assert s1 > 1 / sigma
    assert e2 < 0.01
