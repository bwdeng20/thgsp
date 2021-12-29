import pytest
import torch
import math
from thgsp.graphs.generators import random_graph
from ..utils4t import devices, float_dtypes, snr_and_mse
from thgsp.sampling.fastgsss import fastgsss, recon_fastssss


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("cheb", [True, False])
class TestFastGSSS:
    def test_fastgsss(self, device, dtype, cheb):
        N = 1000
        ds = 0.1
        order = 12
        M = 10
        g = random_graph(N, ds, dtype=dtype, device=device)
        S, T = fastgsss(g, M, N // 10, order=order, cheby=cheb)
        print(S)

    def test_fastgsss_rec(self, device, dtype, cheb):
        N = 8
        g = random_graph(N, 0.4, dtype=dtype, device=device)
        fs, U = g.spectral(lap_type="sym")

        M = 4
        bw = 4
        nu = 3
        c = torch.rand(bw, dtype=dtype, device=device)
        f_band = U[:, :bw] @ c
        f_band_noise = f_band + math.sqrt(5e-3) * torch.randn(
            N, dtype=dtype, device=device
        )

        K = 12
        S, T = fastgsss(g, M, bw, nu, cheb, order=K)
        f_hat = recon_fastssss(f_band_noise[S], S, T, order=K)
        s, m = snr_and_mse(f_hat, f_band)
        assert m < 0.5
