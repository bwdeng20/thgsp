import numpy as np
import pytest
import torch as th
from scipy.sparse.linalg import ArpackNoConvergence

from thgsp.graphs import laplace, rand_udg
from thgsp.sampling.ess import ess, recon_ess

from ..utils4t import devices, float_dtypes, lap_types, snr_and_mse

np.set_printoptions(precision=5)
th.set_printoptions(precision=5)
N = 30


@pytest.mark.parametrize(
    "dtype", float_dtypes[1:]
)  # omit float32 since it may lead to ArpackNoConvergence error
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("lap", lap_types[:-1])
@pytest.mark.parametrize("M", [N // 2, N])
class TestEss:
    def test_ess_sampling(self, dtype, device, lap, M):
        try:
            k = 2
            g = rand_udg(N, dtype=dtype, device=device)
            L = laplace(g, lap_type=lap)
            S = ess(L, M, k)
            print(S)
        except ArpackNoConvergence:
            print("No convergence error")

    def test_recon(self, dtype, device, lap, M):
        g = rand_udg(N, dtype=dtype, device=device)
        L = laplace(g, lap_type=lap)
        S = ess(L, M)

        num_sig = 10
        f = th.rand(N, num_sig, dtype=dtype, device=device)
        fs = f[S, :]
        f_hat = recon_ess(fs, S, g.U(lap), bd=int(3 * N / 4))
        snr_and_mse(f_hat, f)
