import pytest
import numpy as np
import torch as th
from ..utils4t import float_dtypes, lap_types, devices
from thgsp.sampling.ess import ess_sampling, power_iteration, power_iteration4min, ess
from thgsp.graphs import rand_udg, laplace
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh

np.set_printoptions(precision=5)
th.set_printoptions(precision=5)
N = 30


@pytest.mark.parametrize('dtype', float_dtypes[1:])
@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('lap', lap_types)
class TestPowerIteration:
    def test_power_iteration_max(self, dtype, device, lap):
        k = 2
        g = rand_udg(N, dtype=dtype, device=device).set_value_(None)
        L = laplace(g, lap_type=lap)
        Sc = th.randint(N, (N // 2,))
        val, vec = power_iteration(L, Sc, k, num_iter=100)
        L = L.to_scipy()
        LtL = L.T ** k * L ** k
        reduced = LtL[np.ix_(Sc, Sc)]
        sigma, psi = eigsh(reduced, k=1, which="LM")
        sigma = sigma[0]
        print(f"\nth vs sci: {val:.4f}:{sigma:.4f}")
        print(f"th vs sci: \n", vec.view(-1), "\n", psi.ravel())
        assert (sigma - val) ** 2 < 1e-2

    def test_power_iteration_shift(self, dtype, device, lap):
        N = 20
        k = 3
        shift = 1.5
        g = rand_udg(N, dtype=dtype, device=device).set_value_(None)
        L = laplace(g, lap_type=lap)
        Sc = th.randint(N, (N // 2,))
        val, vec = power_iteration(L, Sc, k, shift=shift, num_iter=200)
        L = L.to_scipy()
        LtL = L.T ** k * L ** k
        reduced = LtL[np.ix_(Sc, Sc)] - shift * sparse.eye(len(Sc), dtype=LtL.dtype)
        sigma, psi = eigsh(reduced, k=1, which="LM")
        sigma = sigma[0]
        print(f"\nth vs sci: {val:.4f}:{sigma:.4f}")
        print(f"th vs sci: \n", vec.view(-1), "\n", psi.ravel())
        assert (sigma - val) ** 2 < 1e-2

    def test_power_iteration_min(self, dtype, device, lap):
        if lap is 'comb':
            pytest.skip('combinatorial Laplacian operator is not numerically stable according to many experiments')
        N = 200
        k = 2
        g = rand_udg(N, dtype=dtype, device=device).set_value_(None)
        L = laplace(g, lap_type=lap)
        Sc = th.randint(N, (N // 2,))
        val, vec = power_iteration4min(L, Sc, k, num_iter=500)
        L = L.to_scipy()
        LtL = L.T ** k * L ** k
        reduced = LtL[np.ix_(Sc, Sc)]
        sigma, psi = eigsh(reduced, k=1, which="SM")
        sigma = sigma[0]
        print(f"\nth vs sci: {val:.4f}:{sigma:.4f}")
        print(f"th vs sci: \n", vec.view(-1), "\n", psi.ravel())
        assert (sigma - val) ** 2 < 1e-3


@pytest.mark.parametrize('dtype', float_dtypes[1:])  # omit float32 since it may lead to ArpackNoConvergence error
@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('lap', lap_types)
@pytest.mark.parametrize('M', [N // 2, N])
class TestEss:
    def test_ess_sampling(self, dtype, device, lap, M):
        k = 2
        g = rand_udg(N, dtype=dtype, device=device)
        L = laplace(g, lap_type=lap)
        S = ess_sampling(L, M, k)
        print(S)

    def test_ess(self, dtype, device, lap, M):
        k = 2
        g = rand_udg(N, dtype=dtype, device=device)
        L = laplace(g, lap_type=lap)
        S = ess(L, M, k)
        print(S)
