import pytest
import torch

from thgsp.graphs.generators import rand_bipartite, rand_udg
from ..utils4t import float_dtypes

from thgsp.bga.mfs import dkl, amfs1level, amfs, compute_sigma


@pytest.mark.parametrize('dt', [torch.double])
@pytest.mark.parametrize('p', [0.1, 0.8])
class TestUtils:
    def test_dkl(self, p, dt):
        N1 = 20
        N2 = 10
        delta = 0.1
        Blil = rand_bipartite(N1, N2, p, dt).to_scipy('csr').tolil()
        Sigma = compute_sigma(Blil, delta).tocsc()

        print(dkl(Blil, Sigma, delta))

    def test_compute_sigma(self, p, dt):
        N = 100
        delta = 0.1
        Acoo = rand_udg(N, p, dt).to_scipy('coo')
        Sigma = compute_sigma(Acoo, delta)
        assert Sigma.shape == (N, N)


@pytest.mark.parametrize('dt', float_dtypes)
class TestAmfs:
    def test_amfs1level(self, dt):
        N = 100
        delta = 0.1
        W = rand_udg(N, dtype=dt).to_scipy('csr').tolil().astype("d")
        Sigma = compute_sigma(W, delta)
        s1, s2 = amfs1level(W, Sigma, delta)
        print("\n|L|: ", s1)
        print("|H|: ", s2)

    def test_amfs(self, dt):
        from thgsp.bga.utils import is_bipartite_fix
        N = 100
        M = 2
        W = rand_udg(N, dtype=dt)
        bptG, beta = amfs(W, level=M)
        weights = []
        for i in range(len(bptG)):
            assert is_bipartite_fix(bptG[i])[0]
            weights.append(bptG[i].sum())
        assert beta.shape == (N, M)
        assert len(bptG) == M
        print("bptG  weights: ", weights)
        print("total weights: ", sum(weights))
        assert (sum(weights) - W.sum()).sum() < 1e-6
