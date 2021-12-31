import pytest
import torch

from thgsp.graphs.generators import (
    knn,
    radius,
    rand_bipartite,
    rand_dg,
    rand_udg,
    random_graph,
)

from ..utils4t import devices, float_dtypes


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("density", [0.1, 0.6])
class TestRandGraph:
    def test_rand_udg(self, device, dtype, density):
        N = 12
        G = rand_udg(N, density, dtype, device)
        assert G.dtype() == dtype
        assert G.density() - density < 2 / N * (N - 1)

    def test_rand_dg(self, device, dtype, density):
        N = 16
        G = rand_dg(N, density, dtype, device)
        assert G.device() == device
        assert G.density() - density < 2 / N * (N - 1)
        x = G @ torch.rand(N, 2, dtype=dtype, device=device)
        assert x.shape == (N, 2)

    def test_rand_bipartite(self, device, dtype, density):
        N1 = 6
        N2 = 7
        rand_bipartite(N1, N2, density, dtype, device)

    def test_rand_test(self, device, dtype, density):
        N = 10
        G = random_graph(N, density, True, dtype, device)
        assert G.density() - density < 2 / N * (N - 1)
        G = random_graph(N, density, False, dtype, device)
        assert G.density() - density < 2 / N * (N - 1)


def test_check_symmetric():
    G = random_graph(6, 0.4)
    assert G.is_symmetric()
    print("\n", G.to_dense())


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("loop", [True, False])
class TestXy2Graph:
    def test_knn_graph(self, dtype, device, loop):
        N = 10
        xy = torch.rand(N, 2, dtype=dtype)

        graph = knn(xy, 4, loop, dtype=dtype, device=device)
        assert graph.sizes() == [N, N]
        assert graph.device() == device
        assert graph.dtype() == dtype
        if loop:
            assert graph.to_scipy().diagonal().sum() == N
        else:
            assert graph.to_scipy().diagonal().sum() == 0

    def test_radius_graph(self, dtype, device, loop):
        N = 12
        xy = torch.rand(N, 2, dtype=dtype)
        adj = radius(xy, loop=loop, dtype=dtype, device=device)

        assert adj.sizes() == [N, N]
        assert adj.device() == device
        assert adj.dtype() == dtype
        dense_adj = adj.to_dense()
        # radius graph is symmetric
        assert (dense_adj - dense_adj.t()).abs().sum() == 0
        if loop:
            assert adj.to_scipy().diagonal().sum() == N
        else:
            assert adj.to_scipy().diagonal().sum() == 0
