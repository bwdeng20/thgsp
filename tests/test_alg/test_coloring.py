import pytest
from thgsp.alg.coloring import dsatur_py, check_coloring, dsatur_cpp, dsatur
from thgsp.graphs.generators import rand_udg, torch
from thgsp.visual.plotting import draw_cn

from ..utils4t import devices, plot


def test_dsatur_py():
    for _ in range(20):
        N = 13
        G = rand_udg(N, 0.3)
        pos = torch.rand(N, 2)
        vtx_color = dsatur_py(G)
        assert check_coloring(G, vtx_color)
        assert not check_coloring(G, [0] * N)
        draw_cn(G, pos=pos, node_color=vtx_color)
    plot()


def test_dsatur_cpp():
    for _ in range(20):
        N = 13
        G = rand_udg(N, 0.3)
        pos = torch.rand(N, 2)
        vtx_color = dsatur_cpp(G)
        assert check_coloring(G, vtx_color)
        assert not check_coloring(G, [0] * N)
        draw_cn(G, pos=pos, node_color=vtx_color)
    plot()


@pytest.mark.parametrize("device", devices)
def test_dsatur(device):
    N = 10
    G = rand_udg(N, 0.2, device=device)
    pos = torch.rand(N, 2)
    vtx_color = dsatur(G)
    assert check_coloring(G, vtx_color)
    assert not check_coloring(G, [0] * N)
    draw_cn(G, pos=pos, node_color=vtx_color)
    plot()
