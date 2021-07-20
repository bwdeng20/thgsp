import pytest
import torch
from thgsp.graphs.generators import rand_udg, rand_bipartite,Graph
from ..utils4t import float_dtypes, lap_types,plot
from scipy.sparse import coo_matrix
from thgsp.bga.utils import is_bipartite_fix, laplace


class TestIbFix:
    def test_bipartite(self):
        n_sample = 7
        N1, N2 = 4, 6
        N = N1 + N2
        for i in range(n_sample):
            B = rand_bipartite(N1, N2)
            flag, vtx_color, _ = is_bipartite_fix(
                B.to_scipy("csr"), fix_flag=False)
            assert flag
            NB = rand_udg(N, 0.8)  # complete graph must be non-bipartite
            flag, vtx_color, _ = is_bipartite_fix(
                NB.to_scipy("csr"), fix_flag=False)
            assert not flag

    def test_bipartite_fix(self):
        N = 12
        B = rand_udg(N, 0.3)
        flag, vtx_color, Bb = is_bipartite_fix(
            B.to_scipy('csr'), fix_flag=True)
        assert flag  # if fix_flag, always bipartite
        assert is_bipartite_fix(Bb)[0]

        print('the last one\n', B)
        # visual
        from thgsp.visual import draw_cn, draw
        import matplotlib.pyplot as plt
        pos = torch.rand(N, 2)
        _, axes = plt.subplots(1, 2)
        draw(B, pos, ax=axes[0])
        draw_cn(Graph(Bb), pos, vtx_color, ax=axes[1])
        plot()

    def test_bipartite_fix_th(self):
        N1, N2 = 4, 6
        N = N1 + N2
        B = rand_udg(N, 0.6)
        flag, vtx_color, Bb = is_bipartite_fix(
            B.to_dense(), fix_flag=False)
        assert not flag  # if fix_flag, always bipartite

        B = rand_bipartite(N1, N2, 0.6)
        flag, vtx_color, _ = is_bipartite_fix(B.to_dense(), fix_flag=False)
        assert flag


@pytest.mark.parametrize('dtype', float_dtypes)
@pytest.mark.parametrize('lap_type', lap_types)
def test_lil_laplace(dtype, lap_type):
    N = 8
    g = rand_udg(N, dtype=dtype)
    coo_m = g.to_scipy('coo')
    L = laplace(coo_m, lap_type)
    assert L.shape == (N, N)
    assert isinstance(L, coo_matrix)
