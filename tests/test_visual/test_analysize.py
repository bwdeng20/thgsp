import numpy as np
from scipy.sparse import lil_matrix
from thgsp.datasets import Toy
from thgsp.sampling import ess_sampling
from thgsp.visual import show_transform


class TestShowTransform:
    def test_on_usc_toy(self):
        g = Toy()[0]
        g.cache = True
        fs, U = g.spectral(lap_type="comb")
        bands = np.linspace(fs[0], fs[-1], num=9)
        N, M = U.shape
        print("\n", bands)
        sampled_nodes = ess_sampling(g.L("comb"), g.n_node, 4)
        highlights = lil_matrix((M, N))
        highlights[range(M), sampled_nodes] = 1
        fig, _, _ = show_transform(g, U.t(), fs, highlights, cluster=2, bands=bands)
        fig.show()
