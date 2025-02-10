import numpy as np
import pytest
from scipy.sparse import lil_matrix

from thgsp.datasets import Toy
from thgsp.sampling import ess
from thgsp.visual import show_transform

from ..utils4t import plot, remove_downloaded_dataset


@pytest.mark.parametrize("embd", [None, "equispaced"])
def test_on_usc_toy(embd):
    g = Toy(download=True)[0]
    g.cache = True
    fs, U = g.spectral(lap_type="comb")
    bands = np.linspace(fs[0], fs[-1], num=9)
    N, M = U.shape
    print("\n", bands)
    sampled_nodes = ess(g.L("comb"), g.n_node, 4)
    highlights = lil_matrix((M, N))
    highlights[range(M), sampled_nodes] = 1
    fig, _, _ = show_transform(
        g, U.t(), fs, highlights, cluster=2, bands=bands, embedding=embd
    )
    plot(fig)
    remove_downloaded_dataset("GraphStructures-master")
