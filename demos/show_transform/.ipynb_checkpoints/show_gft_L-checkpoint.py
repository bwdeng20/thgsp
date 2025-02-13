import numpy as np
from scipy.sparse import lil_matrix

from thgsp.datasets import Toy
from thgsp.sampling import ess
from thgsp.visual import show_transform

g = Toy(download=True)[0]
lap_type = "comb"
fs, U = g.spectral(lap_type=lap_type)
bands = np.linspace(fs[0], fs[-1], num=9)
bands2 = np.hstack(
    [bands[:-1, None], bands[1:, None]]
)  # both two kinds of bands are supported
N, M = U.shape
print(bands)
print(bands2)
sampled_nodes = ess(g.L(lap_type), g.n_node, k=2)
highlights = lil_matrix((M, N))
highlights[range(M), sampled_nodes] = 1
cluster = np.concatenate([np.ones(12), np.ones(8) * 2, np.ones(7) * 3])
fig, _, _ = show_transform(g, U.t(), fs, highlights, cluster=cluster, bands=bands)
fig.show()
