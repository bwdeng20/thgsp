import numpy as np
from scipy.sparse import lil_matrix
from thgsp.datasets import Toy
from thgsp.sampling import ess_sampling
from thgsp.visual import show_transform
from thgsp.graphs import Graph

# param
sigma = 1.5
lap_type = "comb"

# load data
g0 = Toy(download=True)[0]
exponent = -g0.extra["distances"] ** 2 / (2 * sigma ** 2)
W = np.exp(exponent)
edge_idx, _ = g0.edge_info()
tmp = W[edge_idx[0], edge_idx[1]]
W[...] = 0
W[edge_idx[0], edge_idx[1]] = tmp
g = Graph(W)

# prepare
fs, U = g.spectral(lap_type=lap_type)
bands = np.linspace(fs[0], fs[-1], num=9)
bands2 = np.hstack(
    [bands[:-1, None], bands[1:, None]]
)  # both two kinds of bands are supported
N, M = U.shape
sampled_nodes = ess_sampling(g.L(lap_type), g.n_node, k=2)
highlights = lil_matrix((M, N))
highlights[range(M), sampled_nodes] = 1
cluster = np.concatenate([np.ones(12), np.ones(8) * 2, np.ones(7) * 3])
# plot

fig, _, _ = show_transform(g, U.t(), fs, highlights, cluster=2, bands=bands)

# Before showing the figure, you can adjust the figure on many aspects including but not limited to
# font, text, colors of axes, using APIs provided by `plotly.go.Figure` class.
fig.show()
