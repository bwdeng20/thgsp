import numpy as np
from scipy.sparse import lil_matrix
from torch_sparse import SparseTensor

from thgsp.alg.coloring import dsatur
from .utils import new_order, distribute_color, bipartite_mask


def harary(A: SparseTensor, vtx_color=None, threshold=0.97):
    """
    Harary bipartite decomposition

    Parameters
    ----------
    A:      :py:class:`SparseTensor`
        The adjacency matrix
    vtx_color: array_like, optional
        All valid type for :py:func:`np.asarray` is desired, including :py:class:`torch.Tensor` on cpu. If None,
        this function will invoke :py:func:`thgsp.alg.dsatur` silently.

    threshold: float, optional

    Returns
    -------
    bptG:    array
        A array consisting of :obj`M` bipartite subgraphs formatted as :class:`scipy.sparse.lil_matrix`.
    beta:   array
        :obj:`beta[:,i]` is the bipartite set indicator of :obj:`i`-th subgraph.
    beta_dist:  array
        A table showing the relationship between :obj:`beta` and  :obj:`channels`
    new_vtx_color:  array
        The node colors
    mapper:     dict
        Map **new_vtx_color** to the original ordinal group. For example mapper={1:2, 2:3, 3:1} will
        map 1,2 and 3-th color to 2,3 and 1, respectively.
    """
    if vtx_color is None:
        vtx_color = dsatur(A)
    vtx_color = np.asarray(vtx_color)
    n_color = max(vtx_color) + 1
    if n_color > 256:
        raise RuntimeError(
            "Too many colors will lead to a too complicated channel division")

    A = A.to_scipy(layout='csr').tolil()
    M = int(np.ceil(np.log2(n_color)))  # the number of bipartite graphs
    N = A.shape[-1]  # the number of nodes

    new_color_ordinal = new_order(n_color)
    mapper = {c: i for i, c in enumerate(new_color_ordinal)}
    new_vtx_color = [mapper[c] for c in vtx_color]

    beta_dist = distribute_color(n_color, M)
    bptG = [lil_matrix((N, N), dtype=A.dtype) for _ in range(M)]
    link_weights = -np.ones(M)
    beta = np.zeros((N, M), dtype=bool)
    for i in range(M):
        colors_L = (beta_dist[:, i] == 1).nonzero()[0]
        bt = np.in1d(new_vtx_color, colors_L)

        beta[:, i] = bt
        mask = bipartite_mask(bt)
        bpt_edges = A[mask]
        bptG[i][mask] = bpt_edges
        link_weights[i] = bpt_edges.sum()
        A[mask] = 0

    ratio_link_weights = link_weights.cumsum(0) / link_weights.sum()
    bpt_idx = (ratio_link_weights >= threshold).nonzero()[0]
    M1 = bpt_idx[0] + 1
    bptG = bptG[:M1]
    max_color = np.power(2, M1)

    beta_dist = distribute_color(max_color, M1)
    beta = beta[:, :M1]
    return bptG, beta, beta_dist, vtx_color, mapper
