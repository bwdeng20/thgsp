from typing import Optional

import numpy as np
from scipy.sparse import eye, lil_matrix
from torch_sparse import SparseTensor

from thgsp.typing import VertexColor

from ._utils import bipartite_mask


def osglm(
    A: SparseTensor, lc: Optional[int] = None, vtx_color: Optional[VertexColor] = None
):
    r"""
    The oversampled bipartite graph approximation method proposed in [1]_

    Parameters
    ----------
    A:        SparseTensor
      The adjacent matrix of graph.
    lc:       int
      The ordinal of color marking the boundary such that all nodes with a smaller color
      ordinal are grouped into the low-pass channel while those with a larger color
      ordinal are in the high-pass channel.
    vtx_color:iter
      The graph coloring result

    Returns
    -------
    bptG:         lil_matrix
              The oversampled graph(with additional nodes)
    beta :        np.ndarray
    append_nodes: np.ndarray
              The indices of those appended nodes
    vtx_color:    np.ndarray
              The node colors

    References
    ----------
    .. [1]  Akie Sakiyama, et al, "Oversampled Graph Laplacian Matrix for Graph Filter
            Banks", IEEE trans on SP, 2016.

    """
    if vtx_color is None:
        from thgsp.alg import dsatur

        vtx_color = dsatur(A)
    vtx_color = np.asarray(vtx_color)
    n_color = max(vtx_color) + 1

    if lc is None:
        lc = n_color // 2
    assert 1 <= lc < n_color

    A = A.to_scipy(layout="csr").tolil()
    # the foundation bipartite graph Gb
    Gb = lil_matrix(A.shape, dtype=A.dtype)
    N = A.shape[-1]

    bt = np.in1d(vtx_color, range(lc))
    idx_s1 = np.nonzero(bt)[0]  # L
    idx_s2 = np.nonzero(~bt)[0]  # H

    mask = bipartite_mask(bt)  # the desired edges
    Gb[mask] = A[mask]
    A[mask] = 0
    eye_mask = eye(N, N, dtype=bool)
    A[eye_mask] = 1  # add vertical edges

    degree = A.sum(0).getA1()  # 2D np.matrix -> 1D np.array
    append_nodes = (degree != 0).nonzero()[0]

    Nos = len(append_nodes) + N  # oversampled size
    bptG = [lil_matrix((Nos, Nos), dtype=A.dtype)]  # the expanded graph
    bptG[0][:N, N:] = A[:, append_nodes]
    bptG[0][:N, :N] = Gb
    bptG[0][N:, :N] = A[append_nodes, :]

    beta = np.zeros((Nos, 1), dtype=bool)
    beta[idx_s1, 0] = 1
    # appended nodes corresponding to idx_s2 are assigned to
    # the L channel of oversampled graph with idx_s1
    _, node_ordinal_append, _ = np.intersect1d(
        append_nodes, idx_s2, return_indices=True
    )
    beta[N + node_ordinal_append, 0] = 1
    return bptG, beta, append_nodes, vtx_color
