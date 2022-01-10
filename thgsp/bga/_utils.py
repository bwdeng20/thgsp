import math
import random
from collections import deque
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.sparse import coo_matrix, lil_matrix, spmatrix
from torch import Tensor
from torch_cluster import graclus_cluster
from torch_sparse import SparseTensor

from thgsp.utils.sparse_utils import pool_edge


def kernel_array_from_beta_dist(beta_dist, kernel1, kernel2, in_channels=1):
    f1c = np.where(beta_dist, kernel1, kernel2)
    f1c = np.transpose(f1c)
    return np.stack([f1c] * in_channels, axis=-1)


def cohere_color_idx(colors, color_group):
    """

    Parameters
    ----------
    colors:     Iterable
        Indices(ordinals) of colors. :obj:`np.array([0,2,1])` means the 0,2,1-th color
    color_group: dict
        :obj:`color_group[i]` is a :class:`np.ndarray` of node indices colored with
        :obj:`i`-th color.

    Returns
    -------
    LongTensor
                A Composed by node ordinals which are colored by any one among the
                :obj:`colors`.
    """
    if isinstance(colors, np.ndarray):
        colors = colors.tolist()
    nodes = [color_group[c] for c in colors]
    return torch.cat(nodes)


def new_order(n_color):
    M = math.ceil(math.log2(n_color))
    order = np.arange(0, 2 ** M, dtype=np.uint8)  # 0,1,...,2^M-1
    no = np.unpackbits(order.reshape(-1, 1), axis=1)[:, -M:]
    # no need to flip before pack since bitorder='little'
    no = np.packbits(no, axis=1, bitorder="little")
    return no[no < n_color]


def distribute_color_(n_color, M):
    if M == 1:
        beta_dist = np.arange(n_color - 1, -1, -1, dtype=np.uint8)
    else:
        N1 = math.ceil(n_color / 2)
        N2 = math.floor(n_color / 2)
        beta1 = distribute_color_(N1, M - 1)
        beta2 = distribute_color_(N2, M - 1)
        beta_dist = beta1 + 2 ** (M - 1)
        beta_dist = np.concatenate([beta_dist, beta2])
    return beta_dist


def distribute_color(n_color, M, th=False):
    beta_dist = distribute_color_(n_color, M)
    beta_dist = np.unpackbits(beta_dist.reshape(-1, 1), axis=1)[:, -M:]
    return torch.from_numpy(beta_dist) if th else beta_dist


def beta_dist2channel_name(beta_dist, reverse=False):
    """
    Generate the channel names(:math:`L` or :obj:`H`) for each channel according to
    :obj:`beta_dist`. By default, :obj:`0` and :obj:`1` means :obj:`H` and :obj:`L`,
    respectively.

    Parameters
    ----------
    beta_dist:  ByteTensor(2^M,M)
        An array, see the doc of :class:`thgsp.filters.Qmf` for the details.
    reverse:    bool,optional
        If True, :obj:`0` means :obj:`L`.

    Returns
    -------
    array
        An array composed by :obj:`H` and :obj:`L`. Each row corresponds to one channel.
    """
    channel_name410 = ["L", "H"]  # for 1, 0 respectively in beta_dist
    if reverse:
        channel_name410 = reversed(channel_name410)
    return np.where(beta_dist, *channel_name410)


def beta2color_group(beta, th=True):
    """Actually, beta2channel_group is more exact since "color" here is not
    exact graph coloring. However we can take it as an approximation of graph
    coloring.

    Parameters
    ----------
    beta:   BoolTensor(N,M)
        Bipartite partition indicator tensor. :obj:`N` and :obj:`M` are the node and
        bipartite subgraph numbers, respectively.
    th: bool, optional
        If True, the value of returned dict is :class:`Tensor` otherwise :class:`array`.

    Returns
    -------
    color_group:    dict
        :obj:`color_group[i]` is a :class:`LongTensor` (or :obj:`array`) of node indices
        colored with :obj:`i`-th color.
    beta_dist:      array
        An array, see the doc of :class:`thgsp.filters.QmfCore` for the details.
    """
    N, M = beta.shape
    n_channel = 2 ** M  # pseudo channel number
    beta = beta.numpy().astype("uint8")
    beta_dist = distribute_color(n_channel, M, th=False)  # n_channel x M np.array
    # n_channel x 1 --> n_channel
    colors = np.packbits(beta_dist, axis=1, bitorder="little").reshape(-1).tolist()

    # array(N x 1) --> List(N)
    node_colors = np.packbits(beta, axis=1, bitorder="little").reshape(-1)
    # some channels may be empty
    color_group = {color: None for color in colors}
    for c in color_group:
        index_c = np.where(node_colors == c)[0]  # unpack tuple
        if th:
            index_c = torch.from_numpy(index_c)
        color_group[c] = index_c
    return color_group, beta_dist


def beta2channel_mask(beta):
    """Generate a :obj:`(2^M, N)` bool mask tensor to indicate which
    channel(color group) one node belongs to. All channels have disjoint node
    sets and hence the sum along the channel dimension returns a all-one(True)
    :obj:`(N,)` tensor.

    .. note::
       This function is a common approach to generate channel mask for multi-channel
       wavelet filterbank. Either coloring or numerical based bipartite approximation
       algorithm can employ this function.

    Parameters
    ----------
    beta:   BoolTensor(N,M)
        Bipartite partition indicator tensor. :obj:`N` and :obj:`M` are the node and
        bipartite subgraph numbers, respectively.

    Returns
    -------
    mask:   BoolTensor(2**M,N)
        The indices of :obj:`True` in :obj:`i`-th row are nodes whose signal will be
        kept in :obj:`i`-th channel.
    beta_dist:  array(2**M,M))
        An array, see the doc of :class:`thgsp.filters.QmfCore` for the details.
    """
    N, M = beta.shape
    n_channel = 2 ** M  # pseudo channel number
    mask = torch.zeros(n_channel, N, dtype=torch.bool)
    if isinstance(beta, Tensor):
        beta = np.asarray(beta.cpu())
    beta = beta.astype("uint8")
    beta_dist = distribute_color(n_channel, M, th=False)  # n_channel x M np.array
    # n_channel x 1 --> n_channel
    colors = np.packbits(beta_dist, axis=1, bitorder="little").reshape(-1)

    # array(N x 1) --> List(N)
    node_colors = np.packbits(beta, axis=1, bitorder="little").reshape(-1)
    for i in range(n_channel):
        index_c = node_colors == colors[i]
        mask[i] = torch.from_numpy(index_c)
    return mask, beta_dist


def laplace(
    adj: spmatrix, lap_type: Optional[str] = None, add_loop: bool = False
) -> coo_matrix:
    M, N = adj.shape
    assert M == N
    dt = adj.dtype if adj.dtype in (np.float64, np.float32) else np.float32
    m = coo_matrix(adj, dtype=dt, copy=True)
    w = m.sum(0).getA1() - m.diagonal()  # - self_loop weight
    isolated_node_mask = w == 0
    if lap_type in (None, "sym"):
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        m.data /= w[m.row]
        m.data /= w[m.col]
        m.data *= -1
        if add_loop:
            m.setdiag(1)
        else:
            m.setdiag(1 - isolated_node_mask)

    elif lap_type == "comb":
        m.data *= -1
        if add_loop:
            m.setdiag(1)
        else:
            m.setdiag(1 - isolated_node_mask)

        m.setdiag(w)

    elif lap_type == "rw":
        w = np.where(isolated_node_mask, 1, w)
        m.data /= w[m.row]
        m.data *= -1
        if add_loop:
            m.setdiag(1)
        else:
            m.setdiag(1 - isolated_node_mask)
    else:
        raise RuntimeError("{} is not valid type of Laplacian".format(type(lap_type)))
    return m.astype(dt)


def bipartite_mask(bt, sparse=True):
    r"""
    Generate a bool mask matrix which can extract a bipartite subgraph from any graph
    (represented by a matrix) .

    Parameters
    ----------
    bt:    BoolTensor, array
        The indicator of two bipartite sets. All nodes within a same bipartite set is
        characterized by a same bool value, e.g.,True. Shape: `(N,)`, `N` is the number
        of graph nodes.
    sparse: bool
        If True, return lil_matrix.
    Returns
    -------
    lil_matrix,

    """
    if isinstance(bt, torch.BoolTensor):
        bt = bt.cpu().numpy()
    b1, b2 = np.meshgrid(bt, bt, sparse=True)
    return lil_matrix(b1 ^ b2) if sparse else b1 ^ b2


def is_bipartite_fix(
    A: Union[spmatrix, Tensor], fix_flag: bool = False
) -> Union[Tuple[bool, Iterable[int], spmatrix], Tuple[bool, Iterable[int], Tensor]]:
    if isinstance(A, spmatrix):
        return is_bipartite_fix_scipy(A, fix_flag)
    elif isinstance(A, Tensor):
        return is_bipartite_fix_th(A, fix_flag)
    else:
        raise TypeError(f"{type(A)} not supported!")


def is_bipartite_fix_scipy(
    A: spmatrix, fix_flag: bool = False
) -> Tuple[bool, List[int], spmatrix]:
    A = A.tolil()
    n_node = A.shape[-1]
    vtx2reach = set(range(n_node))
    vtx_color = [-1 for _ in range(n_node)]
    flag = True
    while len(vtx2reach) > 0:
        r = random.sample(list(vtx2reach), 1)[0]  # [r] --> r
        q = deque([r])
        while len(q) > 0:
            cur_node = q.popleft()
            vtx2reach.remove(cur_node)
            if vtx_color[cur_node] == -1:
                vtx_color[cur_node] = 0
            nbr = A[cur_node].rows[0]
            for i in nbr:
                if vtx_color[i] == -1:
                    vtx_color[i] = 1 - vtx_color[cur_node]
                    q.append(i)
                elif vtx_color[i] == vtx_color[cur_node]:
                    if fix_flag:
                        A[cur_node, i] = 0
                    else:  # do not fix
                        flag = False
                        return flag, vtx_color, A

    return flag, vtx_color, A


def is_bipartite_fix_th(
    A: Tensor, fix_flag: bool = False
) -> Tuple[bool, List[int], Tensor]:
    """Check if a graph is bipartite using BFS-based 2-color coloring and
    furthermore can fix a graph to a bipartite one by deleting edges bridging
    two nodes colored with a same color.

    Parameters
    ----------
    A:          Tensor
        The adjacency matrix
    fix_flag:   bool,optional
        If True, fix :obj:`A` to a bipartite graph by zeroing some elements in-place.
    Returns
    -------
    flag:   bool
        If True, the returned adjacency matrix represents a bipartite graph
    vxt_color:  list
        A list recording the vertex colors which are all 1 or 0.
    A:      Tensor
        The adjacency matrix of returned graph.
    """

    n_node = A.shape[-1]
    flag = True
    vtx2reach = set(range(n_node))
    # BFS traversal binary coloring -1 means no color
    vtx_color = [-1 for _ in range(n_node)]
    while len(vtx2reach) > 0:
        r = random.sample(list(vtx2reach), 1)[0]  # [r] --> r
        q = deque([r])
        while len(q) > 0:
            cur_node = q.popleft()
            vtx2reach.remove(cur_node)
            if vtx_color[cur_node] == -1:
                vtx_color[cur_node] = 0
            nbr = A[cur_node].nonzero(as_tuple=True)[0].cpu().numpy()
            for i in nbr:
                if vtx_color[i] == -1:
                    vtx_color[i] = 1 - vtx_color[cur_node]
                    q.append(i)
                elif vtx_color[i] == vtx_color[cur_node]:
                    if fix_flag:
                        A[cur_node, i] = 0
                    else:  # do not fix
                        flag = False
                        return flag, vtx_color, A
                else:  # cur_node and the nbr is colored with different colors
                    continue
    return flag, vtx_color, A


def dict2perm(cluster_dict):
    perm = np.concatenate(list(cluster_dict.values()))
    part = [0] + [len(it) for it in cluster_dict.values()]
    part = np.cumsum(part, dtype=np.int64)
    return perm, part


def graclus_coarsen(A: SparseTensor, level: int):
    row, col, wgt = A.coo()
    coarsen_cluster = []
    for i in range(level):
        cluster = graclus_cluster(row, col, wgt)
        _, cluster = cluster.unique(return_inverse=True)
        (row, col), wgt = pool_edge(cluster, torch.stack([row, col]), wgt)
        coarsen_cluster.append(cluster.cpu().numpy())
    return row, col, wgt, coarsen_cluster


def graclus_refine_raw(assignments, level: int = 1, verbose=False):
    assert level > 0
    max_level = len(assignments)
    coarest = assignments[-1]  # max_level-1
    base_cluster = {c: np.where(c == coarest)[0] for c in range(coarest.max() + 1)}
    for i in range(max_level - 2, level - 2, -1):
        cluster = assignments[i]
        for c in base_cluster:
            node = []
            for supernode in base_cluster[c]:
                node.append(np.where(supernode == cluster)[0])
            node = np.concatenate(node)
            base_cluster[c] = node
        if verbose:
            print("----->")
            print("[level: {}],  refined cluster:\n{}".format(level, base_cluster))
            print("-----<")
    return base_cluster
