import networkx as nx
import numpy as np
import torch
from torch_cluster import knn_graph, radius_graph
from torch_sparse import SparseTensor

from .core import Graph, DiGraph


def rand_udg(N, density=0.1, dtype=None, device=None):
    A = torch.rand(N, N, dtype=dtype, device=device)
    A.fill_diagonal_(0)
    A = A + A.t()
    member, _ = A.view(-1).sort(descending=True)
    thresh = member[int(N * (N - 1) * density)]
    A[A < thresh] = 0
    return Graph(A)


def rand_dg(N, density=0.1, dtype=None, device=None):
    A = torch.rand(N, N, dtype=dtype, device=device)
    A.fill_diagonal_(0)
    member, _ = A.view(-1).sort(descending=True)
    thresh = member[int(N * (N - 1) * density)]
    A[A < thresh] = 0
    return DiGraph(A)


def rand_bipartite(N1, N2, p=0.2, dtype=None, device=None, return_partition=False):
    G = nx.bipartite.random_graph(N1, N2, p)
    csr_sci_adj = nx.adj_matrix(G)
    A = SparseTensor.from_scipy(csr_sci_adj).to(device, dtype)
    bpg = Graph(A)
    if return_partition:
        beta = [1 if d['bipartite'] == 1 else 0 for n, d in G.nodes(data=True)]
        beta = torch.as_tensor(beta, dtype=torch.bool)
        return bpg, beta
    return bpg


def random_bgraph(N1, N2, p=0.2, dtype=None, device=None, seed=None) -> Graph:
    r"""
    Generate a random bipartite graph whose tow bipartite sets have :obj:`N1` and :obj:`N2` nodes with specific data
    type and device.

    Parameters
    ----------
    N1: int
        The number of nodes in the first bipartite set.
    N2: int
        The number of nodes in the second bipartite set.
    p:  float, optional
        Probability for edge creation
    dtype: torch.dtype, optional
    device: torch.device, optional
    seed: TODO decided by u
        Control the random behaviour

    """
    sparse_adj = ...
    return Graph(sparse_adj)


def random_graph(N, density=0.01, directed=False, dtype=None, device=None, weighted=True, seed=None):
    """Generate a sparse matrix of the given shape and density with randomly
    distributed values.
    Parameters
    ----------
    N : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    directed: bool, optional
        If True, construct a directed graph.
    dtype : torch.dtype, optional
        data type of the returned matrix.
    device: torch.device, optional
        device of the returned matrix.
    weighted:bool
        If True, generate weighted graph.
    seed : int optional
        The random seed

    Returns
    -------
    Graph or DiGraph

    """
    if density < 0 or density > 1:
        raise ValueError("density expected to be from [0,1]")
    random_state = np.random.RandomState(seed)

    if not directed:
        n_all_edge = int((N - 1) * N / 2)
        k = int(density * n_all_edge)
        if k < 1:
            raise RuntimeError(
                f"Density {density} is too small to generate 1 edge of {N}-node undirected graph.")
        ind = random_state.choice(n_all_edge, size=k, replace=False)
        data = random_state.rand(k) if weighted else None
        i, j = tri2square(ind)
        # construct the symmetric upper triangular part
        row = np.concatenate([i, j])
        col = np.concatenate([j, i])
        data = None if data is None else np.tile(data, 2)

        row = torch.as_tensor(row)
        col = torch.as_tensor(col)
        value = None if data is None else torch.as_tensor(data).to(dtype)

    else:
        n_all_edge = N * (N - 1)
        k = int(density * n_all_edge)
        if k < 1:
            raise RuntimeError(
                f"Density {density} is too small to generate 1 edge of {N}-node directed graph.")
        ind = random_state.choice(n_all_edge, size=k, replace=False)
        data = random_state.rand(k) if weighted else None
        i, j = flat2squre(ind, N)
        row = torch.as_tensor(i)
        col = torch.as_tensor(j)
        value = None if data is None else torch.as_tensor(data).to(dtype)

    coo_adj = SparseTensor(row=row, col=col, value=value,
                           sparse_sizes=(N, N)).to(device)
    return DiGraph(coo_adj) if directed else Graph(coo_adj)


def tri2square(tri_idx):
    tri_idx = np.asarray(tri_idx).astype(np.float64, copy=False)
    up_bound = (1 + np.sqrt(1 + 8 * tri_idx)) / 2
    i = np.floor(up_bound)
    j = tri_idx - (i - 1) * i / 2
    return i.astype(np.int64, copy=False), j.astype(np.int64, copy=False)


def flat2squre(flat_idx, N):
    N1 = N - 1
    idx = np.asarray(flat_idx).astype(np.int64, copy=False)
    i = idx // N1
    j = idx - N1 * i
    j = j + (j >= i)
    return i, j


def knn(x, k=2, loop=False, dtype=None, device=None):
    N, D = x.shape
    batch = torch.zeros(N, dtype=torch.long)
    edge_index = knn_graph(x, k, batch=batch, loop=loop).to(device)
    edge_val = torch.ones(edge_index.shape[-1], dtype=dtype, device=device)
    return SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_val, sparse_sizes=(N, N))


def radius(x, r=0.5, loop=False, dtype=None, device=None):
    N, D = x.shape
    batch = torch.zeros(N, dtype=torch.long)
    edge_index = radius_graph(x, r, batch=batch, loop=loop).to(device)
    edge_val = torch.ones(edge_index.shape[-1], dtype=dtype, device=device)
    return SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_val, sparse_sizes=(N, N))
