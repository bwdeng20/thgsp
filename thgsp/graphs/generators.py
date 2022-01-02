import networkx as nx
import numba as nb
import numpy as np
import torch
from torch_cluster import knn_graph, radius_graph
from torch_sparse import SparseTensor

from .core import DiGraph, Graph


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
    csr_sci_adj = nx.adjacency_matrix(G)
    A = SparseTensor.from_scipy(csr_sci_adj).to(device, dtype)
    bpg = Graph(A)
    if return_partition:
        beta = [1 if d["bipartite"] == 1 else 0 for n, d in G.nodes(data=True)]
        beta = torch.as_tensor(beta, dtype=torch.bool)
        return bpg, beta
    return bpg


def random_bgraph(N1, N2, p=0.2, dtype=None, device=None, seed=None) -> Graph:
    r"""
    Generate a random bipartite graph whose tow bipartite sets have :obj:`N1` and
    :obj:`N2` nodes with specific data
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


def random_graph(
    N, density=0.01, directed=False, dtype=None, device=None, weighted=True, seed=None
):
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
    rng = np.random.default_rng(seed=seed)

    if not directed:
        n_all_edge = int((N - 1) * N / 2)
        k = int(density * n_all_edge)
        if k < 1:
            raise RuntimeError(
                f"Density {density} is too small to generate 1 edge "
                f"of {N}-node undirected graph."
            )
        ind = rng.choice(n_all_edge, size=k, replace=False)
        data = rng.random(k) if weighted else None
        i, j = tri2square(ind)
        # construct the symmetric upper triangular part
        row = np.concatenate([i, j])
        col = np.concatenate([j, i])
        data = None if data is None else np.tile(data, 2)

        row = torch.as_tensor(row, device=device, dtype=torch.long)
        col = torch.as_tensor(col, device=device, dtype=torch.long)
        value = (
            None if data is None else torch.as_tensor(data, dtype=dtype, device=device)
        )

    else:
        n_all_edge = N * (N - 1)
        k = int(density * n_all_edge)
        if k < 1:
            raise RuntimeError(
                f"Density {density} is too small to "
                f"generate 1 edge of {N}-node directed graph."
            )
        ind = rng.choice(n_all_edge, size=k, replace=False)
        data = rng.random(k) if weighted else None
        i, j = flat2squre(ind, N)
        row = torch.as_tensor(i, device=device, dtype=torch.long)
        col = torch.as_tensor(j, device=device, dtype=torch.long)
        value = (
            None if data is None else torch.as_tensor(data, dtype=dtype, device=device)
        )

    coo_adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))
    return DiGraph(coo_adj) if directed else Graph(coo_adj)


@nb.jit("UniTuple(f8[:],2)(i8[:])", nopython=True)
def tri2square(tri_idx):
    up_bound = (1 + np.sqrt(1 + 8 * tri_idx)) / 2
    i = np.floor(up_bound)
    j = tri_idx - (i - 1) * i / 2
    return i, j


@nb.jit("UniTuple(i8[:],2)(i8[:],i8)", nopython=True)
def flat2squre(flat_idx, N):
    N1 = N - 1
    i = flat_idx // N1
    j = flat_idx - N1 * i
    j = j + (j >= i)
    return i, j


def knn(x, k=2, loop=False, dtype=None, device=None):
    N, D = x.shape
    batch = torch.zeros(N, dtype=torch.long)
    edge_index = knn_graph(x, k, batch=batch, loop=loop).to(device)
    edge_val = torch.ones(edge_index.shape[-1], dtype=dtype, device=device)
    return SparseTensor(
        row=edge_index[0], col=edge_index[1], value=edge_val, sparse_sizes=(N, N)
    )


def radius(x, r=0.5, loop=False, dtype=None, device=None):
    N, D = x.shape
    batch = torch.zeros(N, dtype=torch.long)
    edge_index = radius_graph(x, r, batch=batch, loop=loop).to(device)
    edge_val = torch.ones(edge_index.shape[-1], dtype=dtype, device=device)
    return SparseTensor(
        row=edge_index[0], col=edge_index[1], value=edge_val, sparse_sizes=(N, N)
    )
