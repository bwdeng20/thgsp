import torch
from torch_sparse import SparseTensor


def in_degree(adj: SparseTensor, bunch=None):
    if bunch is None:
        in_deg = adj.sum(0)
    else:
        N = adj.size(0)
        if len(bunch) > int(0.2 * N):
            in_deg = adj.sum(0)[bunch]
        else:
            ptr, idx, val = adj.csc()
            in_deg = val.new_zeros(len(bunch))
            for i, v in enumerate(bunch):
                in_deg[i] = val[ptr[v] : ptr[v + 1]].sum()
    return in_deg


def out_degree(adj: SparseTensor, bunch=None):
    if bunch is None:
        out_deg = adj.sum(1)
    else:
        N = adj.size(0)
        if len(bunch) > int(0.2 * N):
            out_deg = adj.sum(1)[bunch]
        else:
            ptr, idx, val = adj.csr()
            out_deg = val.new_zeros(len(bunch))
            for i, v in enumerate(bunch):
                out_deg[i] = val[ptr[v] : ptr[v + 1]].sum()
    return out_deg


def binary_out_degree(adj: SparseTensor, bunch=None):
    ptr, _, _ = adj.csr()
    ptr = ptr.cpu().numpy()
    deg = ptr[1:] - ptr[:-1]
    if bunch is not None:
        deg = deg[bunch]
    return deg


def degree_matrix(adj: SparseTensor, indeg=True):
    N = adj.size(-1)
    deg = adj.sum(0) if indeg else adj.sum(1)
    row = col = torch.arange(N, device=adj.device())
    degs = torch.as_tensor(deg, device=adj.device())
    return SparseTensor(
        row=row, col=col, value=degs, sparse_sizes=(N, N), is_sorted=True
    )
