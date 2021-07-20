import torch
from torch_sparse import SparseTensor


def laplace(adj: SparseTensor, lap_type=None):
    M, N = adj.sizes()
    assert M == N
    row, col, val = adj.clone().coo()
    val = col.new_ones(col.shape, dtype=adj.dtype()) if val is None else val
    deg = adj.sum(0)

    loop_index = torch.arange(N, device=adj.device()).unsqueeze_(0)
    if lap_type in (None, 'sym'):
        deg05 = deg.pow(-0.5)
        deg05[deg05 == float('inf')] = 0
        wgt = deg05[row] * val * deg05[col]
        wgt = torch.cat([-wgt.unsqueeze_(0), val.new_ones(1, N)], 1).squeeze_()

    elif lap_type == "rw":
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float("inf")] = 0
        wgt = deg_inv[row] * val

        wgt = torch.cat([-wgt.unsqueeze_(0), val.new_ones(1, N)], 1).squeeze_()

    elif lap_type == "comb":
        wgt = torch.cat([-val.unsqueeze_(0), deg.unsqueeze_(0)], 1).squeeze_()

    else:
        raise TypeError("Invalid laplace type: {}".format(lap_type))

    row = torch.cat([row.unsqueeze_(0), loop_index], 1).squeeze_()
    col = torch.cat([col.unsqueeze_(0), loop_index], 1).squeeze_()
    lap = SparseTensor(row=row, col=col, value=wgt, sparse_sizes=(M, N))
    return lap
