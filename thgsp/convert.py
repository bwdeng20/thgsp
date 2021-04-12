import numpy as np
import torch
from scipy.sparse import spmatrix, coo_matrix
from torch_sparse import SparseTensor


def to_torch_sparse(mat):
    if isinstance(mat, torch.Tensor):
        if not mat.is_sparse:
            stm = SparseTensor.from_dense(mat)
        else:
            stm = SparseTensor.from_torch_sparse_coo_tensor(mat)

    elif isinstance(mat, np.ndarray):
        stm = SparseTensor.from_dense(torch.as_tensor(mat))

    elif isinstance(mat, spmatrix):
        stm = SparseTensor.from_scipy(mat)

    elif isinstance(mat, SparseTensor):
        stm = mat
    else:
        raise TypeError("{} not supported now".format(type(mat)))

    return stm


def to_scipy(mat):
    if isinstance(mat, torch.Tensor):
        if not mat.is_sparse:
            mat = mat.to_sparse()
        mat = mat.coalesce()
        row, col = mat._indices()
        value = mat.values()
        smt = coo_matrix((value.cpu(), (row.cpu(), col.cpu())), shape=mat.shape)
    elif isinstance(mat, SparseTensor):
        smt = mat.to_scipy("coo")
    elif isinstance(mat, spmatrix):
        smt = mat
    elif isinstance(mat, np.ndarray):
        smt = coo_matrix(mat)
    else:
        raise TypeError(f"{type(mat)} is not supported now or invalid")

    return smt


def to_np(mat):
    if isinstance(mat, np.ndarray):
        dense = mat
    elif isinstance(mat, torch.Tensor):
        if mat.is_sparse:
            mat = mat.to_dense()
        dense = mat.cpu().numpy()
    elif isinstance(mat, SparseTensor):
        dense = mat.to_dense().cpu().numpy()

    elif isinstance(mat, spmatrix):
        dense = mat.toarray()
    else:
        raise TypeError(f"{type(mat)} is not supported now or invalid")
    return dense
