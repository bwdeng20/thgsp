import numpy as np
import torch
from scipy.sparse import spmatrix
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
