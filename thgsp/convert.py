import warnings

import numpy as np
import torch
from scipy.sparse import spmatrix, coo_matrix, csr_matrix
from torch_sparse import SparseTensor
from torch.utils.dlpack import to_dlpack, from_dlpack


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


def to_scipy(mat, layout="csr"):
    if isinstance(mat, torch.Tensor):
        if not mat.is_sparse:
            smt = SparseTensor.from_dense(mat).to_scipy(layout)
        else:
            smt = SparseTensor.from_torch_sparse_coo_tensor(mat).to_scipy(layout)
    elif isinstance(mat, SparseTensor):
        smt = mat.to_scipy(layout)
    elif isinstance(mat, (spmatrix, np.ndarray)):
        if layout == "csr":
            smt = csr_matrix(mat)
        elif layout == "coo":
            smt = coo_matrix(mat)
        else:
            raise TypeError("not supported layout")
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


def to_cpx(mat):
    assert mat.is_cuda()
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    if isinstance(mat, SparseTensor):
        rowptr, col, wgt = mat.csr()
    elif isinstance(mat, torch.Tensor):
        if mat.is_sparse:
            rowptr, col, wgt = SparseTensor.from_torch_sparse_coo_tensor(mat).csr()
        else:
            rowptr, col, wgt = SparseTensor.from_dense(mat).csr()
    else:
        raise TypeError(f"{type(mat)} is not supported")

    m, n = mat.sizes()
    if wgt is not None:
        cp_wgt = cp.fromDlpack(to_dlpack(wgt))
    else:
        cp_wgt = cp.ones(col.shape[-1], dtype=cp.float64)
    cp_rowptr = cp.fromDlpack(to_dlpack(rowptr))
    cp_col = cp.fromDlpack(to_dlpack(col))
    return csr_matrix((cp_wgt, cp_col, cp_rowptr), shape=(m, n))


def from_cpx(mat):
    wgt = from_dlpack(mat.data.toDlpack())
    rowptr = from_dlpack(mat.indptr.toDlpack()).to(torch.long)
    col = from_dlpack(mat.indices.toDlpack()).to(torch.long)
    return SparseTensor(rowptr=rowptr, col=col, value=wgt, sparse_sizes=mat.shape, is_sorted=True)


def get_array_module(on_gpu):
    if on_gpu:
        try:
            import cupy as xp
            import cupyx.scipy as xscipy
        except ImportError:
            warnings.warn("CuPy is not installed, use numpy and scipy instead")
            import numpy as xp
            import scipy as xscipy
    else:
        import numpy as xp
        import scipy as xscipy
    return xp, xscipy
