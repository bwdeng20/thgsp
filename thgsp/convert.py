import warnings

import numpy as np
import torch
from scipy.sparse import spmatrix, coo_matrix, csr_matrix, csc_matrix
from torch_sparse import SparseTensor
from torch.utils.dlpack import from_dlpack

SparseLayouts = ("csc", "coo", "csr")
numpy_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
    torch.bool: np.bool,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


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
        raise TypeError(f"{type(mat)} not supported now")

    return stm


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


def to_cp(mat):
    import cupy as cp

    if isinstance(mat, torch.Tensor):
        if mat.is_sparse:
            mat = mat.to_dense()
        dense = cp.asarray(mat)
    elif isinstance(mat, (np.ndarray, list)):
        dense = cp.asarray(mat)
    elif isinstance(mat, SparseTensor):
        dense = cp.asarray(mat.to_dense())
    else:
        raise TypeError(f"{type(mat)} is not supported now or invalid")
    return dense


def to_xp(mat: torch.Tensor):
    device = mat.device
    if device.type != "cpu":
        return to_cp(mat)
    return to_np(mat)


def to_scipy(mat, layout="csr", dtype=None):
    assert layout in SparseLayouts
    if isinstance(mat, torch.Tensor):
        if not mat.is_sparse:
            smt = SparseTensor.from_dense(mat).to_scipy(layout, dtype)
        else:
            smt = SparseTensor.from_torch_sparse_coo_tensor(mat).to_scipy(layout, dtype)
    elif isinstance(mat, SparseTensor):
        smt = mat.to_scipy(layout, dtype)
    elif isinstance(mat, (spmatrix, np.ndarray)):
        cls = {"csr": csr_matrix, "csc": csc_matrix, "coo": coo_matrix}[layout]
        smt = cls(mat)
    else:
        raise TypeError(f"{type(mat)} is not supported now or invalid")

    return smt


def to_xcipy(mat, layout: str = "csr", dtype=None):
    if isinstance(mat, SparseTensor):
        device = mat.device()
        if device.type != "cpu":
            return to_cpx(mat, layout, dtype)
        return to_scipy(mat, layout, dtype)
    elif isinstance(mat, spmatrix):
        return to_scipy(mat, layout, dtype)
    else:
        raise TypeError(f"{type(mat)} is not supported now or invalid")


def to_cpx(mat, layout="csr", dtype=None):
    assert layout in SparseLayouts
    import cupy as cp
    import cupyx.scipy as xcipy

    if isinstance(mat, torch.Tensor):
        assert mat.dim() == 2
        assert mat.is_cuda
        if mat.is_sparse:
            smt = SparseTensor.from_torch_sparse_coo_tensor(mat)
        else:
            smt = SparseTensor.from_dense(mat)

    elif isinstance(mat, SparseTensor):
        assert mat.dim() == 2
        assert mat.is_cuda()
        smt = mat

    elif isinstance(mat, xcipy.sparse.spmatrix):
        assert mat.ndim == 2
        cls = {
            "csr": xcipy.sparse.csr_matrix,
            "csc": xcipy.sparse.csc_matrix,
            "coo": xcipy.sparse.coo_matrix,
        }[layout]
        smt = cls(mat)
        return smt

    else:
        raise RuntimeError

    shape = smt.sparse_sizes()

    if layout == "coo":
        row, col, value = smt.coo()
        row = cp.asarray(row.detach())
        col = cp.asarray(col.detach())
        value = (
            cp.asarray(value.detach())
            if smt.has_value()
            else cp.ones(smt.nnz(), dtype=dtype)
        )
        return xcipy.sparse.coo_matrix((value, (row, col)), shape)
    elif layout == "csr":
        rowptr, col, value = smt.csr()
        rowptr = cp.asarray(rowptr.detach())
        col = cp.asarray(col.detach())
        value = (
            cp.asarray(value.detach())
            if smt.has_value()
            else cp.ones(smt.nnz(), dtype=dtype)
        )
        return xcipy.sparse.csr_matrix((value, col, rowptr), shape)
    elif layout == "csc":
        colptr, row, value = smt.csc()
        colptr = cp.asarray(colptr.detach())
        row = cp.asarray(row.detach())
        value = (
            cp.asarray(value.detach())
            if smt.has_value()
            else cp.ones(smt.nnz(), dtype=dtype)
        )
        return xcipy.sparse.csc_matrix((value, row, colptr), shape)

    else:
        raise RuntimeError(
            f"{layout} is not one of valid sparse formats `coo`, `csr` and `csc`."
        )


def from_cpx(mat):
    wgt = from_dlpack(mat.data.toDlpack())
    rowptr = from_dlpack(mat.indptr.toDlpack()).to(torch.long)
    col = from_dlpack(mat.indices.toDlpack()).to(torch.long)
    return SparseTensor(
        rowptr=rowptr, col=col, value=wgt, sparse_sizes=mat.shape, is_sorted=True
    )


def get_array_module(on_gpu):
    if on_gpu:
        try:
            import cupy as xp
            import cupyx.scipy as xcipy
            import cupyx.scipy.sparse.linalg as xsplin
        except ImportError:
            warnings.warn("CuPy is not installed, use numpy and scipy instead")
            import numpy as xp
            import scipy as xcipy
            import scipy.sparse.linalg as xsplin
    else:
        import numpy as xp
        import scipy as xcipy
        import scipy.sparse.linalg as xsplin
    return xp, xcipy, xsplin


def get_ddd(A):
    if isinstance(A, SparseTensor):
        density = A.density()
        dt = A.dtype()
        dv = A.device()
        on_gpu = A.is_cuda()
    elif isinstance(A, torch.Tensor):
        num = torch.prod(torch.as_tensor(A.shape))
        nnz = A._nnz() if A.is_sparse else A.count_nonzero()
        density = (nnz / num).item()
        dt = A.dtype
        dv = A.device
        on_gpu = A.is_cuda
    else:
        raise TypeError(f"Type {type(A)} is not supported")
    return dt, dv, density, on_gpu
