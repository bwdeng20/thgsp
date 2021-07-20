import torch
import pytest
import numpy as np
from thgsp.convert import to_np, to_scipy, to_torch_sparse, to_cpx, from_cpx
from thgsp.convert import spmatrix, SparseTensor, coo_matrix


@pytest.fixture(scope='module')
def mats():
    N = 4
    th_mat = torch.rand(N, N, dtype=torch.float)
    th_mat_gpu = th_mat.cuda() if torch.cuda.is_available() else th_mat
    ths_mat = th_mat.to_sparse()
    ths_mat_gpu = th_mat_gpu.to_sparse()
    np_mat = th_mat.numpy()
    sci_mat = coo_matrix(np_mat)
    sp_mat = SparseTensor.from_dense(th_mat)
    sp_mat_gpu = SparseTensor.from_dense(th_mat_gpu)
    bug_mat = "invalid matrix"
    return ths_mat, ths_mat_gpu, ths_mat, ths_mat_gpu, np_mat, sci_mat, sp_mat, sp_mat_gpu, bug_mat


def test_to_np(mats):
    for mat in mats[:-1]:
        tmp = to_np(mat)
        assert isinstance(tmp, np.ndarray)

    with pytest.raises(TypeError):
        to_np(mats[-1])


def test_to_torch_sparse(mats):
    for mat in mats[:-1]:
        tmp = to_torch_sparse(mat)
        assert isinstance(tmp, SparseTensor)

    with pytest.raises(TypeError):
        to_torch_sparse(mats[-1])


def test_to_scipy(mats):
    for mat in mats[:-1]:
        tmp = to_scipy(mat)
        assert isinstance(tmp, spmatrix)

    with pytest.raises(TypeError):
        to_scipy(mats[-1])


def test_cp():
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("cupy is not installed, skip the test")
    A = SparseTensor.from_dense(torch.rand(3, 3))
    with pytest.raises(AssertionError):
        to_cpx(A)
    to_cpx(A.cuda())

    B = A.cuda()
    Br = from_cpx(to_cpx(B))
    ptr, col, wgt = B.csr()
    ptr1, col1, wgt1 = Br.csr()
    assert (ptr - ptr1).sum() == 0
    assert (wgt - wgt1).sum() == 0