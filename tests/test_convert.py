import torch
import pytest
import scipy
import numpy as np
from thgsp.convert import to_np, to_scipy, to_torch_sparse, to_cpx, from_cpx, to_xcipy
from thgsp.convert import spmatrix, SparseTensor, coo_matrix, get_ddd, get_array_module
from .utils4t import sparse_formats, float_dtypes, devices


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


@pytest.mark.parametrize("layout", sparse_formats)
@pytest.mark.parametrize("dt", float_dtypes)
def test_cp(layout, dt):
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("cupy is not installed, skip the test")
    Ad = torch.rand(3, 3, dtype=dt)

    to_cpx(Ad.cuda(), layout, dt)

    A = SparseTensor.from_dense(Ad)
    with pytest.raises(AssertionError):
        to_cpx(A, layout)
    Acpx = to_cpx(A.cuda(), layout=layout, dtype=dt)
    Acpx2 = to_cpx(Acpx, layout, dt)

    assert Acpx.format == layout
    assert Acpx2.format == layout

    B = A.cuda()
    Br = from_cpx(to_cpx(B))
    ptr, col, wgt = B.csr()
    ptr1, col1, wgt1 = Br.csr()
    assert (ptr - ptr1).sum() == 0
    assert (wgt - wgt1).sum() == 0


@pytest.mark.parametrize("on_gpu", [True, False])
def test_get_array_module(on_gpu):
    xp, xcipy, _ = get_array_module(on_gpu=on_gpu)
    if on_gpu:
        try:
            import cupy as cp
            import cupyx.scipy as xscipy
            assert xp == cp
            assert xscipy == xscipy
        except ImportError:
            pytest.skip("CuPy is not installed, use numpy and scipy instead")

    else:
        assert xp == np
        assert xcipy == scipy


def test_get_ddd():
    a = torch.rand(3, 3)
    spa = a.to_sparse()
    spa2 = SparseTensor.from_dense(a)
    if torch.cuda.is_available():
        acu = a.cuda()
        spacu = spa.cuda()
        spa2cu = spa2.cuda()
        print(get_ddd(acu))
        print(get_ddd(spacu))
        print(get_ddd(spa2cu))
    print(get_ddd(a))
    print(get_ddd(spa))
    print(get_ddd(spa2))


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("layout", sparse_formats)
def test_to_xcipy(device, layout):
    spa = SparseTensor.from_dense(torch.rand(3, 3, device=device))
    xcia = to_xcipy(spa, layout=layout)
    assert xcia.shape == (3, 3)
    assert xcia.format == layout
