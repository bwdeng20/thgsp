import pytest
import numpy as np

from thgsp.filters.approximation import cheby_coeff, cheby_op, SparseTensor, nla, torch, cheby_op_basis
from thgsp.filters.kernels import meyer_kernel, ideal_kernel, meyer_mirror_kernel
from ..utils4t import devices, float_dtypes

krns = np.array([[[ideal_kernel, meyer_kernel],
                  [meyer_kernel, ideal_kernel],
                  [meyer_kernel, ideal_kernel]]])  # 1(n_graph) x 3(Cout) x 2(Cin) kernel


@pytest.mark.parametrize('dtype', float_dtypes)
@pytest.mark.parametrize('device', devices)
class TestCheby:
    def test_cheby_coefficient(self, device, dtype):
        K = 5
        coeff = cheby_coeff(ideal_kernel, K=K, dtype=dtype, device=device)
        assert coeff.shape == (1, 1, 1, K + 1)

        M, Co, Ci = krns.shape
        c = cheby_coeff(krns, K, device=device, dtype=dtype)
        assert c.shape == (M, Co, Ci, K + 1)

        with pytest.raises(AssertionError):
            cheby_coeff(krns, lam_max=-0.5)

    def test_cheby_op(self, device, dtype):
        K = 3
        c = cheby_coeff(krns, K=K, device=device, dtype=dtype)
        M, Co, Ci = krns.shape

        N = 500
        L = SparseTensor.eye(N).to(dtype).to(device)
        x = c.new_empty(N).random_()
        r = cheby_op(x, L, c[0])
        assert r.shape == (Co, N, Ci)


@pytest.mark.parametrize('dtype', float_dtypes)
@pytest.mark.parametrize('device', devices)
class TestNla:
    @pytest.mark.parametrize('frac', [0.2, 1])
    def test_frac(self, dtype, device, frac):
        Co, N, Ci = 4, 11, 1
        x = torch.rand(Co, N, Ci, dtype=dtype, device=device)
        y = nla(x, frac=frac)
        num_non_zero = y.nonzero().shape[0]
        assert num_non_zero - Ci * int(frac * N) == 0

    @pytest.mark.parametrize('k', [2, 10])
    def test_k(self, dtype, device, k):
        Co, N, Ci = 4, 11, 1
        x = torch.rand(Co, N, Ci, dtype=dtype, device=device)
        y = nla(x, k=k)
        num_non_zero = y.nonzero().shape[0]
        assert num_non_zero - Ci * k == 0


@pytest.mark.parametrize('dtype', float_dtypes)
def test_cheby_op_basis(dtype):
    N = 8
    A = torch.rand(N, N, dtype=dtype)
    A = A + A.t()
    A.fill_diagonal_(0)

    Ats = SparseTensor.from_dense(A)
    Asc = Ats.to_scipy('csr')
    x = torch.ones(N, 1, dtype=dtype)
    xnp = x.numpy()

    from thgsp.bga.utils import laplace as sci_laplace
    from thgsp.graphs.laplace import laplace as ts_laplace
    Lts = ts_laplace(Ats, 'sym')
    Lsc = sci_laplace(Asc, 'sym').tocsr()

    krn = np.array([[[meyer_kernel],
                     [meyer_mirror_kernel]]])  # 1(n_graph) x 2(Cout) x 1(Cin) kernel
    coeff = cheby_coeff(krn, K=6)
    yts = cheby_op(x, Lts, coeff[0])
    yts0, yts1 = yts[..., 0].numpy()

    H0, H1 = cheby_op_basis(Lsc, coeff[0].squeeze_())

    y0 = H0 @ xnp
    y1 = H1 @ xnp
    err = 1e-5 if dtype is float_dtypes[0] else 1e-12
    assert np.abs(yts0 - y0.ravel()).sum() < err
    assert np.abs(yts1 - y1.ravel()).sum() < err
