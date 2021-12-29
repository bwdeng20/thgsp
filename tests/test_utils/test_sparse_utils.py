import torch
import pytest
import numpy as np
from thgsp.utils.sparse_utils import (
    img2graph,
    absv,
    absv_,
    matrix_power,
    SparseTensor,
    multivariate_normal,
    sparse_xcipy_logdet,
)
from ..utils4t import plot, devices, float_dtypes

try:
    from sksparse.cholmod import cholesky

    is_sksparse_installed = True
except ImportError:
    is_sksparse_installed = False


@pytest.mark.parametrize("threshold", [None, 50])
@pytest.mark.parametrize(
    "shape", [(32, 32), (256, 256), (300, 246), (512, 512), [1024, 1024]]
)
def test_img2graph(shape, threshold):
    H, W = shape
    N = H * W
    pseudo_img = torch.rand(3, H, W) * 255
    pseudo_img = pseudo_img.int()
    Ar, Ad, beta_r, beta_d, pixels, xy = img2graph(pseudo_img, threshold, True)
    assert Ar.sizes() == [N, N]
    assert xy.shape == (N, 2)
    if N < 256 * 256:
        from thgsp.visual.plotting import draw_cn
        from thgsp.graphs.core import Graph
        import matplotlib.pyplot as plt

        xy = xy * 10
        plt.subplot(121)
        draw_cn(
            Graph(Ar),
            pos=xy,
            node_color=beta_r,
            font_size=5,
            node_size=10,
            with_labels=False,
        )
        plt.subplot(122)
        draw_cn(
            Graph(Ad),
            pos=xy,
            node_color=beta_d,
            font_size=5,
            node_size=10,
            with_labels=False,
        )
        plot()


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("k", [0, 2, 5])
def test_matrix_power(device, dtype, k):
    n = 5
    a = torch.rand(n, n, dtype=dtype, device=device)
    out0 = matrix_power(a, k)
    tar = torch.matrix_power(a, k)
    tar2 = np.linalg.matrix_power(a.cpu().numpy(), k)
    tar2 = torch.from_numpy(tar2)
    out = matrix_power(SparseTensor.from_dense(a), k).to_dense()
    if not torch.allclose(
        tar.cpu(), tar2
    ):  # when numpy and torch differ in the matrix_power
        print("\n", tar2 - tar.cpu())
        print(out - tar)
        print(out.cpu() - tar2)
        print(out0.cpu() - tar2)
    assert torch.allclose(out.cpu(), tar2)
    #  the  results of dense torch.Tensor coincide with native pytorch
    assert torch.allclose(out0.cpu(), tar.cpu())


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", float_dtypes)
def test_matrix_abs_elementwise(device, dtype):
    n = 5
    a = torch.rand(n, n, dtype=dtype, device=device)

    spa = SparseTensor.from_dense(a)
    row, col, val = spa.coo()

    out = absv(spa)
    row0, col0, val0 = out.coo()
    assert id(row) == id(row0)
    assert id(col) == id(col0)
    assert id(val) != id(val0)

    assert torch.allclose(out.to_dense(), a.abs())
    c = absv_(spa)
    row1, col1, val1 = c.coo()
    assert id(row) == id(row1)
    assert id(col) == id(col1)
    assert id(val) == id(val1)
    #   assert torch.allclose(out, tar)  # use np.linalg.norm() as the target


@pytest.mark.skipif(
    is_sksparse_installed is False, reason="scikit-sparse is not installed"
)
@pytest.mark.parametrize("num", [1, 5000000])
def test_multivariate_normal(num):
    import scipy
    from scipy.sparse.linalg import inv

    n = 4
    tmp = scipy.sparse.rand(n, n, 0.3, format="lil")
    tmp.setdiag(1)
    Cov = tmp @ tmp.T
    Cov = Cov.tocsc()
    Pre = inv(Cov).tocsc()
    z1, z2 = multivariate_normal(
        mean=1, cov=Cov, precision=Pre, num=num, return_th=False
    )
    z0 = np.random.multivariate_normal(np.ones(n), Cov.A, num)
    m1, v1 = np.mean(z1, 1), np.var(z1, 1)
    m2, v2 = np.mean(z2, 1), np.var(z2, 1)
    m0, v0 = np.mean(z0, 0), np.var(z0, 0)
    print("\n===================================")
    print(m1, v1)
    print(m2, v2)
    print(m0, v0)
    print("===================================")

    if num > 1e6:
        assert np.abs(m1 - m0).mean() < 1e-2
        assert np.abs(m2 - m0).mean() < 1e-2
        assert np.abs(v1 - v0).mean() < 1e-2
        assert np.abs(v2 - v0).mean() < 1e-2

    x1, x2 = multivariate_normal(
        mean=1, cov=Cov, precision=Pre, num=num, delta=1e-5, return_th=True
    )
    assert x1.shape == x2.shape
