import torch
import pytest
import numpy as np
from thgsp.utils.sparse_utils import img2graph, absv, absv_, matrix_power, SparseTensor, get_ddd
from ..utils4t import plot, devices, float_dtypes


@pytest.mark.parametrize('threshold', [None, 50])
@pytest.mark.parametrize('shape', [(32, 32), (256, 256), (300, 246), (512, 512), [1024, 1024]])
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
        draw_cn(Graph(Ar), pos=xy, node_color=beta_r,
                font_size=5, node_size=10, with_labels=False)
        plt.subplot(122)
        draw_cn(Graph(Ad), pos=xy, node_color=beta_d,
                font_size=5, node_size=10, with_labels=False)
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
    if not torch.allclose(tar.cpu(), tar2):  # when numpy and torch differ in the matrix_power
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
