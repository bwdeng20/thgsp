import torch
import pytest

from thgsp.utils.sparse_utils import img2graph
from ..utils4t import plot


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
