import torch
import matplotlib.pyplot as plt
from thgsp.visual.plotting import draw_cn, draw_signal
from thgsp.graphs.generators import rand_udg, rand_dg
from ..utils4t import plot
N = 10
density = 0.3
G1 = rand_udg(N, density=density)
G2 = rand_dg(N, density=density)


class TestDrawColorNode:
    def test_draw_cn(self):
        fig, axes = plt.subplots(1, 2)
        draw_cn(G1, ax=axes[0])
        draw_cn(G2, ax=axes[1])
        plot()

    def test_draw_cn_pos(self):
        pos = torch.rand(N, 2)
        fig, axes = plt.subplots(1, 2)
        draw_cn(G1, pos, ax=axes[0])
        draw_cn(G2, pos, ax=axes[1])
        plot()

    def test_draw_cn_nc(self):
        fig, axes = plt.subplots(1, 2)
        draw_cn(G1, node_color=torch.rand(N), ax=axes[0])
        draw_cn(G2, node_color=torch.randn(N), ax=axes[1], with_labels=False)
        plot()


class TestDrawSignal:
    def test_draw_signal(self):
        fig, axes = plt.subplots(1, 2)
        draw_signal(G1, ax=axes[0])
        draw_signal(G2, ax=axes[1])
        plot()
