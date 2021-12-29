import networkx as nx
import matplotlib.pyplot as plt
from thgsp.alg.traverse import bfs_lil
from ..utils4t import plot


def set_color(G, serial, r=1):
    node = list(G.nodes)
    length = len(node)
    idx = node.index(serial)
    nc = [10] * length
    nc[idx] = r + 1
    return nc


def test_lil_bfs():
    n = 10
    r = 1
    g1 = nx.random_regular_graph(d=3, n=10)
    g1_lil = nx.adj_matrix(g1, nodelist=range(n)).tolil()

    tree_gt = nx.bfs_tree(g1, source=r)
    tree = bfs_lil(g1_lil, r=r)

    pos = nx.spring_layout(g1)
    plt.subplot(131)
    plt.title("graph")
    nx.draw(g1, pos=pos, node_color=set_color(g1, r, 1), with_labels=True)

    plt.subplot(132)
    plt.title("bfs_nx_tree")
    nx.draw(tree_gt, pos=pos, node_color=set_color(tree_gt, r, 1), with_labels=True)

    plt.subplot(133)
    plt.title("bfs_my_tree")
    nx.draw(
        nx.from_dict_of_lists(tree, create_using=nx.DiGraph),
        pos=pos,
        node_color=set_color(tree_gt, r, 1),
        with_labels=True,
    )

    plot()
