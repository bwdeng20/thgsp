from thgsp.bga.greedy import greedy_bga, is_bipartite_fix
from thgsp.graphs.generators import random_graph, np


def test_greedy_bga():
    N = 10
    num_iter = 10
    g = random_graph(N, 0.4)
    flag, vtx_color, A = is_bipartite_fix(g.to_dense(), fix_flag=False)
    assert not flag

    B, bset1 = greedy_bga(g.to_scipy("coo"), num_iter, verbose=True)
    flag1, vtx_color1, Ab = is_bipartite_fix(B, fix_flag=False)
    assert flag1
    assert (Ab - B).sum() == 0

    # no change
    C, bset2 = greedy_bga(B, num_iter, verbose=False)
    print(bset1)
    print(vtx_color1)
    print(bset2)
    assert (C - B).toarray().sum() == 0

    # since BFS-bipartite coloring root is chosen at random
    assert (np.allclose(bset2, vtx_color1) or np.allclose(bset2, 1 - np.asarray(vtx_color1)))
