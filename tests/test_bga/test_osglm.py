from thgsp.bga.osglm import osglm
from thgsp.bga.utils import is_bipartite_fix
from thgsp.graphs.generators import rand_udg


def test_osglm():
    N = 40
    G = rand_udg(N, 0.3)

    bptG, beta, append_nodes, vtx_color = osglm(G)
    assert is_bipartite_fix(bptG[0])

    try:
        from torgsp.sampling import osglm as dense_osglm
        from torgsp.types import ColorG
        import torch as th
        dense_bptG, dense_beta, dense_append_nodes, _ = dense_osglm(G.to_dense(),
                                                                    colorg=ColorG(vtx_color, max(vtx_color) + 1))

        print("      beta: ", beta[:, -1])
        print("dense beta: ", dense_beta.numpy()[:, -1])

        assert (th.as_tensor(bptG[0].toarray()) -
                dense_bptG[0]).abs().sum() == 0

    except ImportError as err:
        print(err)
