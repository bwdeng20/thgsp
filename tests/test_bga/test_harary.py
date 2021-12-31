from thgsp.bga._utils import is_bipartite_fix
from thgsp.bga.harary import harary
from thgsp.graphs.generators import rand_udg

from ..utils4t import remove_downloaded_dataset


def test_harary():
    for _ in range(30):
        N = 20
        G = rand_udg(N, 0.3)
        print("\nA: ", G.sum())
        try:
            bptG, beta, beta_dist, colors, _ = harary(G)
            for i in range(len(bptG)):
                assert is_bipartite_fix(bptG[i])
                print("bptG[{}]: {} ".format(i, bptG[i].sum()))
        except RuntimeError as err:
            raise err


def test_harary_minnesota():
    from thgsp.convert import SparseTensor
    from thgsp.datasets import Minnesota

    ds = Minnesota(download=True, connected=True)
    A = ds.A
    bptG, beta, beta_dist, vtx_color, _ = harary(SparseTensor.from_scipy(A))
    for i in range(len(bptG)):
        assert is_bipartite_fix(bptG[i])
    remove_downloaded_dataset("minnesota-usc")
