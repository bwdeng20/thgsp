from thgsp.datasets import Toy


def test_minnesota():
    ds = Toy()
    assert ds.A.shape == (27, 27)
    assert ds.distances.shape == (27, 27)
    assert ds[0].n_node == 27
