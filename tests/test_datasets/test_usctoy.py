import pytest
from thgsp.datasets import Toy


def test_toy():
    with pytest.raises(RuntimeError):
        Toy()
    ds = Toy(download=True)
    assert ds.A.shape == (27, 27)
    assert ds.distances.shape == (27, 27)
    assert ds[0].n_node == 27
