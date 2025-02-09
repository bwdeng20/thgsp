
from thgsp.datasets import Toy

from ..utils4t import remove_downloaded_dataset


def test_toy():
    ds = Toy(download=True)
    assert ds.A.shape == (27, 27)
    assert ds.distances.shape == (27, 27)
    assert ds[0].n_node == 27
    remove_downloaded_dataset("GraphStructures-master")
