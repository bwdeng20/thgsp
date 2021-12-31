import pytest

from thgsp.graphs.generators import rand_bipartite, rand_udg
from thgsp.graphs.is_bipartite import is_bipartite

from ..utils4t import devices


@pytest.mark.parametrize("device", devices)
def test_is_bipartite1(device):
    ts_spm = rand_udg(50, device=device)
    assert not is_bipartite(ts_spm)[0]


@pytest.mark.parametrize("device", devices)
def test_is_bipartite2(device):
    ts_spm = rand_bipartite(4, 6, device=device)
    assert is_bipartite(ts_spm)[0]
