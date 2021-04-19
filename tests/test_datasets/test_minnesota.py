import pytest
from thgsp.datasets import Minnesota


def test_minnesota():
    with pytest.raises(RuntimeError):
        Minnesota()
    ds = Minnesota(connected=True, download=True)
    assert ds.F.shape == (ds.A.shape[0],)
    print(ds.F.max(), ds.F.min())
