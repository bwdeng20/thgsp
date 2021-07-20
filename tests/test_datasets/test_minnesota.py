import pytest
from thgsp.datasets import Minnesota
from ..utils4t import remove_downloaded_dataset


def test_minnesota():
    with pytest.raises(RuntimeError):
        Minnesota()
    ds = Minnesota(connected=True, download=True)
    assert ds.F.shape == (ds.A.shape[0],)
    print(ds.F.max(), ds.F.min())
    remove_downloaded_dataset("minnesota-usc")
