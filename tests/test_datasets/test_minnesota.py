from thgsp.datasets import Minnesota


def test_minnesota():
    ds = Minnesota(connected=True)
    assert ds.F.shape == (ds.A.shape[0],)
    print(ds.F.max(), ds.F.min())
