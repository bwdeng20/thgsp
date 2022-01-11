import numpy as np
import pytest
import torch

from thgsp.filters import Filter, check_signal, ideal_kernel, meyer_kernel
from thgsp.graphs import random_graph


@pytest.fixture(scope="module")
def filters():
    N = 4
    g = random_graph(N, density=0.7, dtype=torch.float)
    g = g.set_value_(torch.ones(g.nnz(), device=g.device(), dtype=g.dtype()))
    assert g.is_symmetric()
    krns1 = np.array(
        [
            [meyer_kernel, ideal_kernel],
            [ideal_kernel, meyer_kernel],
            [meyer_kernel, ideal_kernel],
        ]
    )  # 3(Cout) x 2(Cin) kernel

    def identity(x):
        return 1

    krns2 = np.tile(np.array([identity]), [3, 2])
    krns2[1][0] = meyer_kernel
    krns2[2][1] = meyer_kernel
    filters = {
        "11": Filter(g, meyer_kernel),
        "23": Filter(g, krns1, order=20),
        "plain": Filter(g, krns2, order=50),
        "13": Filter(g, identity, in_channels=1, out_channels=3),
        "31": Filter(g, identity, in_channels=3, out_channels=1),
    }
    return filters


class TestFilter:
    def test_filter_evaluate(self, filters):
        for k in filters:
            flt = filters[k]
            kernel_response_on_spectrum = flt.evaluate()
            num_node = flt.G.n_node
            assert kernel_response_on_spectrum.shape == (*flt.kernels.shape, num_node)

    def test_check_signal(self, filters):
        flt = filters["23"]
        with pytest.raises(RuntimeError):  # rank-4 tensor not valid
            check_signal(torch.rand(4, 2, 4, 10), 5)
        # N. rank-1
        check_signal(torch.rand(flt.N), flt.N)

        # N x Ci. rank-2
        check_signal(torch.rand(flt.N, flt.Ci), flt.N)

        # Co x N x Ci . rank-3
        check_signal(torch.rand(flt.Co, flt.N, flt.Ci), flt.N)

        # M(!=N)x Ci. not valid
        with pytest.raises(RuntimeError):
            check_signal(torch.rand(100, flt.Ci), flt.N)

    def test_evd_filter(self, filters):
        flt = filters["plain"]
        x = torch.rand(flt.N, flt.Ci)
        y = flt.filter(x)
        assert y.shape == (flt.Co, flt.N, flt.Ci)
        print(x)
        print(y)
        assert torch.allclose(y[0], x, atol=1e-6)
        assert not torch.allclose(y[1], x)
        assert not torch.allclose(y[2], x)

    def test_cheby_filter(self, filters):
        flt = filters["plain"]
        x = torch.rand(flt.N, flt.Ci)
        y_hat = flt.cheby_filter(x)
        assert y_hat.shape == (flt.Co, flt.N, flt.Ci)
        print("x:\n", x)
        print("y_hat:\n", y_hat)
        print("y_hat[0] - x:\n", y_hat[0] - x)
        assert torch.allclose(y_hat[0], x, atol=1e-5)
        assert not torch.allclose(y_hat[1], x)
        assert not torch.allclose(y_hat[2], x)
        y = flt.filter(x)
        print("y:\n", y)
        print("y[0]-x: \n", y[0] - x)

    def test_call(self, filters):
        for k in filters:
            flt = filters[k]
            x = torch.rand(flt.N, flt.Ci)
            y = flt(x)
            if flt.Co == 1:
                assert y.shape == (flt.N,)
            else:
                assert y.shape == (flt.N, flt.Co)
