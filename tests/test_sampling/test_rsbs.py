import pytest
import numpy as np
import torch

from thgsp.datasets import Minnesota
from thgsp.graphs.generators import random_graph
from ..utils4t import remove_downloaded_dataset, devices, float_dtypes
from thgsp.sampling.rsbs import cheby_coeff4ideal_band_pass, estimate_lk, rsbs


def test_cheby_coeff4ideal_band_pass():
    order = 30
    ceoff = cheby_coeff4ideal_band_pass(0, 1, 0, 2, order)
    assert ceoff.shape == (order + 1,)
    print(ceoff)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", float_dtypes)
class TestRsbs:
    def test_estimater_lk_on_minnesota(self, dtype, device):
        ds = Minnesota(connected=True, download=True)
        g = ds[0].to(device, dtype)
        g.to_spm(device, dtype)
        lmax = g.max_frequency(lap_type="comb")
        print(lmax)
        band_limit = 10
        lambda_k, cum_coh = estimate_lk(g, band_limit, lmax=lmax, lap_type="comb", verbose=False, num_estimation=1)
        print(lambda_k)
        print(cum_coh)
        remove_downloaded_dataset("minnesota-usc")

    @pytest.mark.parametrize("return_list", [True, False])
    def test_rsbs(self, dtype, device, return_list):
        N = 100
        k = 10
        M = 30
        appropriate_num_rv = np.int32(2 * np.round(np.log(N)))
        g = random_graph(N, dtype=dtype, device=device)
        nodes = rsbs(g, M, k, num_rv=appropriate_num_rv, return_list=return_list)
        print(nodes)
        if return_list:
            assert isinstance(nodes, list)
        else:
            assert isinstance(nodes, torch.Tensor)
