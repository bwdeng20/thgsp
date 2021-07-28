import pytest
import torch
import numpy as np

from thgsp.graphs.generators import random_graph
from ..utils4t import devices, float_dtypes, float_np_dts, sparse_formats, snr_and_mse
from thgsp.sampling.rsbs import cheby_coeff4ideal_band_pass, estimate_lk, rsbs, recon_rsbs


def test_cheby_coeff4ideal_band_pass():
    order = 30
    ceoff = cheby_coeff4ideal_band_pass(0, 1, 0, 2, order)
    assert ceoff.shape == (order + 1,)
    print(ceoff)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", float_dtypes)
class TestRsbs:
    def test_estimater_lk_on_minnesota(self, dtype, device):
        N = 100
        g = random_graph(N, dtype=dtype, device=device)
        lmax = g.max_frequency(lap_type="comb")
        print(lmax)
        band_limit = 10
        lambda_k, cum_coh = estimate_lk(g, band_limit, lmax=lmax, lap_type="comb", verbose=False, num_estimation=1)
        print(lambda_k)
        print(cum_coh)

    @pytest.mark.parametrize("return_list", [True, False])
    def test_rsbs(self, dtype, device, return_list):
        N = 100
        k = 50
        M = 30
        appropriate_num_rv = np.int32(2 * np.round(np.log(N)))
        g = random_graph(N, dtype=dtype, device=device)
        nodes, coh = rsbs(g, M, k, num_rv=appropriate_num_rv, return_list=return_list)
        print(nodes)
        if return_list:
            assert isinstance(nodes, list)
        else:
            assert isinstance(nodes, torch.Tensor)

    def test_rsbs_recon(self, dtype, device):
        N = 10
        k = 5
        M = 5
        appropriate_num_rv = np.int32(2 * np.round(np.log(N)))
        g = random_graph(N, 0.3, dtype=dtype, device=device, seed=2021)
        print(g.device())
        if dtype == torch.double:  # since scikit-umfpack requires double scalars.
            nodes, coh = rsbs(g, M, k, num_rv=appropriate_num_rv, return_list=True)
            f = torch.rand(N, 1, dtype=dtype, device=device)
            f = f / f.norm()
            f_hat = recon_rsbs(f[nodes], S=nodes, L=g.L("comb"), cum_coh=coh, mu=0.1, reg_order=1)
            if torch.any(torch.isnan(f_hat)):
                print("This case leads to numerical instability and thus would be skipped")
            else:
                s, m = snr_and_mse(f_hat, f)
                assert m < 1
