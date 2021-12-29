import torch
from torch_sparse import SparseTensor
from thgsp import loadmat
from thgsp.sampling.rsbs import recon_rsbs
from thgsp.utils import mse, snr


def snr_and_mse(x, target):
    s, m = snr(x, target), mse(x, target)
    print(f"SNR: {s:.4f} | MSE: {m:.4e}")
    return s, m


num_sig = 10  # repeat the signal for num_sig times

"""
Load Data captured from one trial of official Matlab code available at http://grsamplingbox.gforge.inria.fr
With the official code, the reconstruction SNR is about 30 dB, which is consistent with `rsbs` reconstruction in thgsp.
"""

data = loadmat("rsbs.mat")
coh1 = data["weight"].ravel()  # distribution of nodes being sampled
S1 = data["ind_obs"].ravel().astype(int) - 1  # sampling node set
mu1 = data["mu"].item()  # the factor of regularization term
f = torch.as_tensor(data["x"])  # the original bandlimited signal
y = torch.as_tensor(data["ynoise_init"]).repeat(1, num_sig)  # the contaminated signal

L1 = SparseTensor.from_scipy(data["L"])  # the combinatorial laplacian used

f_hat = recon_rsbs(y, S=S1, L=L1, cum_coh=coh1, mu=mu1, reg_order=1)
s, m = snr_and_mse(f_hat.view(-1, num_sig), f)
