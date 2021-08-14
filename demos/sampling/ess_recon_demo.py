import torch
from thgsp.sampling import recon_ess, ess_sampling, ess
from thgsp.utils import mse, snr
from thgsp import loadmat, Graph


def snr_and_mse(x, target):
    s, m = snr(x, target), mse(x, target)
    print(f"SNR: {s:.4f} | MSE: {m:.4e}")
    return s, m


dv = torch.device("cpu")
dt = torch.double

data = loadmat("ess-comm.mat")
S = data['S'].ravel().astype(int) - 1

bw = data['bw'][0, 0].astype(int)
order = data['K'].astype(int)[0, 0]
print(order)

graph = Graph(data['A']).to(dv).to(dt)
f = torch.as_tensor(data['f']).to(dt).to(dv)
y = torch.as_tensor(data['f_or']).to(dt).to(dv)
L = graph.L("comb")
U = graph.U("comb")

M = len(S)
S1 = ess_sampling(L, M, k=order // 2)
print(S)
print(S1)

f_hat1 = recon_ess(y[S], S, U, bw)
f_hat2 = recon_ess(y[S1], S1, U, bw)

s, m = snr_and_mse(f_hat1, f.to(dv))
_, _ = snr_and_mse(f_hat2, f.to(dv))
