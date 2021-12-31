import torch

from thgsp import Graph, loadmat
from thgsp.sampling.bsgda import bsgda, recon_bsgda
from thgsp.utils import mse, snr


def snr_and_mse(x, target):
    s, m = snr(x, target), mse(x, target)
    print(f"SNR: {s:.4f} | MSE: {m:.4e}")
    return s, m


dv = torch.device("cpu")
dt = torch.float

data = loadmat("bsgda.mat")
S = data["S"].ravel().astype(int) - 1

bw = data["bw"].item()
epsilon = data["epsilon"].item()
mu = data["mu"].item()
M = len(S)

f = torch.as_tensor(data["f"]).to(dt).to(dv)
y = torch.as_tensor(data["f_or"]).to(dt).to(dv)
L = Graph(data["L"]).to(dv, dt)
g = Graph(data["A"])
N = g.n_node

S2, _ = bsgda(L, M, mu, epsilon, 12, boost=False)
S1, _ = bsgda(L, M, mu, epsilon, 12)

print("MATLAB:       ", S)
print("Thgsp[boost]: ", S1)
print("Thgsp[nonbs]: ", S2)

f_hat = recon_bsgda(y[S], S, L, mu)
f_hat1 = recon_bsgda(y[S1], S1, L, mu)
f_hat2 = recon_bsgda(y[S1], S1, L, mu)
_, _ = snr_and_mse(f_hat, f)
_, _ = snr_and_mse(f_hat1, f)
_, _ = snr_and_mse(f_hat2, f)

# import numpy as np
# from thgsp.visual import draw_cn
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 2, figsize=(6, 3))
# pos = torch.rand(N, 2)
# s1 = np.zeros(N)
# s = np.zeros(N)
# s1[S1] = 1
# s[S] = 1
# draw_cn(g, pos, s1, ax=axes[0])  # plain draw of graph
# draw_cn(g, pos, s, ax=axes[1])  # plain draw of graph
# plt.show()
