import torch
import numpy as np
from tqdm import trange
from functools import partial
from thgsp.convert import get_ddd
from thgsp.graphs.core import GraphBase, SparseTensor
from thgsp.filters import cheby_coeff, heat_kernel, cheby_op_basis
from thgsp.utils import matrix_power, absv
from thgsp.convert import to_xcipy, get_array_module


def fastgsss(G: GraphBase, M, bandwidth=100, nu=75., cheby=True, order=12):
    """
    FastGSSS proposed in [4]_ .

    Parameters
    ----------
    G:      GraphBase
        The graph whose signals are to sample
    M:      int
        The desired number of sampled nodes
    bandwidth: int
        The estimated bandwidth of original signals
    nu: float
        The parameter controlling the width of the heat diffusion filter kernel.
    cheby:  bool
        If :obj:`True`, compute the localization operator with Chebyshev approximation; otherwise
        compute it from a direct expensive EVD.
    order: int
        The order of Chebyshev approximation.

    Returns
    -------
    S:      List
        The sample set
    T: SparseTensor
        The localization operator

    References
    ----------
    .. [4]  A. Sakiyama, Y. Tanaka, T. Tanaka, and A. Ortega, "Eigendecomposition-free rsbs_recon_compare2matlab set selection
            for graph signals," IEEE Transactions on Signal Processing, 202.

    """
    N = G.size(1)
    num_edge = G.numel() // 2
    pe = num_edge / N  # edge probability
    ps = M / N  # rsbs_recon_compare2matlab ratio
    pf = bandwidth / N  # normalized bandwidth

    assert pe > 0
    assert ps > 0
    assert pf > 0

    dtype = G.dtype()

    if not cheby:
        fs, U = G.spectral(lap_type="sym")
        lmax = fs.max().item()
        g_lam = heat_kernel(fs, tau=nu * pe * ps * pf / lmax)
        T_g_tmp = U @ torch.diag(g_lam) @ U.t()
        T = SparseTensor.from_dense(T_g_tmp)
        T_g_tmp = SparseTensor.from_dense(T_g_tmp.abs_())

    else:
        heat_krn = partial(heat_kernel, tau=nu * pe * ps * pf / 2.)
        coeff = cheby_coeff(heat_krn, order, dtype=dtype)  # cpu coefficient
        T = cheby_op_basis(G.L("sym"), coeff.squeeze_(), lam_max=2., return_st=True)
        T_g_tmp = absv(T)

    S = []
    T_g = T_g_tmp.sum(0)
    T_g[S] = 0
    selected = torch.argmax(T_g).item()
    S.append(selected)

    for _ in trange(1, M, initial=1, desc=f"Sampling nodes", total=M):
        Ts = T_g_tmp[:, S].sum(1)
        W = Ts.mean() - Ts
        W = W.relu_()
        T_g = T_g_tmp @ W.view(-1, 1)
        T_g[S] = 0
        selected = torch.argmax(T_g).item()
        S.append(selected)

    return S, T


def recon_fastssss(y, S, T, order, sd=0.5):
    """
    A primary implementation of reconstruction method associated with "FastSSS" rsbs_recon_compare2matlab algorithm.

    Parameters
    ----------
    y:  Tensor
        The measurements on rsbs_recon_compare2matlab set :obj:S:. If the localization operator :obj:`T` has a density
        greater than the threshold :obj:`sd`, :obj:`y` has a shape of either :obj:`(M,)`，:obj:`(M,1)`，
        or :obj:`(M,C)`; otherwise :obj:`y` could only be  either :obj:`(M,)` or :obj:`(M,1)`.
    S:  List
        A list consisting of all indices of sampled nodes.
    T:  Tensor, SparseTensor
        The localization operator
    order: int
    sd: float
        The threshold of :obj:`T`'s density that controls when we use a dense or sparse linear solver.

    Returns
    -------
    Tensor
        The signal recovered from the measurements :obj:`y` .
    """
    T_k = matrix_power(T, order)
    dt, dv, density, on_gpu = get_ddd(T_k)
    if density > sd:
        T_k = T_k.to_dense()
        tmp = torch.linalg.solve(T_k[np.ix_(S, S)], torch.as_tensor(y))
        x_hat = T_k[:, S] @ tmp

    else:
        T_k = to_xcipy(T_k)
        xp, xcipy, xsplin = get_array_module(on_gpu)
        tmp = xsplin.spsolve(T_k[xp.ix_(S, S)], xp.asarray(y))
        x_hat = T_k[:, S] * tmp
    return torch.as_tensor(x_hat, device=dv)
