import torch
import warnings
import numpy as np
from numba import jit, prange
from thgsp.graphs.core import GraphBase, SparseTensor
from thgsp.filters import cheby_op
from thgsp.convert import get_ddd, to_xcipy, to_xp, get_array_module
from ._utils import construct_sampling_matrix, construct_dia


@jit('f8[:](f8,f8,f8,f8,i4)', nopython=True)
def cheby_coeff4ideal_band_pass(a, b, lmin, lmax, order):
    a1 = (lmax - lmin) / 2
    a2 = (lmin + lmax) / 2
    a = (a - a2) / a1
    b = (b - a2) / a1

    coeff = np.zeros(order + 1)
    acosa = np.arccos(a)
    acosb = np.arccos(b)
    coeff[0] = 2 / np.pi * (acosa - acosb)
    for j in prange(1, order + 1):
        coeff[j] = 2 / (np.pi * j) * (np.sin(j * acosa) - np.sin(j * acosb))

    return coeff


def estimate_lk(G, k, num_estimation=1, num_rv=None, epsilon=1e-2, lmin=None, lmax=None,
                return_coherence=True, order=30, lap_type="comb", verbose=False):
    r"""
    Estimate the optimal rsbs_recon_compare2matlab distribution according to which the bandlimited graph signals are
    sampled [3]_ .

    Parameters
    ----------
    G: GraphBase
        The graph
    k: int
        The :obj:`k`-th smallest eigenvalue of graph Laplacian
    num_estimation: int
        The number of times the estimation of :math:`\lambda_{k}` is going to run
    num_rv:    int, None
        The number of random vectors used
    epsilon: float
        The tolerance of binary search to find approximated :math:`\lambda_{k}`
    lmin: float
        The smallest frequency of graph Laplacian
    lmax: float
        The largest frequency of graph Laplacian
    return_coherence: bool
        If :obj:`True`, return the estimated square of graph local cumulative coherence [3]_ of all nodes
    order: int
        The order of the Chebyshev approximation
    lap_type: str
        :obj:`comb`, :obj:`sym`, and :obj:`rw` represent combinatorial, symmetric normalized, and random-walk normalized
        Laplacian, separately
    verbose: bool

    Returns
    -------
    lambda_k: float
        The eventual estimated :obj:`k`-th smallest graph frequency
    cum_coh:  Tensor or(None)
        If :obj:`return_coherence` is :obj:`True` , return the estimated graph local cumulative coherence [3]_ of
        every node, otherwise :obj:`None`

    References
    ----------
    .. [3] G. Puy, N. Tremblay, R. Gribonval, and P. Vandergheynst, “Random rsbs_recon_compare2matlab of
            bandlimited signals on graphs,” Applied and Computational Harmonic Analysis, 2018.
    """
    N = G.size(1)
    appropriate_num_rv = np.int32(2 * np.round(np.log(N)))
    if num_rv is None:
        num_rv = appropriate_num_rv
    elif num_rv < appropriate_num_rv:
        warnings.warn(f"Using at least {appropriate_num_rv} random vectors are recommended.")
        num_rv = appropriate_num_rv
    else:
        if verbose:
            print(f"Use {num_rv} random vectors to estimate the distribution")

    L = G.L(lap_type)
    if lmin is None:
        lmin = 0.
    if lmax is None:
        lmax = G.max_frequency(lap_type)

    device = G.device()
    dtype = G.dtype()

    x = None
    coeff = torch.zeros(1, num_rv, order + 1, dtype=dtype, device=device)
    norm_UK = torch.zeros(num_estimation, N, dtype=dtype, device=device) if return_coherence else None
    estimated_lam_k = np.zeros(num_estimation)
    for i in range(num_estimation):
        sig = torch.randn(N, num_rv, dtype=dtype, device=device) / np.sqrt(num_rv)
        counts = 0
        lambda_min, lambda_max = lmin, lmax
        while counts != k or (lambda_max - lambda_min) / lambda_max > epsilon:
            lambda_mid = (lambda_min + lambda_max) / 2
            coeff[...] = torch.from_numpy(cheby_coeff4ideal_band_pass(0., lambda_mid, 0., lmax, order))
            x = cheby_op(sig, L, coeff, lmax).squeeze_()
            counts = torch.round_(torch.sum(x ** 2))
            if counts >= k:
                lambda_max = lambda_mid
            else:
                lambda_min = lambda_mid
            if verbose:
                print(f"[estimating lambda_k]counts: {int(counts):8d}, bottom: {lambda_min:.4f}, top: {lambda_max:.4f}")
        estimated_lam_k[i] = (lambda_min + lambda_max) / 2
        if verbose:
            print(
                f"{i:4d} estimation lambda_k: {estimated_lam_k[i]:8f}, bottom: {lambda_min:.4f}, top: {lambda_max:.4f}")

        if return_coherence:
            norm_UK[i] = (x ** 2).sum(1)

    lambda_k = np.mean(estimated_lam_k)
    if verbose:
        print(f"Final lambda_k: {lambda_k:.4f}")
    cum_coh = torch.mean(norm_UK, 0) if return_coherence else None
    return lambda_k, cum_coh


def rsbs(G, M, k=None, num_estimation=1, num_rv=None, epsilon=1e-2, lmin=None, lmax=None, order=30, lap_type="comb",
         return_list=False, verbose=False):
    r"""
    Random rsbs_recon_compare2matlab algorithm for bandlimited signals [3]_ .

    Parameters
    ----------
    G: GraphBase
        The graph to be handled
    M: int
        The number of vertices to be sampled
    k: int, optional
        The :obj:`k`-th smallest eigenvalue of graph Laplacian. If None, :obj:`k = M`.
    num_estimation: int
        The number of times the estimation of :math:`\lambda_{k}` is going to run
    num_rv:    int, None
        The number of random vectors used
    epsilon: float
        The tolerance of binary search to find approximated :math:`\lambda_{k}`
    lmin: float
        The smallest frequency of graph Laplacian
    lmax: float
        The largest frequency of graph Laplacian
    return_list: bool
        If :obj:`True`, return :class:`List` otherwise a :obj:`Tensor` having the same :obj:`dtype` and :obj:`device` as
        the input graph :obj:`G` .
    order: int
        The order of the Chebyshev approximation
    lap_type: str
        :obj:`comb`, :obj:`sym`, and :obj:`rw` represent combinatorial, symmetric normalized, and random-walk normalized
        Laplacian, separately
    verbose: bool

    Returns
    -------
    sampled_nodes: List, Tensor
        The rsbs_recon_compare2matlab set
    cum_coh: Tensor
        The rsbs_recon_compare2matlab possibilities of all nodes

    """
    lambda_k, cum_coh = estimate_lk(G, k, num_estimation, num_rv, epsilon, lmin, lmax, True, order, lap_type, verbose)
    sampled_nodes = torch.multinomial(cum_coh, M, replacement=True)
    sampled_nodes = sampled_nodes.cpu().tolist() if return_list else sampled_nodes
    return sampled_nodes, cum_coh


def recon_rsbs(y, S, L: SparseTensor, cum_coh, mu: float = 0.01, reg_order: int = 1, **kwargs):
    N = L.size(-1)
    M = len(S)
    assert M > 0
    if y.shape[0] != M:
        raise RuntimeError(f"y is expected to have a shape ({M},num_signal) or ({M},), not {y.shape}")

    dt, dv, density, on_gpu = get_ddd(L)
    xp, xcipy, xsplin = get_array_module(on_gpu)

    yp = to_xp(y).astype("d")

    L = to_xcipy(L, layout="csr").astype("d")
    H = construct_sampling_matrix(N, S, torch.double, dv)
    Psinv = construct_dia(S, cum_coh, ps=True, inverse=True, dtype=torch.double, device=dv)

    Bl = H.T * Psinv
    B = Bl * H + mu * L ** reg_order

    HPy = Bl @ yp

    if xp == np:
        x_hat = xsplin.spsolve(B, HPy, **kwargs)
    else:
        num_sig = yp.shape[-1]
        x_hat = xp.empty((N, num_sig), dtype=yp.dtype)
        for j in range(num_sig):
            x_hat[:, j] = xsplin.spsolve(B, HPy[:, j], **kwargs)

    return torch.as_tensor(x_hat, device=dv)
