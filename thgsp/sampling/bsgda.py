import torch
import warnings
import numpy as np
from thgsp.convert import get_ddd, get_array_module, to_xcipy, SparseTensor
from ._utils import construct_sampling_matrix, construct_hth
from typing import Tuple, List


def computing_sets(spm: SparseTensor, T, mu=0.01, p_hops=12):
    rowptr, col, wgt = spm.csr()
    wgt = col.new_ones(col.size(), dtype=spm.dtype()) if wgt is None else wgt
    ripples, ripple_size = torch.ops.thgsp.computing_sets(rowptr, col, wgt, T, mu, p_hops)
    return ripples, ripple_size


def solving_set_covering(sets, set_size, K):
    selected_pebbles, vf = torch.ops.thgsp.solving_set_covering(sets, set_size, K)
    return selected_pebbles, vf


def greedy_sampling(spm, K, T, mu=0.01, p_hops=12):
    ripples, ripple_size = computing_sets(spm, T, mu, p_hops)
    selected_pebbles, vf = solving_set_covering(ripples, ripple_size, K)
    return selected_pebbles, vf


def greedy_gda_sampling(spm, K, T, mu=0.01, p_hops=12):
    rowptr, col, wgt = spm.csr()
    wgt = col.new_ones(col.size(), dtype=spm.dtype()) if wgt is None else wgt
    selected_pebbles, vf = torch.ops.thgsp.greedy_gda_sampling(rowptr, col, wgt, K, T, mu, p_hops)
    return selected_pebbles, vf


def bsgda(spm: SparseTensor, K: int, mu: float = 0.01, epsilon: float = 1e-5, p_hops: int = 12,
          boost=True) -> Tuple[List, float]:
    r"""
    A fast deterministic vertex sampling algorithm on Gershgorin disc alignment and for smooth graph signals [2]_.

    Parameters
    ----------
    spm: SparseTensor
        The sparse adjacency matrix.
    K:  int
        The desired number of sampling nodes.
    mu: float
        The parameter for graph Laplacian based signal reconstruction. Refer to Eq(7) [2]_ for the details.
    epsilon: float
        The numerical precision for binary search (1e-5 by default
    p_hops: int
        Estimate the coverage subsets(refer to Definition 1 [2]_) within the :obj:`p_hops` neighborhood.
    boost: bool
    Returns
    -------
    sampled_nodes: List
        The sampled nodes.
    left: float
        The approximate threshold :obj:`T`, i.e., the lower bound of the eigenvalues of
        :math:`\mathbf{H}^{\top} \mathbf{H}+\mu \mathbf{L}`.

    References
    ----------
    .. [2] Y. Bai, et al., “Fast graph sampling set selection using Gershgorin disc alignment,” IEEE TSP, 2020.

    """
    assert K >= 1
    if not spm.has_value():
        spm = spm.fill_value(1.)
    adj = spm.to("cpu")
    left = 0.
    right = 1.
    T = (left + right) / 2.
    flag = False
    greedy_func = greedy_gda_sampling if boost else greedy_sampling
    while abs(right - left) > epsilon:
        _, vf = greedy_func(adj, K, T, mu, p_hops)
        if vf:
            left = T
            T = (right + left) / 2.
            flag = True
        else:
            right = T
            T = (right + left) / 2.

        if right < left:
            raise RuntimeError("Binary search error")

    if not flag:
        warnings.warn("epsilon(the precision of BS) is set too large, sub-optimal lower bound is output.")

    sampled_nodes, _ = greedy_gda_sampling(adj, K, T, mu, p_hops)
    return sampled_nodes, left


def recon_bsgda(y, S, L: SparseTensor, mu: float = 0.01, reg_order: int = 1, **kwargs):
    N = L.size(-1)
    M = len(S)
    assert N >= M > 0
    if y.ndim == 1:
        y = y.view(-1, 1)
    if y.shape[0] != M:
        raise RuntimeError(f"y is expected to have a shape ({M},num_signal) or ({M},), not {y.shape}")

    dt, dv, density, on_gpu = get_ddd(L)
    xp, xcipy, xsplin = get_array_module(on_gpu)

    L = to_xcipy(L)
    Ht = construct_sampling_matrix(N, S, dtype=dt, device=dv, layout="csr").T
    HtH = construct_hth(N, S, dtype=dt, device=dv, layout="csr")
    B = HtH + mu * L ** reg_order
    Hty = Ht @ xp.asarray(y)

    if xp == np:
        x_hat = xsplin.spsolve(B, Hty, **kwargs)
    else:
        num_sig = Hty.shape[-1]
        x_hat = xp.empty((N, num_sig), dtype=Hty.dtype)
        for j in range(num_sig):
            x_hat[:, j] = xsplin.spsolve(B, Hty[:, j], **kwargs)
    return torch.as_tensor(x_hat, device=dv)
