import torch
import warnings
from torch_sparse import SparseTensor
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


def bsgda(spm: SparseTensor, K: int, mu: float = 0.01, epsilon: float = 1e-5, p_hops: int = 12) -> Tuple[List, float]:
    r"""
    A fast deterministic vertex sampling algorithm on Gershgorin disc alignment and for smooth graph signals [2]_.

    Parameters
    ----------
    spm: SparseTensor
        The sparse adjacency matrix.
    K:  int
        The desired number of sampling nodes.
    mu: float
        The parameter for graph Laplacian based signal reconstruction. Refer to Eq(7)[2]_ for the details.
    epsilon: float
        The numerical precision for binary search (1e-5 by default
    p_hops: int
        Estimate the coverage subsets(refer to Definition 1[2]_) within the :obj:`p_hops` neighborhood.

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
    left = 0.
    right = 1.
    T = (left + right) / 2.
    flag = False
    while abs(right - left) > epsilon:
        _, vf = greedy_gda_sampling(spm, K, T, mu, p_hops)
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

    sampled_nodes, _ = greedy_gda_sampling(spm, K, T, mu, p_hops)
    return sampled_nodes, left
