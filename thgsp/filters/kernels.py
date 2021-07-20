import math
from itertools import combinations

import numpy as np
import torch
from scipy.special import comb


# Utils
def get_kernel_name(kernel_array, identation=False):
    get_name = np.vectorize(lambda f: '_'.join(f.__name__.split('_')[
                                               :-1]) if '_' in f.__name__ else f.__name__)
    info = get_name(kernel_array)
    return "\t" + str(info).replace('\n', '\n\t') if identation else str(info)


def get_kernel_id(kernel_array):
    ids = np.vectorize(lambda f: id(f))(kernel_array)
    return ids


# KERNELS
# ideal_kernel(either low or high pass)
def ideal_kernel(x, c=None, cut_off=1.0, high=False):
    cons = math.sqrt(2.0)
    if c is None:
        c = cons
    else:
        assert c > 1.0
    h = x.new_zeros(x.shape)
    mask0 = x < cut_off - 1e-12
    mask1 = x > cut_off + 1e-12
    if high:
        h[mask1] = c
        h[~(mask0 | mask1)] = c / cons
    else:
        h[mask0] = c
        h[~(mask0 | mask1)] = c / cons
    return h


# Meyer kernel
def vf(x):
    y = x.new_ones(x.shape)
    mask0 = x <= 0
    mask1 = x <= 1
    mask1 = mask0 ^ mask1
    y[mask0] = 0
    y[mask1] = 3 * x[mask1].pow(2) - 2 * x[mask1].pow(3)
    return y


def meyer_kernel(x):
    h = x.new_zeros(x.shape)
    mask = x > 0
    h[mask] = torch.sqrt(2 * vf(2 - 1.5 * x[mask]))
    h[~mask] = torch.sqrt(2 * vf(2 + 1.5 * x[~mask]))
    return h


def meyer_mirror_kernel(x):
    return meyer_kernel(2. - x)


def design_p(K):
    r"""
    Auxiliary to design the biorthgonal kernel :math:`h_0(\lambda)`
    and :math:`g_0(\lambda)` metioned in :func:`design_biorth_kernel`.

    .. math::
        p(\lambda)	=(2-\lambda)^{K}[1+\sum_{m=1}^{K-1}r_{m}(\lambda-1)^{m}
        =(1-l)^{K}[1+\sum_{m=1}^{K-1}r_{m}l^{m}],\lambda=l+1.

    has :math:`K` roots at :math:`\lambda=2`, and other :math:`K-1` roots which are also roots of the residual
    polynomial below.

    ..  math::
        R(\lambda)=1+\sum_{m=1}^{K-1}r_{m}(\lambda-1)^{m}

    This function can figure out the roots of :math:`R(\lambda)` by firstly finding the zeros(roots) of :math:`R(l)`.

    Parameters
    ----------
    K:  int
        The order of :math:`p(\lambda)` is :math:`2K-1`

    Returns
    -------
    Rlam: array
        The roots of R(lam),
    r_high: float
        The coefficient of the highest degree term.
    """

    assert K % 2 == 0  # the best design usually requires K=k0+k1=2k
    Ns = np.array([K] * (K + 1))
    c1 = comb(Ns, np.arange(K + 1))  # float64 on x86-64
    A = np.zeros((2 * K, K))
    for i in range(K):
        A[i:i + K + 1, i] = c1
    # due to the expansion of (1-k)^K which includes (-1)^K for all odd degree terms
    A[:, 1::2] = -A[:, 1::2]
    A_even = A[::2]
    c_p_even = np.zeros(K)
    c_p_even[0] = 1.
    r = np.linalg.solve(A_even, c_p_even)  # 0, 1, ..., K-1 order
    zeros_of_Rl = np.roots(np.flip(r))
    idx = np.argsort(abs(zeros_of_Rl.imag))
    Rlam = zeros_of_Rl[idx] + 1
    r_high = r[-1]
    return Rlam, r_high


def estimate_orthogonality(h0, g0):
    lams = np.linspace(0, 2, 100)
    energy = (h0(lams) ** 2 + g0(2 - lams) ** 2) / 2
    A = np.sqrt(np.min(energy))
    B = np.sqrt(np.max(energy))
    theta = 1 - np.abs(B - A) / np.abs(B + A)
    return theta


def design_biorth_kernel(k):
    r"""
    Compute the coefficients of biorthogonal kernels :math:`h_0` and :math:`g_0`, which are nearly the most orthogonal
    and maximally balanced pair among all possible factorizations of

    .. math::
        p(\lambda)	=(2-\lambda)^{K}[1+\sum_{m=1}^{K-1}r_{m}(\lambda-1)^{m}],

    where :math:`K=2k`. The design scheme detailed in [1]_ satisfies :math:`k_0=k_1=k, K=2k, l_0=2k`, and
    :math:`l_1=2k-1`, implying that the residual polynomial :math:`R(\lambda)` is :math:`2k-1` degree, with exactly one
    real root and :math:`(k-1)` complex conjugate pairs of roots.
    If :math:`k` is odd, assign the only real root and :math:`(k-1)/2` complex conjugate  pairs of roots to
    :math:`h_{0}(\lambda)`. Thus we have a :math:`2k`-degree :math:`h_{0}(\lambda)` having :math:`k` roots at
    :math:`\lambda=2` and other :math:`k` roots which are zeros of :math:`R(\lambda)`. The :math:`2k-1` degree
    :math:`g_{0}(\lambda)` has :math:`k` roots at :math:`\lambda=2` and other `k-1` roots from :math:`R(\lambda)`.
    If :math:`k` is even, assign :math:`k/2` complex conjugate root pairs of :math:`R(\lambda)` to
    :math:`h_{0}(\lambda)`; the remaining :math:`k/2-1` pairs of roots and
    the only real root to :math:`g_{0}(\lambda)`.

    Parameters
    ----------
    k: int
        This integer determines the degree of all polynomial kernels.

    Returns
    -------
    h0_c:   array
        The coefficients of :math:`h_0`.
    g0_c:   array
        The coefficients of :math:`g_0`.
    theta_best:     float
        A float within :math:`[0,1]`. The closer it is to :math:`1` , the more orthogonal the wavelet bases determined
        by :math:`h_0` and :math:`g_0` are.

    Notes
    -----
    Large :math:`k(k>14)` may lead to a failed kernel design, possibly due to the accumulated computation
    error arising in the high order power operations. Empirically speaking, a smaller :math:`k(2<k<12)` works
    well.

    References
    ----------
    .. [1]  S. Narang and A. Ortega, “Compact support biorthogonal wavelet filterbanks for arbitrary
            undirected graphs,” IEEE Trans on Signal Processing, 2013.

    """
    K = 2 * k
    zeros_of_R_lam, r_highest = design_p(K)  # ascending in abs(imaginary)
    h0_c_highest = g0_c_highest = np.sqrt(r_highest)
    assert len(zeros_of_R_lam) == K - 1
    idx4rr = np.array([0])  # index for the real root of R(lam)

    idx_pair = range(1, k - 1 + 1)  # 1, ..., k-1 pair conjugate roots
    all_fac = combinations(idx_pair, k // 2)  # k is either odd or even
    k_roots = np.ones(k) * 2

    h0_best = np.zeros(K + 1)  # K, K-1, ..., 3, 2, 1, 0 order
    g0_best = np.zeros(K)  # K-1, ..., 0
    theta_best = 0.

    for one_fac in all_fac:
        idx4h0_pair = np.array(one_fac) * 2
        if k % 2 == 1:
            idx4h0 = np.concatenate([idx4h0_pair, idx4h0_pair - 1, idx4rr])
        else:  # k % 2 == 0
            idx4h0 = np.concatenate([idx4h0_pair, idx4h0_pair - 1])

        roots_h0_from_R = zeros_of_R_lam[idx4h0]
        idx4g0 = np.setdiff1d(np.arange(K - 1), idx4h0)
        roots_g0_from_R = zeros_of_R_lam[idx4g0]

        roots_h0 = np.concatenate([roots_h0_from_R, k_roots])
        roots_g0 = np.concatenate([roots_g0_from_R, k_roots])
        h0 = np.poly1d(roots_h0, True)
        g0 = np.poly1d(roots_g0, True)

        theta = estimate_orthogonality(h0, g0)
        if theta > theta_best:
            theta_best = theta
            h0_best = h0.c
            g0_best = g0.c
    sign = (-1) ** (k % 2)  # to flip it along the X-axis
    h0_c = sign * h0_c_highest * h0_best
    g0_c = sign * g0_c_highest * g0_best
    return h0_c, g0_c, theta_best


def heat_kernel(x, tau=10.):
    return torch.exp(-tau * x)
