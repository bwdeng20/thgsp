import numpy as np
import torch
from scipy.sparse import csr_matrix, eye
from torch_sparse import SparseTensor


def normalize_laplace(L: SparseTensor, lam_max: float = 2.):
    Ln = L.clone()
    row, col, val = Ln.coo()
    diag_mask = row == col
    val[...] = (2. * val) / lam_max
    val.masked_fill_(val == float('inf'), 0)
    val[diag_mask] -= 1
    return Ln


def cheby_op(x: torch.Tensor, L: SparseTensor, coeff: torch.Tensor, lam_max: float = 2.):
    """ Chebyshev approximation of graph filtering

    Parameters
    ----------
    x:          Tensor
        The input graph signal. It's shape can be either :obj:`(N,)` , :obj:`(N,Ci)` or :obj:`(Co,N,Ci)`, wherein
        :obj:`N`, :obj:`Ci` and :obj:`Co` are the numbers of nodes, input channels, and output channels respectively.
    L:          SparseTensor
        The :obj:`(N,N)` Laplacian matrix.
    coeff:      Tensor
        The :obj:`(Co,Ci,K+1)` Chebyshev coefficients for :obj:`Ci*Co` kernels, wherein :obj:`K` is the order of
        approximation.
    lam_max:    float,optional
        The maximal graph frequency, i.e., :math:`\lambda_{max}`

    Returns
    -------
    Tensor
        The filtered signals of shape :obj:`(Co,N,Ci)`
    """
    Co, Ci, K = coeff.shape
    N = L.size(-1)

    if x.dim() == 1:
        assert x.size() == (N,)
        x = x[None, ..., None]  # (N,) --> 1 x N x 1
    elif x.dim() == 2:  # N x Ci --> Co x N x Ci
        assert x.size() == (N, Ci)
        x = x.unsqueeze(0)
    elif x.dim() == 3:  # Co x N x Ci
        assert x.size() == (Co, N, Ci) or (1, N, Ci)
    else:
        raise RuntimeError(
            "The input signals has mismatched dimensions: {}".format(x.size()))

    K = K - 1
    c = coeff.unsqueeze(1)  # Co x Ci x K --> Co x 1 x Ci x K
    L_norm = normalize_laplace(L, lam_max)
    twf_old = x
    twf_cur = L_norm @ x  # Co x N x Ci
    result = 0.5 * c[..., 0] * twf_old + c[..., 1] * twf_cur
    for k in range(2, K + 1):
        twf_new = 2 * (L_norm @ twf_cur) - twf_old
        result = result + c[..., k] * twf_new
        twf_old = twf_cur
        twf_cur = twf_new

    return result


def cheby_op_basis(L: csr_matrix, coeff: torch.Tensor, lam_max=2.):
    Co, K = coeff.shape
    K = K - 1
    dt = L.dtype
    c = np.asarray(coeff)
    N = L.shape[-1]
    I = eye(N, dtype=L.dtype)
    Ln = L * (2 / lam_max) - I
    Tl_old = I
    Th_old = I
    Tl_cur = Ln
    Th_cur = Ln
    Hl = 0.5 * c[0, 0] * Tl_old + c[0, 1] * Tl_cur
    Hh = 0.5 * c[1, 0] * Th_old + c[1, 1] * Th_cur
    for k in range(2, K + 1):
        Tl_new = 2 * Ln @ Tl_cur - Tl_old
        Hl = Hl + c[0, k] * Tl_new
        Tl_old = Tl_cur
        Tl_cur = Tl_new

        Th_new = 2 * Ln @ Th_cur - Th_old
        Hh = Hh + c[1, k] * Th_new
        Th_old = Th_cur
        Th_cur = Th_new
    return Hl.astype(dt), Hh.astype(dt)


def cheby_coeff(kernels, K=10, lam_max=2., num_points=None, dtype=None, device=None):
    if num_points is None:
        num_points = K + 1
    assert lam_max > 0

    if isinstance(kernels, np.ndarray):
        M, Co, Ci = kernels.shape
    else:  # pass only a single kernel function
        kernels = np.array([[[kernels]]])  # 1 x 1 x 1 array
        M = Co = Ci = 1

    points = np.pi * (torch.arange(num_points, dtype=dtype,
                                   device=device) + 0.5) / num_points

    gs = torch.zeros(M, Co, Ci, 1, num_points, dtype=dtype, device=device)
    kernel_cache = dict()
    for m in range(M):
        for i in range(Ci):
            for j in range(Co):
                krn = kernels[m, j, i]
                kid = id(krn)
                if kid in kernel_cache:
                    gs[m, j, i] = kernel_cache[kid]
                else:  # kernel_cache stores the reference to a part of memory used by gs.
                    gs[m, j, i] = krn(lam_max / 2 * (torch.cos(points) + 1))
                    kernel_cache[kid] = gs[m, j, i]
    order_cos = torch.arange(
        K + 1, dtype=dtype, device=device).reshape(1, -1) * points.reshape(-1, 1)
    order_cos.cos_()  # num x K+1
    coeff = (gs @ order_cos) * 2. / num_points  # M x Co x Ci x 1 x K+1
    return coeff.squeeze_(-2)  # M x Co x Ci x K+1


def polyval(c, x):
    """
    Evaluate `N`-order polynomial at the points `x` with the given `N+1` coefficients
    in descending order.

    Parameters
    ----------
    c:  Tensor, a list of Tensor
        The coefficients can be either a list of  tensors which are broadcastable with the input `x`
        or an concatenated tensor of them.

    x:  Tensor, scalar
        Arbitrary rank-D tensor(D=1,2,...) is valid due to the broadcasting semantics.

    Returns
    -------
    Tensor
           A tensor with the same shape as `x`
    """
    order = len(c) - 1
    f = c[0] * x + c[1]
    for i in range(2, order + 1):
        f = f * x + c[i]
    return f


def nla(x, frac=0.4, k=None, scheme='abs'):
    Co, N, Ci = x.shape

    if k is not None:
        k_largest = k
    else:
        k_largest = int(frac * N)

    fuse = x.reshape(-1, Ci)
    if scheme is 'abs':
        _, idx = fuse.abs().topk(k_largest, dim=0)
        val = fuse.gather(0, idx)
    elif scheme is 'naive':
        val, idx = fuse.topk(k_largest, dim=0)

    elif scheme is "keeplow":
        fuse_high = fuse[N:, ...]
        _, idx = fuse_high.abs().topk(k_largest, dim=0)
        val = fuse_high.gather(0, idx)
        res = fuse_high.new_zeros((Co - 1) * N, Ci)
        res.scatter_(0, idx, val)
        return torch.cat([x[0, ...], res]).reshape(Co, N, Ci)

    else:
        raise RuntimeError(
            "{} is not a valid supported non-linear approximation scheme".format(scheme))
    res = x.new_zeros(Co * N, Ci)
    res.scatter_(0, idx, val)
    return res.reshape(Co, N, Ci)


def hard_threshold(x, T=0.3, lowest=False):
    if not lowest:
        x[x.abs() < T] = 0
    else:
        x_high_pass = x[1:, ...]
        x_high_pass[x_high_pass.abs() < T] = 0
    return x
