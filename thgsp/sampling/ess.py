import numpy as np
import torch
import torch as th
from thgsp.convert import SparseTensor, get_array_module, to_scipy
from thgsp.utils import consecutive_spmv


def ess_sampling(operator, M, k=2):
    r"""
    This function has the same functionality as :func:`ess` but directly computes the matrix power of specific
    variation operator, e.g., normalized Laplacian.
    """
    import scipy.sparse.linalg as splin

    # add GPU support after cp.setdiff1d is implemented
    # dt, dv, density, on_gpu = get_ddd(operator)
    # xp, xcipy, xsplin = get_array_module(on_gpu)

    L = to_scipy(operator)
    N = L.shape[-1]
    LtL = L.T ** k * L ** k
    V = np.arange(N)
    S = list()
    while len(S) < M:
        Sc = np.setdiff1d(V, S)
        length = len(Sc)
        if length == 1:
            S.append(Sc[0])
            break
        reduced = LtL[np.ix_(Sc, Sc)]

        sigma, psi = splin.lobpcg(reduced, X=np.random.rand(length, 1), largest=False)
        psi = psi.ravel()
        v = Sc[np.argmax(np.abs(psi)).item()]
        S.append(v)
    return S


def ess(operator, M, k=2, max_iter=int(5e2)):
    r"""
    An efficient sampling set selection method for bandlimited graph signals [1]_.

    Parameters
    ----------
    operator: SparseTensor
        The chosen variation operators, e.g., graph normalized Laplacian.
    M:  int
        The number of desired sampled nodes.
    k:  int
        The proxy order. Refer to the literature for details.
    max_iter:   int
        The maximum number of iterations acceptable in Power Iteration.

    Returns
    -------
    S:  list
        A list containing sampled nodes with the sampling order

    References
    ----------
    .. [1]  Aamir Anis, Akshay Gadde, and Antonio Ortega, “Efficient sampling set selection for bandlimited graph
            signals using graph spectral proxies,” IEEE Trans on Signal Processing, 2016.

    """
    N = operator.size(-1)
    V = np.arange(N)
    S = []
    while len(S) < M:
        Sc = np.setdiff1d(V, S)
        if len(Sc) == 1:
            S.append(Sc[0])
            break
        sigma, psi = power_iteration4min(operator, Sc, k, max_iter)
        v = Sc[th.argmax(psi.abs())]
        S.append(v)
    return S


def power_iteration(L: SparseTensor, Sc: iter, k=2, shift=0, num_iter=50, tol=1e-6):
    Sc = th.as_tensor(Sc)
    Lt = L.t()
    I = SparseTensor.eye(L.size(0), dtype=L.dtype(), device=L.device())
    Isv = I[Sc, :]
    Ivs = I[:, Sc]
    x0 = th.rand(len(Sc), 1, device=L.device(), dtype=L.dtype())
    for _ in range(num_iter):
        x1 = Ivs @ x0
        x1 = consecutive_spmv(L, x1, k)
        x1 = consecutive_spmv(Lt, x1, k)
        x1 = Isv @ x1
        if shift != 0:
            x1 = x1 - shift * x0
        x1 = x1 / x1.norm()
        if (x1 - x0).norm() < tol:
            break
        x0 = x1
    t = Ivs @ x0
    t = consecutive_spmv(L, t, k)
    lam = (t ** 2).sum()
    if shift != 0:
        lam = lam - shift * (x0 ** 2).sum()
    return lam.item(), x0


def power_iteration4min(L: SparseTensor, Sc: iter, k=2, num_iter=50):
    lam_max, _ = power_iteration(L, Sc, k, num_iter=num_iter)
    lam_min_minus_max, v = power_iteration(L, Sc, k, shift=lam_max, num_iter=num_iter)
    lam_min = lam_min_minus_max + lam_max
    return lam_min, v


def recon_ess(y, S, U, bd, **kwargs):
    """
    Naive implementation of ESS sampling reconstruction.

    Parameters
    ----------
    y:  Tensor
        Dense Shape: :obj:`(N)`
    S:  List
        The sampling set
    U:  Tensor
        Dense :obj:`(N, bd)`
    bd: int
        The bandwidth of target signal
    kwargs: dict
        The optional arguments of `xp.linalg.lstsq`
    Returns
    -------
    f_hat:  Tensor
        The reconstructed signal
    """
    assert bd > 1
    assert len(S) > 1
    dv = U.device
    xp, _, _ = get_array_module(U.is_cuda)
    tmp = xp.linalg.lstsq(xp.asarray(U[S, :bd]), xp.asarray(y), **kwargs)[0]
    f_hat = U[:, :bd] @ torch.as_tensor(tmp, device=dv)
    return f_hat
