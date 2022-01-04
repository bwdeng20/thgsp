import numpy as np
import torch

from thgsp.convert import get_array_module, to_scipy


def ess(operator, M, k=2):
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

    Returns
    -------
    S:  list
        A list containing sampled nodes with the sampling order

    References
    ----------
    .. [1]  Aamir Anis, et al., “Efficient sampling set selection for bandlimited graph
            signals using graph spectral proxies,” IEEE TSP, 2016.


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


def recon_ess(y, S, U, bd, **kwargs):
    """Naive implementation of ESS sampling reconstruction.

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
