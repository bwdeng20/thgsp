import warnings

import torch

from thgsp.convert import get_array_module, get_ddd, to_xcipy

warnings.filterwarnings("ignore", message="Exited at iteration", category=UserWarning)


def ess(operator, M, k=2, block_size=2):
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
    block_size:  int
        The block size of lobpcg.
    Returns
    -------
    S:  list
        A list containing sampled nodes with the sampling order

    References
    ----------
    .. [1]  Aamir Anis, et al., “Efficient sampling set selection for bandlimited graph
            signals using graph spectral proxies,” IEEE TSP, 2016.


    """
    dt, dv, density, on_gpu = get_ddd(operator)
    xp, xcipy, xsplin = get_array_module(on_gpu)

    L = to_xcipy(operator)
    N = L.shape[-1]
    LtL = L.T**k * L**k
    V = xp.arange(N)
    S = list()
    while len(S) < M:
        Sc = xp.setdiff1d(V, S)
        length = Sc.shape[0]
        if length == 1:
            S.append(Sc[0].item())
            break
        reduced = LtL[xp.ix_(Sc, Sc)]

        guess = xp.random.rand(length, block_size)
        sigma, psi = xsplin.lobpcg(reduced, X=guess, largest=False, maxiter=100)

        psi = psi[:, 0].ravel()
        v = Sc[xp.argmax(psi**2).item()].item()
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
    tmp = xp.linalg.lstsq(xp.asarray(U[S, :bd]), xp.asarray(y), **kwargs, rcond=None)[0]
    f_hat = U[:, :bd] @ torch.as_tensor(tmp, device=dv)
    return f_hat
