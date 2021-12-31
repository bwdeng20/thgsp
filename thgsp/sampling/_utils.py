import torch

from thgsp.convert import SparseTensor, to_xcipy


def construct_sampling_matrix(
    N, S, dtype=None, device=None, layout="csr", return_ts=False
):
    r"""
    Construct the sampling matrix :math:`\mathbf{H} \in\{0,1\}^{M \times N}` defined as
    follows.

    .. math::
       \mathbf{H}_{i j}= \begin{cases}1, & j=\mathcal{S}_{i} \\ 0, & \text { otherwise }
       \end{cases}

    Parameters
    ----------
    N: int
        The total number of nodes
    S:  list, array, torch.Tensor
        The 1-D list of sampled nodes.
    dtype: torch.dtype, optional
        The dtype
    layout: str
        The memory layout of the generated sparse matrix. One of ("csc", "csr", "coo").
    device: torch.device, str, optional
        If :py:`True` and  `cupy` is installed, use `cupy`as backend; otherwise `scipy`.
    return_ts: bool,
        If False, return `scipy.sparse.spmatrix` of device is GPU else
        'cupyx.scipy.sparse.spmatrix'

    Returns
    -------
    H: spmatrix
        The :obj:`(M,N)` scipy sparse matrix with :obj:`layout` format

    """
    M = len(S)
    row = torch.arange(M, device=device)
    col = torch.as_tensor(S, device=device)
    data = torch.ones(M, dtype=dtype, device=device)

    spa = SparseTensor(row=row, col=col, value=data, sparse_sizes=(M, N))
    H = spa if return_ts else to_xcipy(spa, layout)
    return H


def construct_hth(N, S, D=None, dtype=None, device=None, layout="csr", return_ts=False):
    M = len(S)
    row = torch.as_tensor(S, device=device)
    col = row.clone()
    data = torch.ones(M, dtype=dtype, device=device)
    if D is not None:
        data = data * torch.as_tensor(D, dtype=dtype, device=device)

    spa = SparseTensor(row=row, col=col, value=data, sparse_sizes=(N, N))
    HtH = spa if return_ts else to_xcipy(spa, layout)
    return HtH


def construct_dia(
    S,
    diag_data,
    ps=True,
    inverse=False,
    dtype=None,
    device=None,
    layout="csr",
    return_ts=False,
):
    N = len(diag_data)
    M = len(S)
    row = torch.arange(M, device=device) if ps else torch.arange(N, device=device)
    col = row.clone()
    shape = (M, M) if ps else (N, N)
    data = torch.as_tensor(diag_data, dtype=dtype, device=device)
    data = data ** -1 if inverse else data
    data = data[S] if ps else data
    spa = SparseTensor(row=row, col=col, value=data, sparse_sizes=shape)
    P = spa if return_ts else to_xcipy(spa, layout)
    return P
