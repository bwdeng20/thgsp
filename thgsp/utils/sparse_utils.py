import warnings

import numpy as np
import torch
from torch_sparse import SparseTensor, coalesce
from torch_sparse import eye as ts_eye

from thgsp.convert import coo_matrix, get_array_module, to_scipy


def img2graph(img, threshold: int = None, grid=False):
    img = np.asarray(img)
    shape = img.shape
    if len(shape) == 2:
        H, W = shape
        pixels = img.reshape(-1)
    elif len(shape) == 3:
        weights = np.array([0.3, 0.59, 0.11]).reshape(3, 1)
        C, H, W = shape
        pixels = img.reshape(C, -1)
        pixels = (weights * pixels).sum(0)
    else:
        raise RuntimeError(
            "RGB(3-dim) or Gray(2-dim) expected, but got {} array(or tensor)".format(
                img.shape
            )
        )

    def filter_edges(r, c):
        if threshold:
            diff = pixels[r] - pixels[c]
            idx = abs(diff) < threshold
            r = r[idx]
            c = c[idx]

        i = np.concatenate([r, c])
        j = np.concatenate([c, r])
        return coo_matrix((np.ones(i.shape), (i, j)), shape=(N, N))

    N = H * W
    pixel_order = np.arange(N).reshape(H, W)
    row_h = pixel_order[:, :-1].reshape(-1)
    col_h = row_h + 1
    row_v = pixel_order[:-1, :].reshape(-1)
    col_v = row_v + W
    row_r = np.concatenate([row_h, row_v])
    col_r = np.concatenate([col_h, col_v])

    Ar = filter_edges(row_r, col_r)

    row_br = pixel_order[:-1, :-1].reshape(-1)
    col_br = row_br + W + 1

    row_bl = pixel_order[:-1, 1:].reshape(-1)
    col_bl = row_bl + W - 1

    row_d = np.concatenate([row_br, row_bl])
    col_d = np.concatenate([col_br, col_bl])
    Ad = filter_edges(row_d, col_d)

    Ar = SparseTensor.from_scipy(Ar)
    Ad = SparseTensor.from_scipy(Ad)

    beta_r = np.zeros((H, W), dtype=bool)
    beta_r[::2, ::2] = 1
    beta_r[1::2, 1::2] = 1
    beta_r = beta_r.reshape(-1)

    beta_d = np.zeros((H, W), dtype=bool)
    beta_d[::2] = 1
    beta_d = beta_d.reshape(-1)

    xy = None
    if grid:
        x, y = np.meshgrid(np.arange(W), np.arange(H - 1, -1, -1))
        xy = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

    return Ar, Ad, beta_r, beta_d, pixels, xy


def remove_self_loops(edge_index, edge_attr=None):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def pool_edge(cluster, edge_index, edge_attr=None):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
    return edge_index, edge_attr


def consecutive_spmv(A: SparseTensor, v: torch.Tensor, k=2):
    if v.dim() == 1:
        v = v.reshape(A.shape[-1], 1)
    for i in range(k):
        v = A @ v
    return v


def eye(n, dtype=None, device=None):
    index, value = ts_eye(n, dtype, device)
    return SparseTensor(
        row=index[0], col=index[1], value=value, sparse_sizes=(n, n), is_sorted=True
    )


def matrix_power(A, k: int):
    if isinstance(A, SparseTensor):
        m, n = A.sparse_sizes()
        dt = A.dtype()
        dv = A.device()
    else:
        m, n = A.shape
        dt = A.dtype
        dv = A.device

    if m != n:
        raise RuntimeError("The input matrix is not square!")
    if k < 0 or not isinstance(k, int):
        warnings.warn(f"The input power{k} is not a positive integer!")
    if k == 0:
        return (
            eye(m, dt, dv)
            if isinstance(A, SparseTensor)
            else torch.eye(m, dtype=dt, device=dv)
        )

    z = result = None
    while k > 0:
        z = A if z is None else z @ z
        k, bit = divmod(k, 2)
        if bit:
            result = z if result is None else result @ z
    return result


def absv(src: SparseTensor):
    """
    The input and ouput SparseTensors will share the memory of row,col,rowptr fields
    except value.

    Parameters
    ----------
    src: SparseTensor

    Returns
    -------
    SparseTensor
    """
    val = src.storage.value()
    assert val is not None
    abs_val = val.abs()
    return src.set_value(abs_val, layout="csr")


def absv_(src: SparseTensor):
    """
    The input and ouput SparseTensors will share the memory of row,col,rowptr, etc.
    Inplace version of :func:`absv`

    Parameters
    ----------
    src: SparseTensor

    Returns
    -------
    SparseTensor
    """
    val = src.storage.value()
    assert val is not None
    return src.set_value_(val.abs_(), layout="csr")


def multivariate_normal(
    mean=0, cov=None, precision=None, num=1, delta=0.0, return_th=True
):
    """Generate signals conforming with multinormal distribution characterized
    by either covariance matrix or precision matrix.

    Parameters
    ----------
    mean:  scalar, array
         If being a scalar, this is the mean vector of all variables are equal to it;
         otherwise this array must have the length of :obj:`N`, where :obj:`N` is the
         number of variables.
    cov:  spmatrix, None, SparseTensor
    precision: spmatrix, None, SparseTensor
    num:    int
        The number of samples to generate
    delta: scalar
    return_th: bool
        If True, return Tensor instead of np.ndarray

    Returns
    -------
    z1:  array, None, Tensor
        If :arg:`cov` is specified, this consists of samples according to it;
        otherwise None.
    z2: array, None, Tensor
        If :arg:`precision` is specified, this consists of samples according to it;
        otherwise None.
    The shape is :obj:`(N,num)`
    """
    try:
        from sksparse.cholmod import cholesky
    except ImportError:
        raise ImportError(
            "scikit-sparse (https://github.com/scikit-sparse/scikit-sparse) is required"
        )

    if cov is None and precision is None:
        raise RuntimeError("One of Cov and Precision matrices is required")
    if cov is not None:
        cov = to_scipy(cov, "coo").tocsc()
    if precision is not None:
        precision = to_scipy(precision, "coo").tocsc()
    N = cov.shape[-1] if precision is None else precision.shape[-1]
    y = np.random.randn(N, num)
    z1 = None
    z2 = None
    if cov is not None:
        fc = cholesky(cov, beta=delta)
        p = fc.P()
        p_inverse = np.empty_like(p)
        p_inverse[p] = np.arange(N)
        z1 = mean + (fc.L() @ y)[p_inverse]
        z1 = torch.as_tensor(z1) if return_th else z1
    if precision is not None:
        fc = cholesky(precision, beta=delta)
        p = fc.P()
        p_inverse = np.empty_like(p)
        p_inverse[p] = np.arange(N)
        z2 = fc.solve_Lt(y, use_LDLt_decomposition=False)[p_inverse] + mean
        z2 = torch.as_tensor(z2) if return_th else z2
    return z1, z2


def sparse_xcipy_logdet(A, beta=0):
    xp, xcipy, xsplin = get_array_module(1)
    if not isinstance(
        A, (xp.ndarray, xcipy.sparse.spmatrix)
    ):  # cupy is installed but A is a cpu array
        xp, xcipy, xsplin = get_array_module(0)
    betaI = beta * xcipy.sparse.eye(A.shape[-1], dtype=A.dtype)
    slu = xsplin.splu(A + betaI)
    ld = xp.log(slu.U.diagonal()).sum()
    return ld.item()
