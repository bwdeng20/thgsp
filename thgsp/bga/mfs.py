from typing import List, Tuple

from scipy.sparse import eye, lil_matrix
from scipy.sparse.csgraph import breadth_first_order, structural_rank
from scipy.sparse.linalg import inv
from torch_sparse import SparseTensor

from thgsp.alg import dsatur
from thgsp.utils import sparse_xcipy_logdet

from ._utils import bipartite_mask, laplace, np


def amfs(
    A: SparseTensor,
    Sigma=None,
    level=None,
    delta=0.1,
    thresh_kld=1e-6,
    priority=True,
    verbose=False,
) -> Tuple[List[lil_matrix], np.ndarray]:
    r"""
    AMFS bipartite approximation for graph  wavelet signal processing [3]_.

    Parameters
    ----------
    A:          SparseTensor
        The adjacency matrix.
    Sigma:      scipy.spmatrix, optional
        The covariance matrix specified by the Laplacian matrix L. If None,
        :math:`\Sigma^{-1}=L+\delta I`
    level:      int, optional
        The number of bipartite subgraphs, i.e., the decomposition level. If None,
        :math:`level=\lceil log_2( \mathcal{X}) \rceil`, where :math:`\mathcal{X}` is
        the chromatic number of  :obj:`A`.
    delta:      float, optional
        :math:`1/\delta` is interpreted as the variance of the DC compnent. Refer to
        [4]_ for more details.
    thresh_kld: float, optional
        Threshold of Kullback-Leibler divergence to perform `AMFS` decomposition.
    priority:   bool,optional
        If True, KLD holds priority.
    verbose:    bool,optional

    Returns
    -------
    bptG:   List[SparseTensor]
        The bipartite subgraphs.
    beta:   Tensor(N, M)
        The indicator of bipartite sets

    References
    ----------
    .. [3]  Jing Zen, et al, "Bipartite Subgraph Decomposition for Critically
            Sampledwavelet Filterbanks on Arbitrary Graphs," IEEE trans on SP, 2016.
    .. [4]  A. Gadde, et al, "A probablistic interpretation of sampling theory of graph
            signals". ICASSP, 2015.

    """

    N = A.size(-1)
    # compute_sigma consists of laplace matrix which prefers "coo"
    A = A.to_scipy(layout="coo").astype("d")
    if Sigma is None:
        Sigma = compute_sigma(A, delta)
    else:
        assert Sigma.shape == (N, N)
    if level is None:
        chromatic = dsatur(A).n_color
        level = np.ceil(np.log2(chromatic))

    A = A.tolil()
    beta = np.zeros((N, level), dtype=bool)
    bptG = [lil_matrix((N, N), dtype=A.dtype) for _ in range(level)]
    for i in range(level):
        if verbose:
            print(f"\n|----decomposition in level: {i:4d} ----|")
        s1, s2 = amfs1level(A, Sigma, delta, thresh_kld, priority, verbose)
        bt = beta[:, i]
        bt[s1] = 1  # set s1 True
        mask = bipartite_mask(bt)
        bptG[i][mask] = A[mask]
        A[mask] = 0
    return bptG, beta


def amfs1level(
    W: lil_matrix,
    Sigma: lil_matrix = None,
    delta=0.1,
    thresh_kld=1e-6,
    priority=True,
    verbose=True,
):
    if Sigma is None:
        Sigma = compute_sigma(W, delta)
    N = W.shape[-1]
    not_arrived = np.arange(N)
    nodes = breadth_first_order(W, i_start=0, return_predecessors=False)
    not_arrived = np.setdiff1d(not_arrived, nodes)
    s1 = [0]
    s2 = []
    nodes = nodes[1:]

    while len(not_arrived) > 0:
        new_root = not_arrived[0]
        other_nodes = breadth_first_order(
            W, i_start=new_root, return_predecessors=False
        )
        not_arrived = np.setdiff1d(not_arrived, other_nodes)
        s1.append(new_root)
        nodes = np.append(nodes, other_nodes[1:])

    balance_flag = True
    for i, v in enumerate(nodes):
        if verbose:
            print("handling {:5d}-th node: {:5d}, ".format(i, v), end="")
        N1 = len(s1)
        s = [*s1, v, *s2]
        W_local = W[np.ix_(s, s)]

        Wb1 = W_local.copy()
        Wb2 = W_local.copy()
        Wb2[:N1, :N1] = 0
        Wb2[N1:, N1:] = 0
        Wb1[: N1 + 1, : N1 + 1] = 0
        Wb1[N1 + 1 :, N1 + 1 :] = 0
        if priority:  # KLD holds priority
            S_local = Sigma[np.ix_(s, s)]
            DK1 = dkl(Wb1, S_local, delta)
            DK2 = dkl(Wb2, S_local, delta)
            diff = DK1 - DK2
            if verbose:
                print("DK1-DK2: {:5f}".format(diff))
            if abs(diff) > thresh_kld:
                if diff > 0:
                    s2.append(v)
                else:
                    s1.append(v)
            else:
                rank1 = structural_rank(Wb1.tocsr())
                rank2 = structural_rank(Wb2.tocsr())
                if rank1 > rank2:
                    s1.append(v)
                elif rank1 < rank2:
                    s2.append(v)
                else:
                    if balance_flag:
                        s1.append(v)
                    else:
                        s2.append(v)
                    balance_flag = not balance_flag
        else:
            rank1 = structural_rank(Wb1)
            rank2 = structural_rank(Wb2)
            if rank1 > rank2:
                s1.append(v)
            elif rank1 < rank2:
                s2.append(v)
            else:
                S_local = Sigma[np.ix_(s, s)]
                DK1 = dkl(Wb1, S_local, delta)
                DK2 = dkl(Wb2, S_local, delta)
                if DK1 < DK2:
                    s1.append(v)
                elif DK1 > DK2:
                    s2.append(v)
                else:
                    if balance_flag:
                        s1.append(v)
                    else:
                        s2.append(v)
                    balance_flag = not balance_flag
    return s1, s2


def dkl(Wb: lil_matrix, Sigma, delta: float):
    N = Wb.shape[-1]
    Lb = laplace(Wb, lap_type="comb").tocsc()  # coo -> csc
    temp = Lb + delta * eye(N, dtype=Lb.dtype, format="csc")
    detemp = sparse_xcipy_logdet(temp)
    try:
        dk = (Lb @ Sigma).diagonal().sum() - detemp  # cholesky prefers `csc`
    except Exception as err:
        raise err
    return dk


def compute_sigma(A, delta, precision_mat=False) -> lil_matrix:
    Sigma_inv = laplace(A, lap_type="comb").tocsc() + delta * eye(
        A.shape[-1], dtype=A.dtype, format="csc"
    )
    if precision_mat:
        return Sigma_inv
    Sigma = inv(Sigma_inv)  # csc more efficient
    Sigma = Sigma + Sigma.T
    Sigma.data *= 0.5
    return Sigma.tolil()
