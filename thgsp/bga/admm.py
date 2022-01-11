import logging
from typing import List, Optional, Tuple

import numpy as np
import ray
import torch
from scipy.sparse import block_diag, coo_matrix, lil_matrix
from torch import Tensor
from torch_sparse import SparseTensor, partition

from ._utils import (
    bipartite_mask,
    dict2perm,
    graclus_coarsen,
    graclus_refine_raw,
    is_bipartite_fix,
)

logger = logging.getLogger(__name__)


def admm_simple(A: Tensor, n_eig: Optional[int] = None, min_eig: float = 0.0) -> Tensor:
    N = A.shape[1]
    if n_eig:  # if specify the number of eigen components
        if (0 < n_eig <= N) and (n_eig % 2 == 0):
            pass
        else:
            raise ValueError(
                f" {n_eig} not a valid number of eigen component to reserve"
            )
    else:
        n_eig = N

    # eigenvalue decomposition of A, ascending
    delta, V = torch.linalg.eigh(A)

    # to descending order
    delta = torch.flip(delta, [0])
    V = torch.flip(V, [1])

    # compute eigenvalue of the bipartite bga of A
    lambda_ = A.new_zeros(n_eig)
    if N % 2 == 0:
        lambda_[: N // 2] = 0.5 * (delta[: N // 2] - torch.flip(delta[N // 2 :], [0]))
        lambda_[N // 2 :] = torch.flip(-lambda_[: N // 2], [0])
    elif N % 2 != 0:
        lambda_[: N // 2] = 0.5 * (
            delta[: N // 2] - torch.flip(delta[N // 2 + 1 :], [0])
        )
        lambda_[N // 2 + 1 :] = torch.flip(-lambda_[: N // 2], [0])

    if n_eig < N:  # delete the middle N-n_eig eigenvalues of B
        lambda_[n_eig / 2 : -n_eig / 2] = 0

    if min_eig > 0:
        idx = (torch.abs(lambda_) < min_eig).nonzero()[:, 0]
        lambda_[idx] = (lambda_[idx]).sign() * min_eig

    # recover B
    B = V.mm(torch.diag(lambda_)).mm(V.t())
    return (B + B.t()) / 2


def admm_bga(
    A: Tensor,
    M: int = 1,
    alpha: float = 100.0,
    metric: str = "fro21",
    cut_edge: bool = True,
    B0: Optional[Tensor] = None,
    convergence_marker: float = 1e-8,
    check_step: int = 1000,
    verbose: bool = False,
    nonnegative: bool = True,
    rho: float = 0.01,
    eta: float = 1.01,
    max_iter: int = int(1e5),
    early_stop: bool = False,
    max_rho: float = 1e10,
) -> Tensor:
    r"""
    The program is to find the bipartite graph approximation via solving the
    following optimization problem [2]_ .

    .. math::
        \begin{array}{ll}
        \min _{\left\{\mathbf{B}_{i}\right\}} & \sum_{i=1}^{L}\left\|\mathbf{
        A}-\mathbf{B}_{i}\right\|_{F}^{2}+\alpha
        \sum_{i=1}^{L} \sum_{j \neq i} \operatorname{Tr}\left\{\mathbf{B}_{i}^{T}
        \mathbf{B}_{j}\right\} \\
        \text { s.t. } & \mathbf{B}_{i} \in \mathcal{B}
        \end{array}

    .. note::
        This algorithm does NOT guarantee bipartite output graphs in some cases.
        Nevertheless, the output graphs are
        almost bipartite,i.e., only a few edges violate the bipartiteness. One can use
        :func:`is_bipartite_fix` to fix the output under such circumstances.

    Parameters
    ----------
    A:Tensor(N x N)
        N is the number of graph nodes
    M:int, optional
        The number of bipartite graphs to learn
    alpha: float,optional
        The parameter which controls the orthogonality among learnt bipartite graphs
    metric:str,optional
        The error metric between the learned bipartite graphs and the original graph
    cut_edge:bool,optional
         If True, cut the edges in the learned bipartite graphs that not exist in the
         original graph
    B0:Tensor,optional
         The initialization of B
    convergence_marker:float,optional
         the procedure converges if the error between B and A < this value
    check_step:int,optional
         the step interval to display iteration information
    verbose:bool,optional
    nonnegative:bool,optional
         If True, allow negative edge weight in the learned bipartite graphs B
    rho: float,optional
         A step argument
    eta: float,optional
         The update factor of rho
    max_iter:int,optional
         The max ADMM iteration times
    early_stop: bool,optional
         If True, the procedure will return B once all learned graphs are bipartite
    max_rho: float,optional

    Returns
    -------
    B:Tensor(M,N,N)
        The learnt bipartite subgraph(s)

    References
    ----------
    .. [2] Aimin Jiang, et al, "Admm-based Bipartite Graph Approximation", ICASSP, 2019.

    """
    if A.dtype is not torch.double:
        logger.warning(
            "ADMM-based method is sensitive to the precision(double is much faster)"
        )

    N = A.shape[-1]
    if cut_edge:
        disjoint_edge_mask = A == 0
    else:
        disjoint_edge_mask = None

    # initialization B
    if B0 is not None:
        assert B0.shape[-1] == N
        B0 = B0.to(A.device)
        if B0.ndim == 2:
            B = torch.stack([B0 for _ in range(M)])
        elif B0.ndim == 3:
            assert B0.shape == (M, N, N)
            B = B0
        else:
            raise RuntimeError(f"B0 is expected to be `({M},{N},{N})` or ({N},{N})")
    else:
        B = torch.stack([A.new_zeros(A.shape) for _ in range(M)])

    Z = A.new_zeros(M, N, N)
    W = A.new_zeros(M, N, N)

    # auxiliary params
    idxM = torch.arange(0, M, device=A.device)

    for times in range(max_iter):
        # Update B
        Z_tilde = Z - W / rho
        for m in range(M):
            B_bar_m = B[m != idxM].sum(0)
            if metric == "fro21":
                B_tilde = (2 * A + rho * Z_tilde[m] - alpha * B_bar_m) / (2 + rho)
            else:  # metric == 'fro22'
                B_tilde = (2 * A + rho * Z_tilde[m] - (2 + alpha) * B_bar_m) / (2 + rho)

            B[m] = B_tilde  # Eq.30

            # -----------tricks not mentioned in the paper--------------------
            # set diagonal elements equal to zeros
            B[m].fill_diagonal_(0)  # inplace op
            # cut edges disjointed in the original graph
            if cut_edge:
                B[m][disjoint_edge_mask] = 0

            B[m] = (B[m] + B[m].t()) * 0.5
            # hard thresholding operation on B
            B[m][torch.abs(B[m]) <= 1e-6] = 0

            if nonnegative:
                B[m].clamp_(min=0)  # inplace op

        # Update Z
        for m in range(M):
            Z[m] = admm_simple(B[m] + W[m] / rho)

        # Update W and rho
        W = W + rho * (B - Z)
        rho = min(eta * rho, max_rho)

        # Check convergence per check_step iterations
        if times % check_step == 1:
            M_bipartite = 0

            dist = torch.zeros(M, 2)
            for m in range(M):
                dist[m, 0] = torch.norm(B[m] - Z[m], "fro")
                if early_stop:
                    M_bipartite += is_bipartite_fix(B[m], fix_flag=True)[0]
                if verbose:
                    # eigenvalues in a ascending order
                    a = torch.linalg.eigvalsh(B[m])
                    b = torch.flip(a, dims=[0])
                    dist[m, 1] = torch.norm(a[: N // 2 + 1] + b[: N // 2 + 1])
            if verbose:
                logger.info(
                    f"Iter {times:5d}: {dist[:, 0].max().item():5.3e}\t"
                    f"{dist[:, 1].max().item():5.3e}\t{rho:5.3e}"
                )

            if times > 1 and (dist[:, 0].max().item() <= convergence_marker):
                break

            if times > max_iter:
                if verbose:
                    logger.info("\n=======> Max iteration achieved <======\n")
                break

            if early_stop and M_bipartite == M:
                logger.info(
                    "\n====> Early Stop once all subgraphs are bipartite <====\n"
                )
                break
    return B


@ray.remote(num_returns=2)
def lbga_block(Ab: Tensor, M: int, **kwargs) -> Tuple[Tensor, Tensor]:
    Nb = Ab.shape[-1]
    Bb = admm_bga(Ab, M, **kwargs)
    betab = torch.zeros(Nb, M, dtype=torch.bool, device=Ab.device)
    for i in range(M):
        _, vtx_color, _ = is_bipartite_fix(Bb[i], fix_flag=True)
        betab[:, i] = torch.as_tensor(vtx_color)
    return Bb, betab


def admm_lbga_ray(
    A: SparseTensor,
    M: int = 1,
    block_size: int = 32,
    style: int = 1,
    weighted: bool = False,
    part: str = "metis",
    num_cpus: Optional[int] = None,
    iperm: bool = True,
    verbose: bool = False,
    ray_log_to_driver: bool = False,
    **kwargs,
) -> Tuple[List[lil_matrix], Tensor, np.ndarray, np.ndarray]:
    N = A.size(-1)
    if N < block_size:
        raise RuntimeError(
            "Block size should be smaller than the number of graph nodes"
        )

    n_cluster = N // block_size
    if part == "metis":
        Ap, partptr, perm = partition(A, n_cluster, weighted)
        perm = perm.cpu().numpy()
        partptr = partptr.cpu().numpy()
        Ap_lil = Ap.to_scipy("coo").tolil()

    elif part == "graclus":
        coarsen_level = int(np.ceil(np.log2(block_size)))
        _, _, _, multi_level_clusters = graclus_coarsen(A, level=coarsen_level)
        coarsen_partition = graclus_refine_raw(multi_level_clusters)
        perm, partptr = dict2perm(coarsen_partition)
        A_lil = A.to_scipy("coo").tolil()
        Ap_lil = A_lil[np.ix_(perm, perm)]
    else:
        raise RuntimeError(f"{part} is not a valid graph partition strategy")

    if Ap_lil.dtype != np.double:
        logger.warning(
            "ADMM-based method is sensitive to precision (double is much faster)"
        )
        Ap_lil = Ap_lil.astype(np.double)

    bptG = []
    cluster_sizes = partptr[1:] - partptr[:-1]
    block_mask = block_diag(
        [np.ones((size, size), dtype=bool) for size in cluster_sizes], format="lil"
    )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=ray_log_to_driver)

    if style == 1:
        global_beta = []
        for i in range(M):
            bipartite_futures, beta_futures = list(), list()
            if verbose:
                logger.warning(
                    f"constructing {i + 1:4d}-th bipartite subgraph ... (style:1) "
                )

            for t in range(len(cluster_sizes)):
                s = partptr[t]
                e = partptr[t + 1]
                Ab = Ap_lil[s:e, s:e].toarray()
                Ab = torch.from_numpy(Ab)
                bipartite_future, beta_future = lbga_block.options(
                    num_cpus=num_cpus
                ).remote(Ab, M=1, **kwargs)
                bipartite_futures.append(bipartite_future)
                beta_futures.append(beta_future)

            Bbs = ray.get(bipartite_futures)
            local_betas = ray.get(beta_futures)
            beta = torch.cat(local_betas).squeeze_(-1)  # N x (M=1) --> (N,)
            global_beta.append(beta)

            mask = bipartite_mask(beta)
            B = block_diag([coo_matrix(Bb.squeeze_()) for Bb in Bbs], format="lil")
            append_mask = mask.copy()
            append_mask[block_mask] = False

            # add links not in diagonal block
            B[append_mask] = Ap_lil[append_mask]
            Ap_lil[mask] = 0
            bptG.append(B)
        global_beta = torch.stack(global_beta).t_()  # (N,M)

    elif style == 2:
        bipartite_futures, beta_futures = list(), list()
        for t in range(len(cluster_sizes)):
            s = partptr[t]
            e = partptr[t + 1]
            Ab = Ap_lil[s:e, s:e].toarray()
            Ab = torch.from_numpy(Ab)
            bipartite_future, beta_future = lbga_block.options(
                num_cpus=num_cpus
            ).remote(Ab, M=M, **kwargs)
            bipartite_futures.append(bipartite_future)
            beta_futures.append(beta_future)

        Bbs = ray.get(bipartite_futures)
        local_betas = ray.get(beta_futures)
        global_beta = torch.cat(local_betas)

        for i in range(M):
            mask = bipartite_mask(global_beta[:, i])
            B = block_diag([coo_matrix(Bb[i]) for Bb in Bbs], format="lil")
            append_mask = mask.copy()
            append_mask[block_mask] = 0
            B[append_mask] = Ap_lil[append_mask]
            Ap_lil[mask] = 0
            bptG.append(B)

    else:
        raise RuntimeError(f"style should be either 1 or 2, but got {str(style)}")

    if iperm:
        inv_perm = np.argsort(perm)
        trick_idx = np.ix_(inv_perm, inv_perm)
        bptG = [B[trick_idx] for B in bptG]
        global_beta = global_beta[inv_perm]

    return bptG, global_beta, partptr, perm
