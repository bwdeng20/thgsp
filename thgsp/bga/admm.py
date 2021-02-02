import warnings

import numpy as np
import ray
import torch
from scipy.sparse import block_diag, lil_matrix, coo_matrix
from torch_sparse import SparseTensor, partition

from .utils import is_bipartite_fix, bipartite_mask, graclus_coarsen, graclus_refine_raw, dict2perm


def admm_simple(A, n_eig=None, min_eig=0.0):
    N = A.shape[1]
    if n_eig:  # if specify the number of eigen components
        if (0 < n_eig <= N) and (n_eig % 2 == 0):
            pass
        else:
            raise ValueError(
                " {} not a valid number of eigen component to reserve".format(n_eig))
    else:
        n_eig = N

    # eigenvalue decomposition of A, ascending
    delta, V = torch.symeig(A, eigenvectors=True)

    # to descending order
    delta = torch.flip(delta, [0])
    V = torch.flip(V, [1])

    # compute eigenvalue of the bipartite bga of A
    lambda_ = A.new_zeros(n_eig)
    if N % 2 == 0:
        lambda_[:N // 2] = 0.5 * \
                           (delta[:N // 2] - torch.flip(delta[N // 2:], [0]))
        lambda_[N // 2:] = torch.flip(-lambda_[:N // 2], [0])
    elif N % 2 != 0:
        lambda_[:N // 2] = 0.5 * \
                           (delta[:N // 2] - torch.flip(delta[N // 2 + 1:], [0]))
        lambda_[N // 2 + 1:] = torch.flip(-lambda_[:N // 2], [0])

    if n_eig < N:  # delete the middle N-n_eig eigenvalues of B
        lambda_[n_eig / 2:-n_eig / 2] = 0

    if min_eig > 0:
        idx = (abs(lambda_) < min_eig).nonzero()[:, 0]
        lambda_[idx] = (lambda_[idx]).sign() * min_eig

    # recover B
    B = V.mm(torch.diag(lambda_)).mm(V.t())
    return (B + B.t()) / 2


def admm_bga(A, M=1, alpha=100.0, metric='fro21', cut_edge=True, init_B=None,
             convergence_marker=1e-8, check_step=1000, verbose=False, nonnegative=True,
             rho=0.01, eta=1.01, max_iter=int(1e5), early_stop=False, max_rho=1e10):
    if A.dtype is not torch.double:
        warnings.warn("ADMM-based method is sensitive to the precision(double is much faster)")
    N = A.shape[0]
    if cut_edge:
        disjoint_edge_mask = (A == 0)
    else:
        disjoint_edge_mask = None

    # initialization
    if init_B:
        init_B = init_B.to(A.device)
        B = torch.stack([init_B for _ in range(M)])
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
            if metric == 'fro21':
                B_tilde = (2 * A + rho *
                           Z_tilde[m] - alpha * B_bar_m) / (2 + rho)
            else:  # metric == 'fro22'
                B_tilde = (2 * A + rho *
                           Z_tilde[m] - (2 + alpha) * B_bar_m) / (2 + rho)

            B[m] = B_tilde  # Eq.30

            # -----------tricks not mentioned in the paper--------------------
            # set diagonal elements equal to zeros
            B[m].fill_diagonal_(0)  # inplace op
            # cut edges disjointed in the original graph
            if cut_edge:
                B[m][disjoint_edge_mask] = 0

            B[m] = (B[m] + B[m].t()) * 0.5
            # hard thresholding operation on B
            B[m][abs(B[m]) <= 1e-6] = 0

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
                dist[m, 0] = torch.norm(B[m] - Z[m], 'fro')
                if early_stop:
                    M_bipartite += is_bipartite_fix(B[m], fix_flag=True)[0]
                if verbose:
                    # eigenvalues in an ascending order
                    a, _ = torch.symeig(B[m])
                    b = torch.flip(a, dims=[0])
                    dist[m, 1] = torch.norm(a[:N // 2 + 1] + b[:N // 2 + 1])
            if verbose:
                print("Iter %5d: %5.3e\t%5.3e\t%5.3e" % (times,
                                                         dist[:, 0].max(
                                                         ).item(),
                                                         dist[:, 1].max().item(), rho))

            if times > 1 and (dist[:, 0].max().item() <= convergence_marker):
                break

            if times > max_iter:
                print("\n===========> Max iteration achieved <===========\n")
                break

            if early_stop and M_bipartite == M:
                print("\n====> Early Stop once all subgraphs are bipartite <====\n")
                break
    return B


@ray.remote
def lbga_block(Ab, M, **kwargs):
    Nb = Ab.shape[-1]
    Bb = admm_bga(Ab, M, **kwargs)
    betab = Ab.new_zeros(Nb, M, dtype=torch.bool)
    for i in range(M):
        _, vtx_color, _ = is_bipartite_fix(Bb[i], fix_flag=True)
        betab[:, i] = torch.as_tensor(vtx_color)
    return Bb, betab


def admm_lbga_ray(A: SparseTensor, M=1, block_size=32, style=1, weighted=False, part="metis",
                  num_cpus=None, iperm=True, verbose=False, **kwargs):
    N = A.size(-1)
    if N < block_size:
        raise RuntimeError("Block size should be smaller than the number of graph nodes")

    n_cluster = N // block_size
    if part == "metis":
        Ap, partptr, perm = partition(A, n_cluster, weighted)
        perm = perm.cpu().numpy()
        partptr = partptr.cpu().numpy()
        Ap_lil = Ap.to_scipy('coo').tolil()

    elif part == "graclus":
        coarsen_level = int(np.ceil(np.log2(block_size)))
        _, _, _, multi_level_clusters = graclus_coarsen(A, level=coarsen_level)
        coarsen_partition = graclus_refine_raw(multi_level_clusters)
        perm, partptr = dict2perm(coarsen_partition)
        A_lil = A.to_scipy('coo').tolil()
        Ap_lil = A_lil[np.ix_(perm, perm)]
    else:
        raise RuntimeError(f"{part} is not a valid graph partition strategy")

    if Ap_lil.dtype != np.double:
        warnings.warn("ADMM-based method is sensitive to the precision(double is much faster)")
        Ap_lil = Ap_lil.astype(np.double)

    bptG = []
    cluster_sizes = partptr[1:] - partptr[:-1]
    block_mask = block_diag([np.ones((size, size), dtype=bool)
                             for size in cluster_sizes], format='lil')

    if not ray.is_initialized():
        ray.init()

    if style == 1:
        global_beta = []
        for i in range(M):
            if verbose:
                print("constructing {:4d}-th bipartite subgraph ... (style:1) ".format(i + 1))
            futures = []
            for t in range(len(cluster_sizes)):
                s = partptr[t]
                e = partptr[t + 1]
                Ab = Ap_lil[s:e, s:e].toarray()
                Ab = torch.from_numpy(Ab)
                futures.append(lbga_block.options(
                    num_cpus=num_cpus).remote(Ab, M=1, **kwargs))

            results = ray.get(futures)
            Bbs, local_betas = list(zip(*results))
            beta = torch.cat(local_betas).squeeze_(-1)  # N x (M=1) --> (N,)
            global_beta.append(beta)

            mask = bipartite_mask(beta)
            B = block_diag([coo_matrix(Bb.squeeze_()) for Bb in Bbs], format="lil")
            append_mask = mask.copy()
            append_mask[block_mask] = 0

            # add links not in diagonal block
            B[append_mask] = Ap_lil[append_mask]
            Ap_lil[mask] = 0
            bptG.append(B)
        global_beta = torch.stack(global_beta).t_()  # (N,M)

    elif style == 2:
        futures = []
        for t in range(len(cluster_sizes)):
            s = partptr[t]
            e = partptr[t + 1]
            Ab = Ap_lil[s:e, s:e].toarray()
            Ab = torch.from_numpy(Ab)
            futures.append(lbga_block.options(
                num_cpus=num_cpus).remote(Ab, M=M, **kwargs))

        results = ray.get(futures)
        Bbs, local_betas = list(zip(*results))
        global_beta = torch.cat(local_betas)

        for i in range(M):
            mask = bipartite_mask(global_beta[:, i])
            B = block_diag([coo_matrix(Bb[i]) for Bb in Bbs], format='lil')
            append_mask = mask.copy()
            append_mask[block_mask] = 0
            B[append_mask] = Ap_lil[append_mask]
            Ap_lil[mask] = 0
            bptG.append(B)

    else:
        raise RuntimeError(
            "style should be either 1 or 2, but got {}".format(str(style)))

    if iperm:
        inv_perm = np.argsort(perm)
        trick_idx = np.ix_(inv_perm, inv_perm)
        bptG = [B[trick_idx] for B in bptG]
        global_beta = global_beta[inv_perm]

    return bptG, global_beta, partptr, perm
