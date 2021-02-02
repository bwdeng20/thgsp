from .utils import is_bipartite_fix


def greedy_bga(A, iterations=5, verbose=False):
    A = A.to_scipy(layout='csr')
    flag, vtx_color = is_bipartite_fix(A, fix_flag=False)
    if flag:
        return A, vtx_color
    best_B = None
    thresh = float('inf')
    for i in range(iterations):
        flag, vtx_color, Br = is_bipartite_fix(A, fix_flag=True)
        err = (A - Br).power(2).sum()

        if thresh > err:
            best_B = Br
            thresh = err
        if verbose:
            print("Iter: {:3d}, \t FrobeniusNorm^2: {:4f}".format(i, err))
    return best_B, vtx_color
