import numpy as np
from scipy.sparse import coo_matrix
from torch_sparse import SparseTensor, coalesce


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
            "RGB(3-dim) or Gray(2-dim) expected, but got {} array(or tensor)".format(img.shape))

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
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                         num_nodes)
    return edge_index, edge_attr


def consecutive_spmv(A, v, k=2):
    if v.dim() == 1:
        v = v.reshape(A.shape[-1], 1)
    for i in range(k):
        v = A @ v
    return v
