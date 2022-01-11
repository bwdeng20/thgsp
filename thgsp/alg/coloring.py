from collections import deque

import numpy as np
import torch


def dsatur_py(spm):
    n_node = spm.size(-1)
    ptr, col, _ = spm.csr()
    ptr = ptr.cpu().numpy()
    col = col.cpu().numpy()

    vtx_color = [-1 for _ in range(n_node)]
    # distinct colors of each neighbor
    distinct_colors = {v: set() for v in range(n_node)}

    deg = ptr[1:] - ptr[:-1]
    u = deg.argmax().item()  # choose the node with largest degree
    vtx_color[u] = 0

    nbr = col[ptr[u] : ptr[u + 1]]
    for v in nbr:
        distinct_colors[v].add(0)

    for i in range(1, n_node):
        saturation = {
            v: len(c) for v, c in distinct_colors.items() if vtx_color[v] == -1
        }
        u = max(saturation, key=lambda v: (saturation[v], deg[v]))

        nbr = col[ptr[u] : ptr[u + 1]]
        nbr_colors = {vtx_color[v] for v in nbr}

        vtx_color[u] = next(color for color in range(n_node) if color not in nbr_colors)

        c = vtx_color[u]
        for v in nbr:
            distinct_colors[v].add(c)

    return np.asarray(vtx_color)


def dsatur_cpp(spm):
    ptr, col, _ = spm.csr()
    vtx_color = torch.ops.thgsp.dsatur(ptr, col)  # noqa
    return vtx_color.cpu().numpy()


def dsatur(spm):
    try:
        return dsatur_cpp(spm)
    except RuntimeError:
        return dsatur_py(spm)


def check_coloring(spm, vtx_color):
    n_node = spm.size(0)
    ptr, col, _ = spm.csr()
    ptr = ptr.cpu().numpy()
    col = col.cpu().numpy()
    vtx2reach = set(range(n_node))
    # BFS traversal binary coloring -1 means no color
    while len(vtx2reach) > 0:
        r = vtx2reach.pop()
        vtx2reach.add(r)
        q_fifo = deque([r])
        while len(q_fifo) > 0:
            cur_node = q_fifo.popleft()
            vtx2reach.remove(cur_node)
            nbr = col[ptr[cur_node] : ptr[cur_node + 1]]
            for i in nbr:
                if vtx_color[i] == vtx_color[cur_node]:
                    return False
    return True
