from collections import deque


def bfs_lil(lil_adj, r=0, father=False):
    rows = lil_adj.rows
    tree = {}
    if father:
        tree[r] = [r]
    else:
        tree[r] = []

    arrived = [False] * lil_adj.shape[-1]
    arrived[r] = True
    q_fifo = deque([r])
    while len(q_fifo) > 0:
        u = q_fifo.popleft()
        arrived[u] = True
        nbr = rows[u]
        for v in nbr:
            if not arrived[v]:
                tree[u].append(v)
                if father:
                    tree[v] = [v]
                else:
                    tree[v] = []
                q_fifo.append(v)
    return tree
