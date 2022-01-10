from collections import deque


def bfs_lil(lil_adj, r=0, father=False):
    rows = lil_adj.rows
    tree = dict()
    if father:
        tree[r] = [r]
    else:
        tree[r] = list()

    arrived = [False] * lil_adj.shape[-1]
    arrived[r] = True
    q = deque([r])
    while len(q) > 0:
        u = q.popleft()
        arrived[u] = True
        nbr = rows[u]
        for v in nbr:
            if not arrived[v]:
                tree[u].append(v)
                if father:
                    tree[v] = [v]
                else:
                    tree[v] = list()
                q.append(v)
    return tree
