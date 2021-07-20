from collections import deque


def bfs_lil(lil_adj, r=0, father=False):
    rows = lil_adj.rows
    tree = dict()
    if father:
        tree[r] = [r]
    else:
        tree[r] = list()

    arrived = set()
    q = deque([r])
    while len(q) > 0:
        u = q.popleft()
        arrived.add(u)
        nbr = rows[u]
        for v in nbr:
            if v not in arrived:
                tree[u].append(v)
                if father:
                    tree[v] = [v]
                else:
                    tree[v] = list()
                q.append(v)
                arrived.add(v)
    return tree
