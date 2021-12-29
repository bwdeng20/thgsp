import collections


def is_bipartite(adj):
    n_node = adj.size(0)
    ptr, col, _ = adj.csr()
    ptr = ptr.cpu().numpy()
    col = col.cpu().numpy()
    vtx2reach = set(range(n_node))
    # BFS traversal binary coloring -1 means no color
    # the same initial color for all nodes
    vtx_color = [-1 for _ in range(n_node)]
    while len(vtx2reach) > 0:
        r = vtx2reach.pop()
        vtx2reach.add(r)
        q = collections.deque([r])
        while len(q) > 0:
            cur_node = q.popleft()
            vtx2reach.remove(cur_node)
            if vtx_color[cur_node] == -1:
                vtx_color[cur_node] = 0
            nbr = col[ptr[cur_node] : ptr[cur_node + 1]]
            for i in nbr:
                if vtx_color[i] == -1:
                    vtx_color[i] = 1 - vtx_color[cur_node]
                    q.append(i)
                elif vtx_color[i] == vtx_color[cur_node]:
                    return False, vtx_color
    return True, vtx_color
