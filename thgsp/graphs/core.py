from typing import Optional

import networkx as nx
import torch
from scipy.sparse.linalg import eigsh

from thgsp.convert import to_torch_sparse, SparseTensor
from .degree import in_degree, out_degree, degree_matrix
from .is_bipartite import is_bipartite
from .laplace import laplace


class GraphBase(SparseTensor):
    def __init__(self, adjacency, coords: Optional[torch.Tensor] = None, cache=True, requires_grad=False, **kwargs):

        try:  # torch.Tensor, np.ndarray, scipy.spmatrix
            M, N = adjacency.shape
        except AttributeError:  # SparseTensor
            M, N = adjacency.sizes()
        assert M == N

        if coords is not None:
            assert coords.shape[0] == N
            coords = torch.as_tensor(coords)
        self._n_node = N
        self.coords = coords
        self.cache = cache

        adj = to_torch_sparse(adjacency)
        row, col, value = adj.coo()
        rowptr, _, _ = adj.csr()
        super(GraphBase, self).__init__(row=row, rowptr=rowptr, col=col, value=value, sparse_sizes=(N, N),
                                        is_sorted=True)
        self.requires_grad_(requires_grad)

        self.extra = kwargs
        # cached members
        self._lap_type = None
        self._L = None
        self._fs = None
        self._U = None

    @property
    def lap_type(self):
        return self._lap_type

    @property
    def n_node(self) -> int:
        return self._n_node

    def edge_info(self):
        row, col, val = self.coo()
        edge_idx = torch.cat([row.clone().unsqueeze_(0), col.clone().unsqueeze_(0)], 0)
        val = val.clone()
        return edge_idx, val

    def L(self, lap_type: str = "sym"):
        r"""
        Compute a specific type of Laplacian matrix. If :py:attr:`self._lap_type` equals to :obj:`lap_type` and
        :py:attr:`self.cache` is True, then return the cached Laplacian matrix. Note that every time you compute
        Laplacian with a different type from the last, the cached matrix will be overwritten.

        Parameters
        ----------
        lap_type: str
            One of ["sym", "comb", "rw"]

        Returns
        -------
        The Laplacian matrix
        """
        assert lap_type in ["sym", "comb", "rw"]
        if self._L is not None and lap_type == self._lap_type:
            lap = self._L
        else:
            lap = laplace(self, lap_type)
            if self.cache:
                self._L = lap
                self._lap_type = lap_type
        return lap

    def U(self, lap_type: str = "sym"):
        lap = self.L(lap_type)
        if self._U is not None:
            U = self._U
        else:
            fs, U = torch.symeig(lap.to_dense(), eigenvectors=True)
            if self.cache:
                fs[fs.abs() < 1e-6] = 0
                self._fs = fs
                self._U = U
        return U

    def spectrum(self, lap_type: str = "sym"):
        lap = self.L(lap_type)
        if self._fs is not None:
            fs = self._fs
        else:
            fs, _ = torch.symeig(lap.to_dense(), eigenvectors=False)
            if self.cache:
                self._fs = fs
        fs[fs.abs() < 1e-6] = 0  # for stability
        return fs

    def max_frequency(self, lap_type: str = "sym"):
        lap = self.L(lap_type)
        if self._fs is not None:
            max_f = max(self._fs).item()
        else:
            max_f = eigsh(lap.to_scipy(layout="csr"), k=1, which="LM", return_eigenvectors=False)[0]
        return max_f

    def spectral(self, lap_type: str = "sym"):
        U = self.U(lap_type)
        fs = self.spectrum(lap_type)
        return fs, U

    def D(self, indeg=True):
        return degree_matrix(self, indeg)

    def get_extra(self, k):
        return self.extra[k]

    def degree(self, bunch=None):
        raise NotImplementedError

    def n_edge(self):
        raise NotImplementedError

    @property
    def is_directed(self):
        raise NotImplementedError

    @classmethod
    def from_networkx(cls, nxg):
        if type(nxg) == nx.DiGraph:
            tor_type = DiGraph
        elif type(nxg) == nx.Graph:
            tor_type = Graph
        else:
            raise TypeError(
                "{} not supported in thgsp at present".format(type(nxg)))

        n_node = nxg.number_of_nodes()
        # <class 'scipy.sparse.csr.csr_matrix'>
        sparse_adj_mat = nx.adjacency_matrix(nxg, nodelist=range(n_node))
        return tor_type(sparse_adj_mat)

    def to_networkx(self, directed=False):
        if directed:
            graph_type = nx.DiGraph
        else:
            graph_type = nx.Graph

        sci_spm = self.to_scipy(layout='csr')
        nxg = nx.from_scipy_sparse_matrix(sci_spm, create_using=graph_type)
        return nxg


class Graph(GraphBase):
    def __init__(self, adjacency,
                 coords: Optional[torch.Tensor] = None,
                 cache=False, requires_grad=False, **kwargs):
        adj = to_torch_sparse(adjacency).to_symmetric(reduce='mean')
        super(Graph, self).__init__(adj, coords, cache, requires_grad, **kwargs)
        self._is_directed = False

    def degree(self, bunch=None):
        return in_degree(self, bunch)

    @property
    def is_directed(self):
        return self._is_directed

    def n_edge(self):
        return self.nnz() / 2

    @property
    def is_bipartite(self):
        return is_bipartite(self)[0]


class DiGraph(GraphBase):
    def __init__(self, adjacency,
                 coords: Optional[torch.Tensor] = None,
                 cache=False, requires_grad=False, **kwargs):
        super(DiGraph, self).__init__(adjacency, coords, cache, requires_grad, **kwargs)
        self._is_directed = True

    def in_degree(self, bunch=None):
        return in_degree(self, bunch)

    def out_degree(self, bunch=None):
        return out_degree(self, bunch)

    def degree(self, bunch=None):
        return self.in_degree(bunch) + self.out_degree(bunch)

    @property
    def is_directed(self):
        return self._is_directed

    def n_edge(self):
        return self.nnz() / 2
