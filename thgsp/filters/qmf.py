from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
from scipy.sparse import diags
from torch_sparse import SparseTensor

from thgsp.bga import (
    admm_bga,
    admm_lbga_ray,
    amfs,
    beta2channel_mask,
    beta_dist2channel_name,
    harary,
    is_bipartite_fix,
    laplace,
    osglm,
)
from thgsp.graphs import Graph
from thgsp.typing import KernelType, Tensor, VertexColor

from .approximation import cheby_coeff, cheby_op, cheby_op_basis, polyval
from .filter import check_signal
from .kernels import (
    design_biorth_kernel,
    get_kernel_name,
    meyer_kernel,
    meyer_mirror_kernel,
)


class QmfCore:
    """The core implementation of GraphQmf [2]_ and GraphBiorth [3]_ filterbanks.

    .. note::
        There is no need to change :attr:`in_channels` when the signals of all input
        channels share a same filterbank since :class:`QmfCore` can automatically
        broadcast one filterbank among all input channels.

    Attributes
    ----------
    num_node: int
        The number of nodes.
    num_bgraph: int
        The number of bipartite (sub)graphs.
    beta: (Bool)Tensor(N,M)
        Each column of :attr:`beta` corresponds to one bipartite (sub)graph It has ones
        at the locations of nodes sampled by the lowpass filter, with zeros at that by
        the highpass filter.
    analyze_kernels:    List[function], Tuple[function], np.ndarray[function], optional

        1.  If None, use meyer kernels, following [2]_.

        2.  A tuple or list consisting of two kernel functions designed for low-pass and
            high-pass filtering, separately. Then the other kernels functions for all
            :obj:`Co=2^M` channels are derived.

        3.  An array of kernel functions with a shape of :obj:`(M, Co, Ci)`. :obj:`Ci`
            and :obj:`Co` is the #s of input channels, i.e., :attr:`in_channels` and
            output channels, respectively. Note that :obj:`Co` is the channels of
            wavelet coefficients and hence is actually :obj:`2^M` .

    synthesis_kernels:  List[function], Tuple[function], np.ndarray[function], optional
        This arg is almost the same with :attr:`analyze_kernels` except that this is for
        synthesis stage.
    in_channels: int, optional
        The # of input channels. Default :obj:`1`. Consider this only when you want to
        process :obj:`(N,Ci)` graph signals and for all :obj:`Ci` channels you want to
        build different graphQmf or graphBiorth wavelet filterbanks
        (via using different analysis and synthesis kernel functions).
    order: int, optional
        The order of Chebyshev approximation.
    lam_max: float, optional
        Compute the Chebyshev approximation within the interval :obj:`[0, lam_max]`.
    zero_dc: bool, optional
         If True, the zero_dc filter bank is employed.

    References
    ----------
    .. [2]  S K. Naran, et al, "Perfect Reconstruction Two-channel Wavelet Filter
            Banks for Graph Structured Data", IEEE TSP, 2012.
    .. [3]  S. Narang and A. Ortega, “Compact support biorthogonal wavelet
            filterbanks for arbitrary undirected graphs,” IEEE TSP, 2013.

    """

    def __init__(
        self,
        bipartite_graphs: List[SparseTensor],
        beta: Union[Tensor, np.ndarray],
        analyze_kernels: Optional[KernelType] = None,
        synthesis_kernels: Optional[KernelType] = None,
        in_channels: int = 1,
        order: int = 24,
        lam_max: float = 2.0,
        zero_dc: bool = False,
    ):
        if len(bipartite_graphs) != beta.shape[-1]:
            raise RuntimeError(
                f"Mismatch of # ({len(bipartite_graphs)}) of bipartite graphs "
                f"and # ({beta.shape[-1]}) of bipartite indicators"
            )
        if bipartite_graphs[0].size(-1) != beta.shape[0]:
            raise RuntimeError(
                f"Mismatch of # of nodes "
                f"({bipartite_graphs[0].size(-1)}!={beta.shape[0]})"
            )
        if lam_max <= 0:
            raise ValueError(
                f"The maximal eigenvalue ({lam_max}) of graph operator "
                f"adopted by QMF and Biorth has to be positive "
            )
        if order <= 0:
            raise ValueError("The order of Chebyshev approximation has to be positive")

        self.num_node, self.num_bgraph = beta.shape
        self.in_channels = in_channels
        self.beta = beta

        self.order = order
        self.dtype = bipartite_graphs[0].dtype()
        self.device = bipartite_graphs[0].device()

        self.lam_max = lam_max
        self.zero_dc = zero_dc

        self.bipartite_laplacian, self.bipartite_neg05pow = self.compute_laplace(
            bipartite_graphs, zero_dc=zero_dc, device=self.device
        )
        self.channel_mask, self.beta_dist = beta2channel_mask(beta)
        self.channel_mask = self.channel_mask.to(self.device)
        self.out_channels, _ = self.beta_dist.shape
        self.channel_name = beta_dist2channel_name(self.beta_dist)

        self.kernel_a = self.parse_kernels(analyze_kernels)
        self.kernel_s = self.parse_kernels(synthesis_kernels)

        self.coefficient_a = cheby_coeff(
            self.kernel_a,
            K=order,
            lam_max=lam_max,
            dtype=self.dtype,
            device=self.device,
        )

        if synthesis_kernels is None:  # GraphQmf Meyer
            self.coefficient_s = self.coefficient_a
        else:  # GraphBiorth
            self.coefficient_s = cheby_coeff(
                self.kernel_s,
                K=order,
                lam_max=lam_max,
                dtype=self.dtype,
                device=self.device,
            )

    @staticmethod
    def compute_laplace(
        bipartite_graphs: List[SparseTensor],
        zero_dc: bool = False,
        device: Optional[Union[torch.device, int, str]] = None,
    ):
        bipartite_laplacian = []
        num_bgraph = len(bipartite_graphs)
        num_node = bipartite_graphs[0].size(-1)
        bipartite_neg05pow = (
            torch.zeros(num_bgraph, num_node, device=device) if zero_dc else None
        )
        loop_index = torch.arange(num_node, device=device).unsqueeze_(0)
        for i, adj in enumerate(bipartite_graphs):
            deg = adj.sum(0)
            row, col, val = adj.clone().coo()
            deg05 = deg.pow(-0.5)
            if zero_dc:
                deg05dc = deg05.clone().detach()
                deg05dc[deg05dc == float("inf")] = 1
                bipartite_neg05pow[i] = deg05dc

            deg05[deg05 == float("inf")] = 0
            wgt = deg05[row] * val * deg05[col]
            wgt = torch.cat(
                [-wgt.unsqueeze_(0), val.new_ones(1, num_node)], 1
            ).squeeze_()

            row = torch.cat([row.unsqueeze_(0), loop_index], 1).squeeze_()
            col = torch.cat([col.unsqueeze_(0), loop_index], 1).squeeze_()
            lap = SparseTensor(row=row, col=col, value=wgt)
            bipartite_laplacian.append(lap)
        return bipartite_laplacian, bipartite_neg05pow

    def parse_kernels(self, raw_kernels):
        if raw_kernels is None:
            kernel = self.kernel_array_from_beta_dist(meyer_kernel, meyer_mirror_kernel)

        # only pass two kernel functions, the then remainder will be derived
        elif isinstance(raw_kernels, (tuple, list)):
            kernel = self.kernel_array_from_beta_dist(*raw_kernels)

        # a complete np.ndarray of kernels(functions)
        elif isinstance(raw_kernels, np.ndarray):
            assert raw_kernels.shape == (
                self.num_bgraph,
                self.out_channels,
                self.in_channels,
            )
            kernel = raw_kernels

        else:
            raise TypeError(f"{type(raw_kernels)} is not a valid type of kernel")
        return kernel

    def kernel_array_from_beta_dist(self, kernel1, kernel2):
        f1c = np.where(self.beta_dist, kernel1, kernel2)
        f1c = np.transpose(f1c)
        return np.stack([f1c] * self.in_channels, axis=-1)

    def empty_channels(self):
        empty_channels = (self.channel_mask.sum(1) == 0).nonzero().view(-1)
        return empty_channels

    def not_empty_channels(self):
        channels = (self.channel_mask.sum(1) != 0).nonzero().view(-1)
        return channels

    def __repr__(self):
        info = (
            "{}(in_channels={}, order={}, max_lambda={}, "
            "    n_channel={}, n_channel(non-empty)={}, N={},\n"
            "    analyze_kernels:\n{},\n synthesize_kernels:\n{} \n)".format(
                self.__class__.__name__,
                self.in_channels,
                self.order,
                self.lam_max,
                self.out_channels,
                len(self.not_empty_channels()),
                self.num_node,
                get_kernel_name(self.kernel_a[0], True),
                get_kernel_name(self.kernel_s[0], True),
            )
        )
        return info

    def _analyze(self, x):
        y = x  # Co x N x Ci
        for g in range(self.num_bgraph):
            y = (
                self.bipartite_neg05pow[g].pow(-1).view(1, -1, 1) * y
                if self.zero_dc
                else y
            )
            y = cheby_op(
                y,
                self.bipartite_laplacian[g],
                self.coefficient_a[g],
                lam_max=self.lam_max,
            )
        # Co x N --> Co x N x 1 for broadcast 'masked_fill_'
        mask = self.channel_mask.unsqueeze(-1)
        y.masked_fill_(~mask, 0)
        return y

    def before_analyze(self, x):
        x = check_signal(x, self.num_node, self.dtype)
        return x

    def analyze(self, x):
        """Conduct a wavelet transform on the input signal.

        .. note::
            Although the returned coefficients are a :obj:`(2^M,N,Ci)` tensor :obj:`Z`,
            there are only :obj:`N` non-zero entries in those coefficients corresponding
            to one input channel. For instance, :obj:`Z[..., 0]` has only :obj:`N`
            non-zero items.

        Parameters
        ----------
        x: Tensor
            The signal to transform. Shape: :obj:`(N,)` or :obj:`(N,Ci)`.  :obj:`Ci` and
            :obj:`N` are the numbers of input channels and graph nodes, separately.

        Returns
        -------
        Tensor
            The wavelet coefficients with a shape of :obj:`(Co,N,Ci)` or equally
            :obj:`(2^M,N,Ci)`. :attr:`M` is the # of output channels.
        """
        x = self.before_analyze(x)
        return self._analyze(x)

    def _synthesize(self, y):
        z = y
        # M-1, M-2, ..., 0 totally M bipartite graphs
        for g in range(self.num_bgraph - 1, -1, -1):
            z = cheby_op(
                z,
                self.bipartite_laplacian[g],
                self.coefficient_s[g],
                lam_max=self.lam_max,
            )
            z = self.bipartite_neg05pow[g].view(1, -1, 1) * z if self.zero_dc else z
        return z  # Co x N x Ci

    def before_synthesize(self, y):  # noqa
        return y

    def after_synthesize(self, z):  # noqa
        return z

    def synthesize(self, y):
        y = self.before_synthesize(y)
        z = self._synthesize(y)
        z = self.after_synthesize(z)
        return z


class QmfOperator:
    def __init__(self, bipartite_graphs, beta, order=24, lam_max=2.0, device=None):
        N, M = beta.shape
        assert len(bipartite_graphs) == M

        self.num_node, self.num_bgraph = N, M
        self.order = order
        self.device = device
        self.lam_max = lam_max

        # 1(n_graph) x 2(Cout) x 1(Cin) kernel
        krn = np.array([[[meyer_kernel], [meyer_mirror_kernel]]])
        # (1,2,1,K) --> (2,K)
        coeff = cheby_coeff(krn, K=order, lam_max=lam_max).squeeze_()

        operator = self.compute_basis(bipartite_graphs, coeff, beta, lam_max)
        self.operator = SparseTensor.from_scipy(operator).to(device)

        self.dtype = self.operator.dtype()
        self.device = self.operator.device()

    def transform(self, x):
        return self.operator @ x

    def inverse_transform(self, y):
        return self.operator.t() @ y

    @staticmethod
    def compute_basis(bipartite_graphs, coeff, beta, lam_max):
        M = len(bipartite_graphs)
        dt = bipartite_graphs[0].dtype
        beta = np.asarray(beta, dtype=dt)
        beta = 1 - beta * 2

        bipartite_laplacian = [
            laplace(B, lap_type="sym", add_loop=True) for B in bipartite_graphs
        ]

        H0 = cheby_op_basis(bipartite_laplacian[0], coeff[0], lam_max)
        H1 = cheby_op_basis(bipartite_laplacian[0], coeff[1], lam_max)
        t1 = H0 + H1
        t2 = H1 - H0
        Ta = t1 + diags(beta[:, 0]) * t2

        for i in range(1, M):
            H0 = cheby_op_basis(bipartite_laplacian[i], coeff[0], lam_max)
            H1 = cheby_op_basis(bipartite_laplacian[i], coeff[1], lam_max)
            t1 = H0 + H1
            t2 = H1 - H0
            Ta_sub = t1 + diags(beta[:, i]) * t2
            Ta = Ta_sub * Ta
        Ta *= 0.5 ** M
        return Ta

    def __call__(self, x):
        return self.transform(x)


class BiorthOperator:
    def __init__(self, bipartite_graphs, beta, k=4, lam_max=2.0, device=None):
        h0_c, g0_c, self.orthogonality = design_biorth_kernel(k)
        h0 = partial(polyval, torch.from_numpy(h0_c))
        h0.__name__ = "h0"
        g0 = partial(polyval, torch.from_numpy(g0_c))
        g0.__name__ = "g0"

        def h1(x):
            return g0(2 - x)

        def g1(x):
            return h0(2 - x)

        self.analysis_krn = np.array([[[h0], [h1]]])

        self.synthesis_krn = np.array([[[g0], [g1]]])

        ana_coeff = cheby_coeff(
            self.analysis_krn, K=2 * k, lam_max=lam_max
        ).squeeze_()  # (1,2,1,K) --> (2,K)
        syn_coeff = cheby_coeff(
            self.synthesis_krn, K=2 * k, lam_max=lam_max
        ).squeeze_()  # (1,2,1,K) --> (2,K)

        operator = QmfOperator.compute_basis(bipartite_graphs, ana_coeff, beta, lam_max)
        self.operator = SparseTensor.from_scipy(operator).to(device)

        inv_operator = QmfOperator.compute_basis(
            bipartite_graphs, syn_coeff, beta, lam_max
        )
        self.inv_operator = SparseTensor.from_scipy(inv_operator).to(device)

    def transform(self, x):
        return self.operator @ x

    def inverse_transform(self, y):
        return self.inv_operator.t() @ y


class ColoringBGA:
    def get_bga(self, adj, strategy, vtx_color, **kwargs):
        dv = adj.device()
        if strategy == "harary":
            bipartite_graphs, beta, beta_dist, new_vtx_color, mapper = harary(
                adj, vtx_color=vtx_color, **kwargs
            )
        elif strategy == "osglm":
            bipartite_graphs, beta, append_nodes, new_vtx_color = osglm(
                adj, vtx_color=vtx_color, **kwargs
            )
            self.append_nodes = append_nodes  # noqa
        else:
            raise RuntimeError(
                f"{strategy} is not a valid color-based decomposition algorithm."
            )
        self.vtx_color = new_vtx_color  # noqa
        self.num_node = adj.size(-1)  # noqa
        self.strategy = strategy  # noqa
        self.dtype = adj.dtype()  # noqa
        bipartite_graphs = [SparseTensor.from_scipy(B).to(dv) for B in bipartite_graphs]
        return bipartite_graphs, beta

    def before_analyze(self, x):
        x = check_signal(x, self.num_node, self.dtype)
        if self.strategy == "osglm":
            x_append = x[:, self.append_nodes, :]
            x = torch.cat([x, x_append], 1)
        return x

    def after_synthesize(self, z):
        if self.strategy == "osglm":
            z = z[:, : self.num_node, :]
        return z


class NumericalBGA:
    @staticmethod
    def get_bga(adj, strategy, num_bgraph, **kwargs):
        num_node = adj.size(-1)
        dv = adj.device()
        if strategy == "admm":
            if num_node < 80:
                bipartite_graphs_dense = admm_bga(
                    adj.to_dense().to(torch.double), M=num_bgraph, **kwargs
                )
                beta = torch.zeros(num_node, num_bgraph, dtype=torch.bool, device=dv)
                bipartite_graphs = []
                for i, B in enumerate(bipartite_graphs_dense):
                    _, vtx_color, _ = is_bipartite_fix(B, fix_flag=True)
                    beta[:, i] = torch.as_tensor(vtx_color)
                    bipartite_graphs.append(SparseTensor.from_dense(B))
                return bipartite_graphs, beta

            else:
                bipartite_graphs, beta, partptr, perm = admm_lbga_ray(
                    adj, num_bgraph, **kwargs
                )

        elif strategy == "amfs":
            bipartite_graphs, beta = amfs(adj, level=num_bgraph, **kwargs)

        else:
            raise RuntimeError(
                f"{str(strategy)} is not a valid numerical decomposition algorithm "
                f"supported at present."
            )
        bipartite_graphs = [SparseTensor.from_scipy(B).to(dv) for B in bipartite_graphs]
        return bipartite_graphs, beta


class ColorQmf(ColoringBGA, QmfCore):
    def __init__(
        self,
        G: Graph,
        kernel: Optional[KernelType] = None,
        in_channels: int = 1,
        order: int = 24,
        strategy: str = "harary",
        vtx_color: Optional[VertexColor] = None,
        lam_max: float = 2.0,
        zero_dc: bool = False,
        **kwargs,
    ):
        bipartite_graphs, beta = self.get_bga(G, strategy, vtx_color, **kwargs)
        super().__init__(
            bipartite_graphs,
            beta,
            analyze_kernels=kernel,
            in_channels=in_channels,
            order=order,
            lam_max=lam_max,
            zero_dc=zero_dc,
        )
        self.num_node = G.size(-1)  # osglm compatible


class BiorthCore(QmfCore):
    def __init__(
        self,
        bipartite_graphs: List[SparseTensor],
        beta: Union[Tensor, np.ndarray],
        k: int = 8,
        in_channels: int = 1,
        order: int = 16,
        lam_max: float = 2.0,
        zero_dc: bool = False,
    ):
        h0_c, g0_c, self.orthogonality = design_biorth_kernel(k)
        h0 = partial(polyval, torch.from_numpy(h0_c))
        h0.__name__ = "h0"
        g0 = partial(polyval, torch.from_numpy(g0_c))
        g0.__name__ = "g0"

        def h1(x):
            return g0(2 - x)

        def g1(x):
            return h0(2 - x)

        super().__init__(
            bipartite_graphs,
            beta,
            analyze_kernels=(h0, h1),
            synthesis_kernels=(g0, g1),
            in_channels=in_channels,
            order=order,
            lam_max=lam_max,
            zero_dc=zero_dc,
        )

    def __repr__(self):
        info = super().__repr__()
        info = info[:-2] + ", orthogonality={}".format(self.orthogonality)
        return info + "\n)"


class ColorBiorth(ColoringBGA, BiorthCore):
    def __init__(
        self,
        G: Graph,
        k: int = 8,
        in_channels: int = 1,
        order: int = 16,
        strategy: str = "harary",
        vtx_color: VertexColor = None,
        lam_max: float = 2.0,
        zero_dc: bool = False,
        **kwargs,
    ):
        bipartite_graphs, beta = self.get_bga(G, strategy, vtx_color, **kwargs)
        super().__init__(
            bipartite_graphs,
            beta,
            k=k,
            in_channels=in_channels,
            order=order,
            lam_max=lam_max,
            zero_dc=zero_dc,
        )
        self.num_node = G.size(-1)  # osglm compatible


class NumQmf(NumericalBGA, QmfCore):
    def __init__(
        self,
        G: Graph,
        kernel: np.ndarray = None,
        in_channels: int = 1,
        order: int = 24,
        strategy: str = "admm",
        level: int = 1,
        lam_max: float = 2.0,
        zero_dc: bool = False,
        **kwargs,
    ):
        bipartite_graphs, beta = self.get_bga(G, strategy, level, **kwargs)
        super().__init__(
            bipartite_graphs,
            beta,
            analyze_kernels=kernel,
            in_channels=in_channels,
            order=order,
            lam_max=lam_max,
            zero_dc=zero_dc,
        )


class NumBiorth(NumericalBGA, BiorthCore):
    def __init__(
        self,
        G: Graph,
        k: int = 8,
        in_channels: int = 1,
        order: int = 24,
        strategy: str = "admm",
        level: int = 1,
        lam_max: float = 2.0,
        zero_dc: bool = False,
        **kwargs,
    ):
        bipartite_graphs, beta = self.get_bga(G, strategy, level, **kwargs)

        super().__init__(
            bipartite_graphs,
            beta,
            k=k,
            in_channels=in_channels,
            order=order,
            lam_max=lam_max,
            zero_dc=zero_dc,
        )


class NumBiorth1(BiorthCore):
    def __init__(
        self,
        G: Graph,
        k: int = 8,
        in_channels: int = 1,
        order: int = 16,
        strategy: str = "admm",
        level: int = 1,
        lam_max: float = 2.0,
        zero_dc: bool = False,
        **kwargs,
    ):
        self.adj = G
        N = self.adj.size(-1)
        self.lam_max = lam_max
        self.strategy = strategy
        self.num_bgraph = level

        device = self.adj.device()
        dtype = self.adj.dtype()
        if strategy == "admm":
            if N < 80:
                bipartite_graphs_dense = admm_bga(
                    self.adj.to_dense().to(torch.double), M=level, **kwargs
                )
                beta = bipartite_graphs_dense.new_zeros(N, level).to(torch.bool)
                bipartite_graphs = []
                bipartite_graphs_dense = bipartite_graphs_dense.to(dtype).to(device)
                for i, B in enumerate(bipartite_graphs_dense):
                    _, vtx_color, _ = is_bipartite_fix(B, fix_flag=True)
                    beta[:, i] = torch.as_tensor(vtx_color)
                    bipartite_graphs.append(SparseTensor.from_dense(B))

            else:
                bipartite_graphs, beta, self.partptr, self.perm = admm_lbga_ray(
                    self.adj, level, **kwargs
                )
                bipartite_graphs = [
                    SparseTensor.from_scipy(B).to(dtype).to(device)
                    for B in bipartite_graphs
                ]

        elif strategy == "amfs":
            bipartite_graphs, beta = amfs(self.adj, level=self.num_bgraph, **kwargs)
            bipartite_graphs = [
                SparseTensor.from_scipy(B).to(dtype).to(device)
                for B in bipartite_graphs
            ]

        else:
            raise RuntimeError(
                f"{strategy} is not a valid numerical decomposition "
                f"algorithm supported at present."
            )

        super().__init__(
            bipartite_graphs, beta, k, in_channels, order, lam_max, zero_dc
        )
