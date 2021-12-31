from functools import partial
from typing import List

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

from .approximation import cheby_coeff, cheby_op, cheby_op_basis, polyval
from .kernels import (
    design_biorth_kernel,
    get_kernel_name,
    meyer_kernel,
    meyer_mirror_kernel,
)


class QmfCore:
    r"""The core implementation of GraphQmf [2]_ and GraphBiorth [3]_ filterbanks.

    .. note::
        There is no need to change :attr:`in_channels` when the signals of all input
        channels share a same filterbank since :class:`QmfCore` can automatically
        broadcast one filterbank among all input channels.

    Attributes
    ----------
    N: int
        The number of nodes.
    M: int
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
    zeroDC: bool, optional
         If True, the zeroDC filter bank is employed.


    References
    ----------
    .. [2]  S K. Naran, et al, "Perfect Reconstruction Two-channel Wavelet Filter
    Banks for Graph Structured Data",
            IEEE trans on Signal Processing, 2012.
    .. [3]  S. Narang and A. Ortega, “Compact support biorthogonal wavelet
    filterbanks for arbitrary undirected graphs,”
            IEEE Trans on Signal Processing, 2013.

    """

    def __init__(
        self,
        bptG: List[SparseTensor],
        beta: torch.Tensor,
        analyze_kernels=None,
        synthesis_kernels=None,
        in_channels=1,
        order=24,
        lam_max=2.0,
        zeroDC=False,
    ):
        assert len(bptG) == beta.shape[-1]
        assert bptG[0].size(-1) == beta.shape[0]
        assert lam_max > 0
        assert order > 0

        self.N, self.M = beta.shape
        self.in_channels = self.Ci = in_channels

        self.bptG = bptG
        self.beta = beta

        self.order = order
        self.dtype = bptG[0].dtype()
        self.device = bptG[0].device()

        self.lam_max = lam_max
        self.zeroDC = zeroDC

        self.bptL, self.bptD05 = self.compute_laplace(bptG)
        self.channel_mask, self.beta_dist = beta2channel_mask(beta)
        self.channel_mask = self.channel_mask.to(self.device)
        self.out_channels, _ = self.beta_dist.shape
        self.Co = self.out_channels  # alias of out_channels
        self.channel_name = beta_dist2channel_name(self.beta_dist)

        self.kernel_a = self.parse_kernels(analyze_kernels)
        self.kernel_s = self.parse_kernels(synthesis_kernels)

        self.coefficient_a = cheby_coeff(
            self.kernel_a,
            lam_max=lam_max,
            K=order,
            dtype=self.dtype,
            device=self.device,
        )

        if synthesis_kernels is None:  # GraphQmf Meyer
            self.coefficient_s = self.coefficient_a
        else:  # GraphBiorth
            self.coefficient_s = cheby_coeff(
                self.kernel_s,
                lam_max=lam_max,
                K=order,
                dtype=self.dtype,
                device=self.device,
            )

    def compute_laplace(self, bptG):
        bptL = []
        N = bptG[0].size(-1)
        bptD05 = (
            torch.zeros(self.M, self.N, device=self.device) if self.zeroDC else None
        )
        loop_index = torch.arange(N, device=self.device).unsqueeze_(0)
        for i, adj in enumerate(bptG):
            deg = adj.sum(0)
            row, col, val = adj.clone().coo()
            deg05 = deg.pow(-0.5)
            if self.zeroDC:
                deg05dc = deg05.clone().detach()
                deg05dc[deg05dc == float("inf")] = 1
                bptD05[i] = deg05dc

            deg05[deg05 == float("inf")] = 0
            wgt = deg05[row] * val * deg05[col]
            wgt = torch.cat([-wgt.unsqueeze_(0), val.new_ones(1, N)], 1).squeeze_()

            row = torch.cat([row.unsqueeze_(0), loop_index], 1).squeeze_()
            col = torch.cat([col.unsqueeze_(0), loop_index], 1).squeeze_()
            lap = SparseTensor(row=row, col=col, value=wgt)
            bptL.append(lap)
        return bptL, bptD05

    def parse_kernels(self, raw_kernels):
        if raw_kernels is None:
            kernel = self.kernel_array_from_beta_dist(meyer_kernel, meyer_mirror_kernel)

        # only pass two kernel functions, the then remainder will be derived
        elif isinstance(raw_kernels, (tuple, list)):
            kernel = self.kernel_array_from_beta_dist(*raw_kernels)

        # a complete np.ndarray of kernels(functions)
        elif isinstance(raw_kernels, np.ndarray):
            assert raw_kernels.shape == (self.M, self.Co, self.Ci)
            kernel = raw_kernels

        else:
            raise TypeError(f"{type(raw_kernels)} is not a valid type of kernel")
        return kernel

    def kernel_array_from_beta_dist(self, kernel1, kernel2):
        f1c = np.where(self.beta_dist, kernel1, kernel2)
        f1c = np.transpose(f1c)
        return np.stack([f1c] * self.Ci, axis=-1)

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
                self.N,
                get_kernel_name(self.kernel_a[0], True),
                get_kernel_name(self.kernel_s[0], True),
            )
        )
        return info

    def _check_signal(self, x):
        if x.dim() == 1:  # N -> 1 x  N x 1
            x = x.reshape(1, -1, 1)
        elif x.dim() == 2:  # N x Ci -> 1 x N x Ci
            x = x.unsqueeze(0)
        elif x.dim() == 3:  # keep Co x N x Ci or 1 x N x Ci # Check for synthesis
            x = x
        else:
            raise RuntimeError(
                "rank-1,2,3 tensor expected, but got rank-{}".format(x.dim())
            )

        if x.shape[-2] != self.N:
            raise RuntimeError(
                f"The penultimate dimension of signal:{x.shape[-2]}!= the number of "
                f"nodes: {self.N}"
            )

        return x.to(self.dtype)

    def _analyze(self, x):
        y = x  # Co x N x Ci
        for g in range(self.M):
            y = self.bptD05[g].pow(-1) * y if self.zeroDC else y
            y = cheby_op(y, self.bptL[g], self.coefficient_a[g], lam_max=self.lam_max)
        # Co x N --> Co x N x 1 for broadcast 'masked_fill_'
        mask = self.channel_mask.unsqueeze(-1)
        y.masked_fill_(~mask, 0)
        return y

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
        x = self._check_signal(x)
        return self._analyze(x)

    def _synthesize(self, y):
        z = y
        for g in range(
            self.M - 1, -1, -1
        ):  # M-1, M-2, ..., 0 totally M bipartite graphs
            z = cheby_op(z, self.bptL[g], self.coefficient_s[g], lam_max=self.lam_max)
            z = self.bptD05[g] * z if self.zeroDC else z
        return z  # Co x N x Ci

    def synthesize(self, y):
        y = self._check_signal(y)
        return self._synthesize(y)


class QmfOperator:
    def __init__(self, bptG, beta, order=24, lam_max=2.0, device=None):
        N, M = beta.shape
        assert len(bptG) == M

        self.N, self.M = N, M
        self.order = order
        self.device = device
        self.lam_max = lam_max

        krn = np.array(
            [[[meyer_kernel], [meyer_mirror_kernel]]]
        )  # 1(n_graph) x 2(Cout) x 1(Cin) kernel
        # (1,2,1,K) --> (2,K)
        coeff = cheby_coeff(krn, K=order, lam_max=lam_max).squeeze_()

        operator = self.compute_basis(bptG, coeff, beta, lam_max)
        self.operator = SparseTensor.from_scipy(operator).to(device)

        self.dtype = self.operator.dtype()
        self.device = self.operator.device()

    def transform(self, x):
        return self.operator @ x

    def inverse_transform(self, y):
        return self.operator.t() @ y

    @staticmethod
    def compute_basis(bptG, coeff, beta, lam_max):
        M = len(bptG)
        dt = bptG[0].dtype
        beta = np.asarray(beta, dtype=dt)
        beta = 1 - beta * 2

        bptL = [laplace(B, lap_type="sym", add_loop=True) for B in bptG]

        H0 = cheby_op_basis(bptL[0], coeff[0], lam_max)
        H1 = cheby_op_basis(bptL[0], coeff[1], lam_max)
        t1 = H0 + H1
        t2 = H1 - H0
        Ta = t1 + diags(beta[:, 0]) * t2

        for i in range(1, M):
            H0 = cheby_op_basis(bptL[i], coeff[0], lam_max)
            H1 = cheby_op_basis(bptL[i], coeff[1], lam_max)
            t1 = H0 + H1
            t2 = H1 - H0
            Ta_sub = t1 + diags(beta[:, i]) * t2
            Ta = Ta_sub * Ta
        Ta *= 0.5 ** M
        return Ta

    def __call__(self, x):
        return self.transform(x)


class BiorthOperator:
    def __init__(self, bptG, beta, k=4, lam_max=2.0, device=None):
        h0_c, g0_c, orthogonality = design_biorth_kernel(k)
        h0 = partial(polyval, torch.from_numpy(h0_c))
        h0.__name__ = "h0"
        g0 = partial(polyval, torch.from_numpy(g0_c))
        g0.__name__ = "g0"

        def h1(x):
            return g0(2 - x)

        def g1(x):
            return h0(2 - x)

        self.orthogonality = orthogonality
        self.analysis_krn = np.array([[[h0], [h1]]])

        self.synthesis_krn = np.array([[[g0], [g1]]])

        ana_coeff = cheby_coeff(
            self.analysis_krn, K=2 * k, lam_max=lam_max
        ).squeeze_()  # (1,2,1,K) --> (2,K)
        syn_coeff = cheby_coeff(
            self.synthesis_krn, K=2 * k, lam_max=lam_max
        ).squeeze_()  # (1,2,1,K) --> (2,K)

        operator = QmfOperator.compute_basis(bptG, ana_coeff, beta, lam_max)
        self.operator = SparseTensor.from_scipy(operator).to(device)

        inv_operator = QmfOperator.compute_basis(bptG, syn_coeff, beta, lam_max)
        self.inv_operator = SparseTensor.from_scipy(inv_operator).to(device)

    def transform(self, x):
        return self.operator @ x

    def inverse_transform(self, y):
        return self.inv_operator.t() @ y


class ColorQmf(QmfCore):
    def __init__(
        self,
        G: Graph,
        kernel=None,
        in_channels=1,
        order=24,
        strategy="harary",
        vtx_color=None,
        lam_max=2.0,
        zeroDC=False,
        **kwargs,
    ):
        self.adj = G
        self.strategy = strategy

        if strategy == "harary":
            bptG, beta, beta_dist, vtx_color, mapper = harary(
                self.adj, vtx_color=vtx_color, **kwargs
            )
        elif strategy == "osglm":
            bptG, beta, append_nodes, vtx_color = osglm(
                self.adj, vtx_color=vtx_color, **kwargs
            )
            self.append_nodes = append_nodes
        else:
            raise RuntimeError(
                f"{strategy} is not a valid color-based decomposition algorithm."
            )
        self.vtx_color = vtx_color

        bptG = [SparseTensor.from_scipy(B).to(G.device()) for B in bptG]

        super(ColorQmf, self).__init__(
            bptG,
            beta,
            analyze_kernels=kernel,
            in_channels=in_channels,
            order=order,
            lam_max=lam_max,
            zeroDC=zeroDC,
        )
        self.N = self.adj.size(-1)  # osglm compatible

    def analyze(self, x):
        x = self._check_signal(x)
        if self.strategy == "osglm":
            x_append = x[:, self.append_nodes, :]
            x = torch.cat([x, x_append], 1)
        return self._analyze(x)

    def synthesize(self, y):
        z = self._synthesize(y)
        if self.strategy == "osglm":
            z = z[:, : self.N, :]
        return z


class NumQmf(QmfCore):
    def __init__(
        self,
        G: Graph,
        kernel=None,
        in_channels=1,
        order=24,
        strategy: str = "admm",
        M=1,
        lam_max=2.0,
        zeroDC=False,
        **kwargs,
    ):
        self.adj = G
        N = self.adj.size(-1)
        self.strategy = strategy
        self.M = M

        device = self.adj.device()
        dtype = self.adj.dtype()
        if strategy == "admm":
            if N < 80:
                bptG_dense = admm_bga(
                    self.adj.to_dense().to(torch.double), M=M, **kwargs
                )
                beta = bptG_dense.new_zeros(N, M, dtype=bool)
                bptG_dense = bptG_dense.to(dtype).to(device)
                bptG = []
                for i, B in enumerate(bptG_dense):
                    _, vtx_color, _ = is_bipartite_fix(B, fix_flag=True)
                    beta[:, i] = torch.as_tensor(vtx_color)
                    bptG.append(SparseTensor.from_dense(B))

            else:
                bptG, beta, self.partptr, self.perm = admm_lbga_ray(
                    self.adj, M, **kwargs
                )
                bptG = [SparseTensor.from_scipy(B).to(dtype).to(device) for B in bptG]

        elif strategy == "amfs":
            bptG, beta = amfs(self.adj, level=self.M, **kwargs)
            bptG = [SparseTensor.from_scipy(B).to(dtype).to(device) for B in bptG]

        else:
            raise RuntimeError(
                f"{str(strategy)} is not a valid numerical decomposition algorithm "
                f"supported at present."
            )

        super(NumQmf, self).__init__(
            bptG,
            beta,
            analyze_kernels=kernel,
            in_channels=in_channels,
            order=order,
            lam_max=lam_max,
            zeroDC=zeroDC,
        )


class BiorthCore(QmfCore):
    def __init__(
        self, bptG, beta, k=8, in_channels=1, order=16, lam_max=2.0, zeroDC=False
    ):
        h0_c, g0_c, orthogonality = design_biorth_kernel(k)
        h0 = partial(polyval, torch.from_numpy(h0_c))
        h0.__name__ = "h0"
        g0 = partial(polyval, torch.from_numpy(g0_c))
        g0.__name__ = "g0"

        def h1(x):
            return g0(2 - x)

        def g1(x):
            return h0(2 - x)

        self.orthogonality = orthogonality
        super(BiorthCore, self).__init__(
            bptG,
            beta,
            analyze_kernels=(h0, h1),
            synthesis_kernels=(g0, g1),
            in_channels=in_channels,
            order=order,
            lam_max=lam_max,
            zeroDC=zeroDC,
        )

    def __repr__(self):
        info = super().__repr__()
        info = info[:-2] + ", orthogonality={}".format(self.orthogonality)
        return info + "\n)"


class ColorBiorth(BiorthCore):
    def __init__(
        self,
        G: Graph,
        k=8,
        in_channels=1,
        order=16,
        strategy="harary",
        vtx_color=None,
        lam_max=2.0,
        zeroDC=False,
        **kwargs,
    ):
        self.adj = G
        self.lam_max = lam_max
        self.strategy = strategy

        if strategy == "harary":
            bptG, beta, beta_dist, vtx_color, mapper = harary(
                self.adj, vtx_color=vtx_color, **kwargs
            )
        elif strategy == "osglm":
            bptG, beta, append_nodes, vtx_color = osglm(
                self.adj, vtx_color=vtx_color, **kwargs
            )
            self.append_nodes = append_nodes
        else:
            raise RuntimeError(
                f"{strategy} is not a valid color-based decomposition algorithm."
            )
        self.vtx_color = vtx_color

        bptG = [SparseTensor.from_scipy(B).to(G.device()) for B in bptG]

        super(ColorBiorth, self).__init__(
            bptG, beta, k, in_channels, order, lam_max, zeroDC
        )
        self.N = self.adj.size(-1)  # osglm compatible

    def analyze(self, x):
        x = self._check_signal(x)
        if self.strategy == "osglm":
            x_append = x[:, self.append_nodes, :]
            x = torch.cat([x, x_append], 1)
        return self._analyze(x)

    def synthesize(self, y):
        z = self._synthesize(y)
        if self.strategy == "osglm":
            z = z[:, : self.N, :]
        return z


class NumBiorth(BiorthCore):
    def __init__(
        self,
        G,
        k=8,
        in_channels=1,
        order=16,
        strategy="admm",
        M=1,
        lam_max=2.0,
        zeroDC=False,
        **kwargs,
    ):
        self.adj = G
        N = self.adj.size(-1)
        self.lam_max = lam_max
        self.strategy = strategy
        self.M = M

        device = self.adj.device()
        dtype = self.adj.dtype()
        if strategy == "admm":
            if N < 80:
                bptG_dense = admm_bga(
                    self.adj.to_dense().to(torch.double), M=M, **kwargs
                )
                beta = bptG_dense.new_zeros(N, M, dtype=bool)
                bptG = []
                bptG_dense = bptG_dense.to(dtype).to(device)
                for i, B in enumerate(bptG_dense):
                    _, vtx_color, _ = is_bipartite_fix(B, fix_flag=True)
                    beta[:, i] = torch.as_tensor(vtx_color)
                    bptG.append(SparseTensor.from_dense(B))

            else:
                bptG, beta, self.partptr, self.perm = admm_lbga_ray(
                    self.adj, M, **kwargs
                )
                bptG = [SparseTensor.from_scipy(B).to(dtype).to(device) for B in bptG]

        elif strategy == "amfs":
            bptG, beta = amfs(self.adj, level=self.M, **kwargs)
            bptG = [SparseTensor.from_scipy(B).to(dtype).to(device) for B in bptG]

        else:
            raise RuntimeError(
                f"{strategy} is not a valid numerical decomposition "
                f"algorithm supported at present."
            )

        super(NumBiorth, self).__init__(
            bptG, beta, k, in_channels, order, lam_max, zeroDC
        )
