import numpy as np
import torch

from thgsp.graphs.core import GraphBase

from .approximation import cheby_coeff, cheby_op
from .kernels import get_kernel_name, meyer_kernel


def check_signal(x, true_num_node, dtype=None):
    if x.dim() == 1:  # N -> 1 x N x 1
        x = x.reshape(1, -1, 1)
    elif x.dim() == 2:  # N x Ci -> 1 x N x Ci
        x = x.unsqueeze(0)
    elif x.dim() == 3:  # keep Co x N x Ci or 1 x N x Ci
        x = x
    else:
        raise RuntimeError(
            "rank-1,2,3 tensor expected, but got rank-{}".format(x.dim())
        )

    if x.shape[-2] != true_num_node:
        raise RuntimeError(
            f"The penultimate dimension of signal: {x.shape[-2]}"
            f"!= {true_num_node} (the number of nodes)"
        )
    return x.to(dtype)


class Filter:
    """A graph filter(bank) class.

    Parameters
    ----------
    G: GraphBase
       Any instance of :py:obj:`GraphBase` or its subclass.
    kernels: array, callable, None
        Case1: The :obj:`(Co,Ci)` array of :obj:`Co*Ci` filters; :obj:`Co` and :obj:`Ci`
        are the dimensions of input and output signals, respectively Case2: A callable
        python object; all :obj:`Co*Ci` filters employ this kernel. Case3: Set all
        :obj:`Co*Ci` filters as ideal low-pass ones.
    lam_max:    float
        The supremum of graph frequencies.

    Attributes
    ----------
    N: int
        The number of graph nodes.
    order: int
        The degree of Chebyshev approximation.
    weight:     Tensor
        Shape: :obj:`(Co,Ci)`.  The signal of every output channel is a weighted sum of
        :obj:`Ci` filtered signals. The :obj:`i,j`-th entry is the weight of filtered
        signal from :obj:`j`-th input channel to :obj:`i`-th output channel. All-one
        tensor in default.
    dtype:      torch.dtype
        The data type of signals and graph adjacency.
    device:     torch.device
        The device of graph.
    """

    def __init__(
        self,
        G: GraphBase,
        kernels=None,
        in_channels=None,
        out_channels=None,
        order=20,
        lam_max=None,
        lap_type="sym",
        weight=None,
    ):
        assert order > 1
        self.G = G
        self.order = order
        self.lam_max = G.max_frequency(lap_type) if lam_max is None else lam_max
        assert self.lam_max > 0

        self.kernels, self.in_channels, self.out_channels = self.check_kernels(
            kernels, in_channels, out_channels
        )
        self.Ci = self.in_channels
        self.Co = self.out_channels

        self.N = G.n_node
        self.dtype = G.dtype()
        self.device = G.device()
        self.lap_type = lap_type

        if weight is None:
            weight = torch.ones(self.Co, self.Ci, dtype=self.dtype, device=self.device)
        else:
            assert weight.shape == (self.Co, self.Ci)
        self.channel_aggr_weight = weight

        # for Chebyshev approximation
        self.max_cheby_order = order
        self._coeff = None

    @staticmethod
    def check_kernels(kernels=None, Ci=None, Co=None):
        if isinstance(kernels, np.ndarray):
            if Ci is None and Co is None:
                Co, Ci = kernels.shape
            else:
                assert kernels.shape == (Co, Ci)

        else:  # ideal low pass filter bank by default
            Ci = 1 if Ci is None else Ci
            Co = 1 if Co is None else Co
            single_krn = np.array([[meyer_kernel if kernels is None else kernels]])
            kernels = np.tile(single_krn, [Co, Ci])
        return kernels, Ci, Co

    def evaluate(self, low=None, high=None, in_channels=None, out_channels=None):
        if low is None:
            low = 0
        if high is None:
            high = self.lam_max
        assert low <= high
        fs = self.G.spectrum(self.lap_type)

        if in_channels is None:
            in_channels = range(self.in_channels)
        if out_channels is None:
            out_channels = range(self.out_channels)

        mask1 = low <= fs
        mask2 = fs <= (1 + 5e-4) * high
        ls2eval = fs[mask1 & mask2]
        if len(ls2eval) == 0:
            raise RuntimeError(
                "No frequency in the interval [ {}, {}]".format(low, high)
            )
        fre_response = torch.zeros(
            self.out_channels,
            self.in_channels,
            len(ls2eval),
            dtype=self.dtype,
            device=self.device,
        )
        kernel_cache = {}
        for i in in_channels:
            for j in out_channels:
                krn = self.kernels[j, i]
                kid = id(krn)
                if kid in kernel_cache:
                    fre_response[j, i] = kernel_cache[kid]
                else:
                    fre_response[j, i] = krn(ls2eval)
                    kernel_cache[kid] = fre_response[j, i]
        return fre_response

    def filter(self, x):
        x = check_signal(x, self.N, dtype=self.dtype)  # (Co,N,Ci)
        U = self.G.U()
        response = self.evaluate()  # (Co,Ci,N)
        gft_coeff = U.t() @ x  # (Co,N,Ci)
        # (Co, N, Ci) * (Co, Ci, N).permute(0, 2, 1) --> (Co, N, Ci)
        spectral_out = gft_coeff * response.permute(0, 2, 1)
        #  (N, N) @ (Co, N, Ci) --> (Co, N, Ci)
        spatial_out = U @ spectral_out
        return spatial_out

    @property
    def cheby_coefficients(self):
        if self._coeff is None:
            coeff = cheby_coeff(
                self.kernels[None, ...],
                K=self.order,
                lam_max=self.lam_max,
                dtype=self.dtype,
                device=self.device,
            )  # the first dim is pseudo
            self._coeff = coeff.squeeze_(0)
        return self._coeff

    def cheby_filter(self, x, order=None):
        if order is None:
            order = self.order
        if order > self.order:
            raise RuntimeError(
                f"The coefficients of Chebyshev polynomials beyond "
                f"order {self.order} are not computed"
            )
        x = check_signal(x, self.N, dtype=self.dtype)
        coeff = self.cheby_coefficients[:, :, : order + 1]  # Co x Ci x K+1
        out = cheby_op(x, self.G.L(self.lap_type), coeff, self.lam_max)  # Co x N X Ci
        return out

    def __call__(self, x, cheby=True):
        """Filter the input signal **x**.

        Parameters
        ----------
        x:  Tensor
            The signal to filter. Shape: :obj:`(N,)` or :obj:`(N,Ci)`.  :obj:`Ci` and
            :obj:`N` are the numbers of input channels and graph nodes, separately.
        cheby: bool
            If :py:obj:`True`, conduct filtering via Chebyshev approximation. Otherwise
            signals are filtered in a brute-force way - do a complete eigenvalue
            decomposition of Laplacian :math:`L` to get the GFT and IGFT matrices.

        Returns
        -------
        Tensor
            Filtered signals. Shape: :obj:`(N,Co)`. :obj:`Co` is the output channels.
        """
        out = self.cheby_filter(x) if cheby else self.filter(x)
        # (Co, N ,Ci) @ (Co, Ci, 1) = (Co, N, 1)
        out_aggr = out @ self.channel_aggr_weight[:, :, None]
        return out_aggr.permute(1, 0, 2).squeeze_()

    def __repr__(self):
        info = (
            f"{self.__class__.__name__}"
            f"(lap_type={self.lap_type}, in_channels={self.in_channels},"
            f" out_channels={self.out_channels}, N={self.N},\n"
            f"kernels:\n"
            f"{get_kernel_name(self.kernels, True)})"
        )

        return info
