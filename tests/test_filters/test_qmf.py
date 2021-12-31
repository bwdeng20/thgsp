import numpy as np
import pytest
import torch
from torch_sparse import SparseTensor

from thgsp.filters.qmf import (
    BiorthCore,
    BiorthOperator,
    ColorBiorth,
    ColorQmf,
    NumBiorth,
    NumQmf,
    QmfCore,
    QmfOperator,
    harary,
    meyer_kernel,
    meyer_mirror_kernel,
)
from thgsp.graphs.generators import rand_bipartite, rand_udg
from thgsp.utils.metrics import snr

from ..utils4t import (
    color_strategies,
    devices,
    float_dtypes,
    num_strategies,
    partition_strategy,
)


def ppprint(dis, pf):
    print("reconstruction SNR:  ", pf, " dB")
    print(
        "|max distortion among all nodes:  ",
        dis.max(),
        " at {}-th node".format(dis.argmax()),
    )
    print(
        "min distortion among all nodes:  ",
        dis.min(),
        " at {}-th node".format(dis.argmin()),
    )
    print("distortion sum                :  ", dis.sum())
    print("|-----------------------------------------------------------------")


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("device", devices)
class TestQmfCore:
    def test_init(self, device, dtype):
        M, N = 2, 7
        K = 10
        Bs = [rand_udg(N, dtype=dtype, device=device) for _ in range(M)]
        beta_np = np.random.rand(N, M) > 0.5

        qmf = QmfCore(bptG=Bs, beta=beta_np, order=K)
        assert qmf.coefficient_a.shape == (M, 2 ** M, 1, K + 1)
        assert qmf.coefficient_a.shape == qmf.coefficient_s.shape

        QmfCore(
            Bs,
            torch.as_tensor(beta_np),
            analyze_kernels=(meyer_kernel, meyer_mirror_kernel),
        )

        qmf = QmfCore(Bs, beta_np, in_channels=3)
        # 20 is the default order of approximation
        assert qmf.coefficient_a.shape == (M, 2 ** M, 3, qmf.order + 1)

    def test_transform(self, device, dtype):
        M, N = 2, 32
        K = 30
        Bs = []
        beta = []

        bpg, bt = rand_bipartite(
            N // 2, N - N // 2, p=0.2, dtype=dtype, device=device, return_partition=True
        )
        Bs.append(bpg)
        beta.append(bt)
        mask = Bs[0].to_dense() != 0
        for i in range(M - 1):
            bpg, bt = rand_bipartite(
                N // 2,
                N - N // 2,
                p=0.2,
                dtype=dtype,
                device=device,
                return_partition=True,
            )
            dense = bpg.to_dense()
            temp_mask = dense != 0

            dense[mask] = 0
            mask = temp_mask + mask
            Bs.append(SparseTensor.from_dense(dense))
            if i % 2 == 0:
                beta.append(bt)
            else:
                beta.append(~bt)

        beta = torch.stack(beta).T

        qmf = QmfCore(bptG=Bs, beta=beta, order=K)
        x = torch.rand(N, dtype=dtype, device=device)
        y = qmf.analyze(x)
        assert y.shape == (2 ** M, N, 1)

        z = qmf.synthesize(y)
        assert z.shape == (2 ** M, N, 1)
        # since beta and Bs are all randomly generated,
        # the transform are not numerically valid
        assert (z.sum(0).squeeze() - x).abs().sum() != 0

        z.squeeze_()
        f_hat = z.sum(0)
        dis = (f_hat - x).abs()
        pf = snr(f_hat, x)
        ppprint(dis, pf)


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("strategy", color_strategies)
class TestColorQmf:
    def test_init(self, dtype, device, strategy):
        N = 10
        graph = rand_udg(N, dtype=dtype, device=device)
        ColorQmf(graph)
        ColorQmf(graph)
        ColorQmf(graph, in_channels=3, zeroDC=True, strategy=strategy)

    @pytest.mark.parametrize("Ci", [5])
    def test_transform(self, dtype, device, strategy, Ci):
        N = 60
        graph = rand_udg(N, device=device, dtype=dtype)
        qmf = ColorQmf(graph, strategy=strategy, in_channels=1)
        M = qmf.M
        f = torch.rand(N, Ci, device=device, dtype=dtype)
        y = qmf.analyze(f)
        z = qmf.synthesize(y)
        z.squeeze_()
        f_hat = z.sum(0)
        dis = (f_hat - f).abs()
        pf = snr(f_hat, f.squeeze_()).item()
        print(
            "\n----- Strategy: {:8s}, M={:4d}  Device: {:5s}, Dtype: {:6s} ---".format(
                strategy, M, str(device), str(dtype)
            )
        )
        ppprint(dis, pf)
        assert pf > 20


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("strategy", num_strategies)
class TestNumQMf:
    @pytest.mark.parametrize("N", [20, 100])
    def test_init(self, dtype, device, strategy, N):
        graph = rand_udg(N, dtype=dtype, device=device)
        NumQmf(graph)
        NumQmf(graph, in_channels=3, zeroDC=True, strategy=strategy)

    @pytest.mark.parametrize("M", [1, 2])
    def test_transform(self, dtype, device, strategy, M):
        N = 32 * 3
        graph = rand_udg(N, device=device, dtype=dtype)
        qmf = NumQmf(graph, strategy=strategy, M=M)
        f = torch.rand(N, device=device, dtype=dtype)
        y = qmf.analyze(f)
        z = qmf.synthesize(y)
        z.squeeze_()
        f_hat = z.sum(0)
        dis = (f_hat - f).abs()
        pf = snr(f_hat, f).item()
        print(
            "\n|----- Strategy: {:8s}, M={}, Device: {:5s}, Dtype: {:6s} -----".format(
                strategy, M, str(device), str(dtype)
            )
        )
        ppprint(dis, pf)
        assert pf > 20


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("device", devices)
class TestBiorthCore:
    def test_init(self, dtype, device):
        M, N = 3, 7
        K = 12
        k = 3
        Bs = [rand_udg(N, dtype=dtype, device=device) for _ in range(M)]
        beta_np = np.random.rand(N, M) > 0.3

        bio = BiorthCore(Bs, beta_np, k=k, order=K)
        assert bio.coefficient_a.shape == (M, 2 ** M, 1, K + 1)
        assert bio.coefficient_a.shape == bio.coefficient_s.shape

        bio = BiorthCore(Bs, torch.as_tensor(beta_np), in_channels=3, zeroDC=True)
        # 16 is the default order of approximation
        assert bio.coefficient_a.shape == (M, 2 ** M, 3, 16 + 1)

    def test_transform(self, device, dtype):
        M, N = 2, 7
        Bs = [rand_udg(N, dtype=dtype, device=device) for _ in range(M)]
        beta = torch.rand(N, M, device=device) > 0.5

        bio = BiorthCore(Bs, beta)
        x = torch.rand(N, dtype=bio.dtype, device=bio.device)
        y = bio.analyze(x)
        assert y.shape == (2 ** M, N, 1)

        z = bio.synthesize(y)
        assert z.shape == (2 ** M, N, 1)
        # since beta and Bs are all randomly generated, the transform won't be
        # numerically valid
        assert (z.sum(0).squeeze() - x).abs().sum() != 0


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("strategy", color_strategies)
class TestColorBiorth:
    def test_init(self, dtype, device, strategy):
        N = 12
        graph = rand_udg(N, dtype=dtype, device=device)
        ColorBiorth(graph)
        ColorBiorth(graph)
        ColorBiorth(graph, in_channels=3, strategy=strategy, zeroDC=True)

    def test_transform(self, dtype, device, strategy):
        N = 100
        graph = rand_udg(N, device=device, dtype=dtype)
        bio = ColorBiorth(graph, strategy=strategy)
        M = bio.M

        f = torch.rand(N, device=bio.device, dtype=bio.dtype)
        y = bio.analyze(f)
        z = bio.synthesize(y)
        z.squeeze_()
        f_hat = z.sum(0)
        dis = (f_hat - f).abs()
        pf = snr(f_hat, f).item()
        print(
            "\n|----- Strategy: {:8s}, M:{:4d}  Device: {:5s}, Dtype: {:6s} "
            "-----------".format(strategy, M, str(device), str(dtype))
        )
        ppprint(dis, pf)
        assert pf > 20


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("strategy", num_strategies)
class TestNumBiorth:
    @pytest.mark.parametrize("N", [20, 100])
    def test_init(self, dtype, device, strategy, N):
        graph = rand_udg(N, dtype=dtype, device=device)
        NumBiorth(graph)
        NumBiorth(graph, in_channels=3, zeroDC=True, strategy=strategy)

    @pytest.mark.parametrize("M", [1, 2])
    @pytest.mark.parametrize("part", partition_strategy)
    def test_transform(self, dtype, device, strategy, M, part):
        print(
            "\n|------ Strategy: {:8s}, M={}, Device: {:5s}, Dtype: {:6s} ----".format(
                strategy, M, str(device), str(dtype)
            )
        )
        kwargs = dict()
        if strategy == "admm":
            print("|admm-lbga part strategy: {}".format(part))
            kwargs["part"] = part
        N = 32 * 3
        graph = rand_udg(N, device=device, dtype=dtype)
        bio = NumBiorth(graph, strategy=strategy, M=M, **kwargs)
        f = torch.randn(N, device=device, dtype=dtype)
        y = bio.analyze(f)
        z = bio.synthesize(y)
        z.squeeze_()
        f_hat = z.sum(0)
        dis = (f_hat - f).abs()
        pf = snr(f_hat, f).item()
        ppprint(dis, pf)
        assert pf > 20


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("device", devices)
class TestQmfWaveletBasis:
    def test_one_level(self, dtype, device):
        N1 = 6
        N2 = 4
        bptG, beta = rand_bipartite(N1, N2, 0.5, dtype=dtype, return_partition=True)
        basis = QmfOperator(
            [bptG.to_scipy("csr")], beta.view(-1, 1), order=20, device=device
        )

        x = torch.ones(N1 + N2, 1, dtype=dtype, device=device)
        y = basis.transform(x)
        z = basis.inverse_transform(y)
        print("\nsnr: ", snr(z.permute(-1, -2), x.permute(-1, -2)).item(), "dB.")
        print("dis: ", (z - x).abs().sum())

    def test_multi_level(self, dtype, device):
        N = 16
        G = rand_udg(N, 0.2, dtype, device)
        bptG, beta, _, _, _ = harary(G)

        basis = QmfOperator(bptG, beta, order=4, device=device)
        x = torch.ones(N, 1, dtype=dtype, device=device)
        y = basis.transform(x)
        y = basis(x)
        z = basis.inverse_transform(y)
        print("\nsnr: ", snr(z.permute(-1, -2), x.permute(-1, -2)).item(), "dB.")
        print("dis: ", (z - x).abs().sum())


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("device", devices)
class TestBiorWaveletBasis:
    def test_one_level(self, dtype, device):
        N1 = 60
        N2 = 40
        bptG, beta = rand_bipartite(N1, N2, 0.2, dtype=dtype, return_partition=True)
        basis = BiorthOperator(
            [bptG.to_scipy("csr")], beta.view(-1, 1), k=2, device=device
        )

        x = torch.ones(N1 + N2, 1, dtype=dtype, device=device)
        y = basis.transform(x)
        z = basis.inverse_transform(y)
        print("\nsnr: ", snr(z.permute(-1, -2), x.permute(-1, -2)).item(), "dB.")
        print("dis: ", (z - x).abs().sum())
        self.display_density(basis.operator, basis.inv_operator)

    def test_multi_level(self, dtype, device):
        N = 120
        G = rand_udg(N, 0.4, dtype, device)
        bptG, beta, _, _, _ = harary(G)

        basis = BiorthOperator(bptG, beta, k=2, device=device)
        x = torch.ones(N, 1, dtype=dtype, device=device)
        y = basis.transform(x)
        z = basis.inverse_transform(y)
        print("\nsnr: ", snr(z.permute(-1, -2), x.permute(-1, -2)).item(), "dB.")
        print("dis: ", (z - x).abs().sum())
        self.display_density(basis.operator, basis.inv_operator)

    def display_density(self, op, inv_op):
        print("Ta       density: ", op.density())
        print("Ta^-1    density: ", inv_op.density())
