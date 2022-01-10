import pytest
import ray
import torch

from thgsp.bga.admm import admm_bga, admm_lbga_ray, is_bipartite_fix
from thgsp.graphs.generators import rand_udg

from ..utils4t import (
    RAY_NUM_CPUS,
    RAY_NUM_GPUS,
    devices,
    float_dtypes,
    partition_strategy,
)

ray.init(
    num_cpus=RAY_NUM_CPUS,
    num_gpus=RAY_NUM_GPUS,
    ignore_reinit_error=True,
    log_to_driver=False,
)


@pytest.mark.parametrize("dtype", float_dtypes[1:])
@pytest.mark.parametrize("M", [1, 2])
@pytest.mark.parametrize("density", [0.1, 0.2])
def test_admm_bga(density, M, dtype):
    N = 32
    G = rand_udg(N, density, dtype)
    bptGs = admm_bga(G.to_dense(), M=M)
    for i in range(M):
        # use is_bipartite_fix() is recommended to ensure the bipartiteness
        assert is_bipartite_fix(bptGs[i], fix_flag=True)[0]

    for i in range(M):  # Check the bipartiteness
        assert is_bipartite_fix(bptGs[i], fix_flag=False)[0]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", float_dtypes[1:])
@pytest.mark.parametrize("density", [0.05])
@pytest.mark.parametrize("style", [1, 2])
@pytest.mark.parametrize("M", [1, 2])
@pytest.mark.parametrize("part", partition_strategy)
class TestAdmmLbga:
    def test_admm_lbga_ray(self, density, style, M, dtype, device, part):
        ray.init(num_cpus=RAY_NUM_CPUS, ignore_reinit_error=True, log_to_driver=False)
        N = 32 * 3 + 10
        G = rand_udg(N, density, dtype=dtype, device=device)
        bptGs, beta, partptr, perm = admm_lbga_ray(
            G, M, block_size=32, style=style, part=part
        )
        print(
            f"\n---num_node: {N}, density: {density}, strategy: {style}, M: {M},"
            f" dtype:{dtype}, device:{device}, part:{part}"
        )
        print("total weights: {}".format(G.sum().item()))
        for i, bptG in enumerate(bptGs):
            assert is_bipartite_fix(bptG)[0]
            print("{}-th subgraph, weights: {}".format(i, bptG.sum()))


def test_admm_lbga_ray_single(
    density=0.1, style=1, M=2, dtype=torch.double, device="cpu", part="graclus"
):
    ray.init(num_cpus=RAY_NUM_CPUS, ignore_reinit_error=True, log_to_driver=False)
    N = 32 * 3
    G = rand_udg(N, density, dtype=dtype, device=device)
    bptGs, beta, partptr, perm = admm_lbga_ray(
        G, M, block_size=5, style=style, part=part
    )
    print(
        f"\n---num_node: {N}, density: {density}, strategy: {style}, M: {M},"
        f" dtype:{dtype}, device:{device}, part:{part}"
    )
    print("total weights: {}".format(G.sum().item()))
    for i, bptG in enumerate(bptGs):
        assert is_bipartite_fix(bptG)[0]
        print("{}-th subgraph, weights: {}".format(i, bptG.sum()))


if ray.is_initialized():
    ray.shutdown()
