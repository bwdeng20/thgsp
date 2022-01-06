import sys

import pytest
import ray

from thgsp.bga.admm import admm_bga, admm_lbga_ray, is_bipartite_fix
from thgsp.graphs.generators import rand_udg

from ..utils4t import devices, float_dtypes, partition_strategy


@pytest.mark.parametrize("dtype", float_dtypes[1:])
@pytest.mark.parametrize("M", [1, 2])
@pytest.mark.parametrize("density", [0.1, 0.2])
def test_admm_bga(density, M, dtype):
    N = 32
    G = rand_udg(N, density, dtype)
    bptGs = admm_bga(G.to_dense(), M=M)
    for i in range(
        M
    ):  # use is_bipartite_fix() is recommended to ensure the bipartiteness
        assert is_bipartite_fix(bptGs[i], fix_flag=True)[0]

    for i in range(M):  # Check the bipartiteness
        assert is_bipartite_fix(bptGs[i], fix_flag=False)[0]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", float_dtypes[1:])
@pytest.mark.parametrize("density", [0.05])
@pytest.mark.parametrize("style", [1, 2])
@pytest.mark.parametrize("M", [1, 2])
@pytest.mark.parametrize("part", partition_strategy)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Skip ADMM-LBGA on Windows due to consistently Ray crashes",
)
class TestAdmmLbga:
    def test_admm_lbga_ray(self, density, style, M, dtype, device, part):
        # https://docs.ray.io/en/latest/auto_examples/testing-tips.html Tip1 & 2
        if not ray.is_initialized:
            ray.init(num_cpus=2)
        N = 32 * 3
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

        for i, bptG in enumerate(bptGs):
            assert is_bipartite_fix(bptG)[0]
            print("{}-th subgraph, weights: {}".format(i, bptG.sum()))

        if ray.is_initialized:
            ray.shutdown()
