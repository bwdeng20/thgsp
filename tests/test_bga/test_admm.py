import pytest
import ray
from thgsp.graphs.generators import rand_udg
from ..utils4t import float_dtypes, devices, partition_strategy
from thgsp.bga.admm import admm_bga, is_bipartite_fix, admm_lbga_ray


@pytest.mark.parametrize('dtype', float_dtypes[1:])
@pytest.mark.parametrize('M', [1, 2])
@pytest.mark.parametrize('density', [0.1, 0.2])
def test_admm_bga(density, M, dtype):
    N = 32
    G = rand_udg(N, density, dtype)
    bptGs = admm_bga(G.to_dense(), M=M)
    for i in range(M):
        assert is_bipartite_fix(bptGs[i])[0]


@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dtype', float_dtypes[1:])
@pytest.mark.parametrize('density', [0.05])
@pytest.mark.parametrize('style', [1, 2])
@pytest.mark.parametrize('M', [1, 2])
@pytest.mark.parametrize('part', partition_strategy)
class TestAdmmLbga:
    def test_admm_lbga_ray(self, density, style, M, dtype, device, part):
        N = 32 * 5
        G = rand_udg(N, density, dtype=dtype, device=device)
        bptGs, beta, partptr, perm = admm_lbga_ray(
            G, M, block_size=32, style=style, part=part)
        print(f"\n---num_node: {N}, density: {density}, strategy: {style}, M: {M},"
              f" dtype:{dtype}, device:{device}, part:{part}")
        print("total weights: {}".format(G.sum().item()))
        for i, bptG in enumerate(bptGs):
            assert is_bipartite_fix(bptG)[0]
            print("{}-th subgraph, weights: {}".format(i, bptG.sum()))
        ray.shutdown()
