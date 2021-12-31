import pytest
import torch
from torch_sparse import SparseTensor

from thgsp.graphs.laplace import laplace

from ..utils4t import devices, lap_types


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("lap_type", lap_types)
def test_laplace(lap_type, device):
    N = 6
    A = torch.rand(N, N)
    A = A + A.t()
    A.fill_diagonal_(0)
    spA = SparseTensor.from_dense(A)
    L = laplace(spA, lap_type).to_dense()
    print("\n:", L)
