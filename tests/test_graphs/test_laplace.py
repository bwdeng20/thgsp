import pytest
import torch
from torch_sparse import SparseTensor

from thgsp.graphs.laplace import laplace

lap_types = ["sym", "comb", "rw"]


@pytest.mark.parametrize("lap_type", lap_types)
def test_laplace(lap_type):
    N = 6
    A = torch.rand(N, N)
    A = A + A.t()
    A.fill_diagonal_(0)
    spA = SparseTensor.from_dense(A)
    L = laplace(spA, lap_type).to_dense()
    print("\n:", L)
