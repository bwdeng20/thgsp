import pytest
import numpy as np
from thgsp.convert import get_array_module
from thgsp.sampling._utils import (
    construct_sampling_matrix,
    construct_hth,
    construct_dia,
)
from ..utils4t import sparse_formats, float_dtypes, devices


@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("layout", sparse_formats)
@pytest.mark.parametrize("dv", devices)
def test_construct_sampling_matrix(dtype, dv, layout):
    on_gpu = "cuda" == dv.type
    xp, xcipy, _ = get_array_module(on_gpu)
    S = [5, 4, 1]
    N = 6
    H = construct_sampling_matrix(N, S, dtype, dv, layout)
    assert H.format == layout
    assert H.shape == (len(S), N)

    D = xcipy.sparse.diags(xp.arange(1, N + 1)[S])
    HtHD0 = H.T @ D @ H
    HtH0 = H.T @ H

    print("\n", H.A)
    print(HtH0.A)
    print(HtHD0.A)
    HtH1 = construct_hth(N, S, dtype=dtype, device=dv, layout=layout)
    HtHD1 = construct_hth(
        N, S, xp.arange(1, N + 1)[S], dtype=dtype, device=dv, layout=layout
    )
    # print(HtH1.A)
    # print(HtHD1.A)
    print(type(HtH1), HtH1.shape, "on_gpu: ", on_gpu)
    print(type(HtHD1), HtHD1.shape, "on_gpu: ", on_gpu)
    assert np.allclose(HtH0.A, HtH1.A)
    assert np.allclose(HtHD0.A, HtHD1.A)


@pytest.mark.parametrize("dt", float_dtypes)
@pytest.mark.parametrize("layout", sparse_formats)
@pytest.mark.parametrize("dv", devices)
@pytest.mark.parametrize("inverse", [True, False])
@pytest.mark.parametrize("narrow", [True, False])
def test_construct_dia(dt, dv, layout, inverse, narrow):
    on_gpu = "cuda" == dv.type
    xp, xcipy, _ = get_array_module(on_gpu)

    N = 7
    S = [5, 1, 3]
    M = len(S)
    diag_data = xp.arange(N)
    P = construct_dia(
        S, diag_data=diag_data, ps=narrow, inverse=inverse, dtype=dt, layout=layout
    )
    if narrow:
        assert P.shape == (M, M)
        if inverse:
            assert xp.allclose(
                P.diagonal(),
                1 / diag_data[S],
            )
        else:
            assert xp.allclose(P.diagonal(), diag_data[S])

    else:
        assert P.shape == (N, N)
        if inverse:
            assert xp.allclose(P.diagonal(), 1 / diag_data)
        else:
            assert xp.allclose(P.diagonal(), diag_data)
