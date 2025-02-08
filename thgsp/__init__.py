import importlib
import os.path as osp

import torch

import thgsp.alg  # noqa
import thgsp.bga  # noqa
import thgsp.datasets  # noqa
import thgsp.filters  # noqa
import thgsp.sampling  # noqa
import thgsp.utils  # noqa
from thgsp.graphs import (  # noqa
    DiGraph,
    Graph,
    GraphBase,
    knn,
    radius,
    rand_bipartite,
    rand_dg,
    rand_udg,
    random_bgraph,
    random_graph,
)

from .convert import (
    from_cpx,
    get_array_module,
    get_ddd,
    to_cpx,
    to_np,
    to_scipy,
    to_torch_sparse,
    to_xcipy,
    to_xp,
)
from .io import loadmat  # noqa

__version__ = "0.1.0"
suffix = "cuda" if torch.cuda.is_available() else "cpu"

cpp_tools = ["_version", "_dsatur", "_bsgda"]
for library in cpp_tools:
    cuda_spec = importlib.machinery.PathFinder().find_spec(
        f"{library}_cuda", [osp.dirname(__file__)]
    )
    cpu_spec = importlib.machinery.PathFinder().find_spec(
        f"{library}_cpu", [osp.dirname(__file__)]
    )
    spec = cuda_spec or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:  # pragma: no cover
        raise ImportError(
            f"Could not find module '{library}_cpu' in " f"{osp.dirname(__file__)}"
        )

cuda_version = torch.ops.torch_sparse.cuda_version()
if torch.version.cuda is not None and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split(".")]

    if t_major != major:
        raise RuntimeError(
            f"Detected that PyTorch and torch_sparse were compiled with "
            f"different CUDA versions. PyTorch has CUDA version "
            f"{t_major}.{t_minor} and torch_sparse has CUDA version "
            f"{major}.{minor}. Please reinstall the torch_sparse that "
            f"matches your PyTorch install."
        )

__all__ = [
    "to_torch_sparse",
    "to_scipy",
    "to_np",
    "to_cpx",
    "from_cpx",
    "to_xcipy",
    "to_xp",
    "loadmat",
    "get_ddd",
    "get_array_module",
]
