import importlib
import os.path as osp

import torch

__version__ = '0.1.0'

cpp_tools = ['_version', '_dsatur']
for tool in cpp_tools:
    torch.ops.load_library(importlib.machinery.PathFinder().
                           find_spec(tool, [osp.dirname(__file__)]).origin)

from .convert import to_torch_sparse, to_scipy, to_np  # noqa
from .io import loadmat  # noqa

import thgsp.graphs  # noqa
import thgsp.alg  # noqa
import thgsp.filters  # noqa
import thgsp.bga  # noqa
import thgsp.utils  # noqa
import thgsp.datasets  # noqa
import thgsp.sampling  # noqa

__all__ = ['to_torch_sparse',
           'to_scipy',
           'to_np',
           'loadmat']
