from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy.typing as npt
from scipy.sparse import spmatrix
from torch import Tensor
from torch_sparse import SparseTensor

ArrayLike = npt.ArrayLike
VertexColor = Union[Tensor, ArrayLike, List[int], Tuple[int]]
KernelType = Union[Tuple[Callable], List[Callable], ArrayLike]
