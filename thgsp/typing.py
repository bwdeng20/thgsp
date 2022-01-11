from typing import Callable, List, Tuple, Union

import numpy.typing as npt
from torch import Tensor

ArrayLike = npt.ArrayLike
VertexColor = Union[Tensor, ArrayLike, List[int], Tuple[int]]
KernelType = Union[Tuple[Callable], List[Callable], ArrayLike]
