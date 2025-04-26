from . import _bsplinex_impl as _impl
from typing import overload
import numpy as np


class OpenUniform:
    def __init__(self, degree: int, knots: np.ndarray | list[float]) -> None:
        self._impl = _impl.OpenUniform(degree, knots)

    @overload
    def evaluate(self, x: float, derivative_order: int = 0) -> float: ...
    @overload
    def evaluate(self, x: np.ndarray, derivative_order: int = 0) -> np.ndarray: ...

    def evaluate(self, x: np.ndarray | float, derivative_order: int = 0) -> np.ndarray | float:
        return self._impl.evaluate(x, derivative_order)
