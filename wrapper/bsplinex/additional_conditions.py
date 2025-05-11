# NOTE: revert back to `.`
from bsplinex import _bsplinex_impl as _impl

import numpy as np
import numpy.typing as npt


class AdditionalConditions:
    """The AdditionalConditions class"""

    _additional_conditions: list[_impl.InterpolantCondition]

    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, derivative_order: npt.ArrayLike) -> None:
        """construct a new AdditionalConditions object

        :param x: x-values at which the additional conditions must hold
        :param y: y-values at which the b-spline (derivative) must pass
        :param derivative_order: derivative order
        """

        x = np.asarray(x)
        y = np.asarray(y)
        derivative_order = np.asarray(derivative_order)

        if derivative_order.dtype is not np.integer:
            raise ValueError("`derivative_order` must be convertible to integer type")

        if not all(len(arr.shape) == 1 for arr in (x, y, derivative_order)):
            raise ValueError("all inputs must be 1D arrays")

        if not all(arr.shape == x.shape for arr in (x, y, derivative_order)):
            raise ValueError("all inputs must be have matching dimensions")

        self._additional_conditions = [
            _impl.InterpolantCondition(x_value=_x, y_value=_y, derivative_order=_order)
            for _x, _y, _order in zip(x, y, derivative_order)
        ]
