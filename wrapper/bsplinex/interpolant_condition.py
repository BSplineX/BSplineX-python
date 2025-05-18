from __future__ import annotations

from . import _bsplinex_impl as _impl


class InterpolantCondition:
    """The InterpolantCondition class"""

    _condition: _impl.InterpolantCondition

    def __init__(self, x_value: float, y_value: float, derivative_order: int) -> None:
        """Create a new InterpolantCondition

        :param x_value: x-value at which the additional conditions must hold
        :param y_value: y-value at which the b-spline (derivative) must pass
        :param derivative_order: derivative order at which this condition must hold
        """
        self._condition = _impl.InterpolantCondition(
            x_value=x_value,
            y_value=y_value,
            derivative_order=derivative_order,
        )
