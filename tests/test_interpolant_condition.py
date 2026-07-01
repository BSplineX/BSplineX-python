"""Tests for the InterpolantCondition wrapper class"""

from bsplinex import InterpolantCondition
from bsplinex._bsplinex_impl import InterpolantCondition as _InterpolantCondition


def test_creation():
    x_value: float = 1.0
    y_value: float = 2.0
    derivative_order: int = 1

    interpolant_condition = InterpolantCondition(x_value=x_value, y_value=y_value, derivative_order=derivative_order)
    _interpolant_condition = _InterpolantCondition(x_value=x_value, y_value=y_value, derivative_order=derivative_order)

    assert interpolant_condition._condition.x_value == _interpolant_condition.x_value
    assert interpolant_condition._condition.y_value == _interpolant_condition.y_value
    assert interpolant_condition._condition.derivative_order == _interpolant_condition.derivative_order
