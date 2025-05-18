from __future__ import annotations
from typing import Union

import numpy as np
import numpy.typing as npt

from . import _bsplinex_impl as _impl
from .interpolating_condition import InterpolantCondition


BSplineTypes = Union[
    _impl.OpenUniform,
    _impl.OpenNonUniform,
    _impl.OpenUniformConstant,
    _impl.OpenNonUniformConstant,
    _impl.ClampedUniform,
    _impl.ClampedNonUniform,
    _impl.ClampedUniformConstant,
    _impl.ClampedNonUniformConstant,
    _impl.PeriodicUniform,
    _impl.PeriodicNonUniform,
]


class BSpline:
    """The BSpline class"""

    _bspline: BSplineTypes

    def __init__(self, bspline: BSplineTypes) -> None:
        """Construct a new BSpline object

        > NOTE: this constructor is meant to be private, please use the factory methods like `make_open_uniform(...)`

        :param bspline: an internal BSpline type
        """

        self._bspline = bspline

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self._bspline == other._bspline
        else:
            return False

    def evaluate(self, x: npt.ArrayLike, derivative_order: int = 0) -> npt.NDArray[np.float64]:
        """Evaluate the b-spline (derivative) at the given value(s)

        :param x: values to evaluate the b-spline at
        :param derivative_order: order of the derivative
        :return: the values of the b-spline
        """

        return np.asarray(self._bspline.evaluate(x, derivative_order))

    def derivative(self, derivative_order: int = 1) -> "BSpline":
        """Get the derivative of this b-spline as a new BSpline

        :param derivative_order: order of the derivative
        :return: the derivative of this b-spline
        """

        return BSpline(self._bspline.derivative(derivative_order))

    def basis(self, value: float, derivative_order: int = 0) -> npt.NDArray[np.float64]:
        """Compute the basis (derivative) at `value`

        :param value: value to compute the basis at
        :param derivative_order: order of the derivative
        :return: the basis values
        """

        return np.asarray(self._bspline.basis(value, derivative_order))

    def nnz_basis(self, value: float, derivative_order: int = 0) -> npt.NDArray[np.float64]:
        """Compute (the derivative of) the non-zero basis at `value`

        :param value: value to compute the non-zero basis at
        :param derivative_order: order of the derivative
        :return: the non-zero basis values
        """

        return np.asarray(self._bspline.nnz_basis(value, derivative_order))

    def domain(self) -> tuple[float, float]:
        """Get the domain of the b-spline

        :return: the b-spline domain
        """

        return self._bspline.domain()

    def fit(self, x: npt.NDArray[np.float64] | list[float], y: npt.NDArray[np.float64] | list[float]) -> None:
        """Fit a b-spline to the given points

        :param x: x-values of the points
        :param y: y-values of the points
        """

        self._bspline.fit(x, y)

    def interpolate(
        self,
        x: npt.NDArray[np.float64] | list[float],
        y: npt.NDArray[np.float64] | list[float],
        additional_conditions: list[InterpolantCondition],
    ) -> None:
        """Compute the b-spline that interpolates the given points and respects the given additional conditions

        :param x: x-values of the points
        :param y: y-values of the points
        :param additional_conditions: list of additional interpolating conditions
        """

        self._bspline.interpolate(x, y, [condition._condition for condition in additional_conditions])

    def get_control_points(self) -> npt.NDArray[np.float64]:
        """Get the control points of the b-spline

        :return: the control points
        """

        return np.asarray(self._bspline.get_control_points())

    def get_knots(self) -> npt.NDArray[np.float64]:
        """Get the knots of the b-spline

        :return: the knots
        """

        return np.asarray(self._bspline.get_knots())

    def get_degree(self) -> float:
        """Get the degree of the b-spline

        :return: the degree
        """

        return self._bspline.get_degree()
