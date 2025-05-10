# NOTE: revert back to `.`
from . import _bsplinex_impl as _impl

from typing import overload
from textwrap import dedent

import numpy as np
import numpy.typing as npt


class BSpline:
    _bspline: _impl.OpenUniform | _impl.OpenNonUniform

    def __init__(self, bspline: _impl.OpenUniform | _impl.OpenNonUniform) -> None:
        self._bspline = bspline

    def evaluate(self, x: npt.ArrayLike, derivative_order: int = 0) -> npt.NDArray[np.float64]:
        """evaluate the b-spline (derivative) at the given value(s)

        :param x: values to evaluate the b-spline at
        :param derivative_order: order of the derivative
        :return: the values of the b-spline evalutated for each x
        """

        return np.asarray(self._bspline.evaluate(x, derivative_order))


@overload
def make_open_uniform(degree: int) -> BSpline:
    """Create a uniform b-spline with open boundary condition

    > NOTE: useful constructor for interpolating points

    :param degree: degree of the b-spline
    :return: the open uniform b-spline
    """
    ...


@overload
def make_open_uniform(degree: int, begin: float, end: float, num_elems: int) -> BSpline:
    """Create a uniform b-spline with open boundary condition

    > NOTE: useful constructor for fitting points

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :return: the open uniform b-spline
    """
    ...


@overload
def make_open_uniform(
    degree: int, begin: float, end: float, num_elems: int, ctrl_points: npt.NDArray[np.float64] | list[float]
) -> BSpline:
    """Create a uniform b-spline with open boundary condition

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :param ctrl_points: control points
    :return: the open uniform b-spline
    """
    ...


def make_open_uniform(
    degree: int,
    begin: float | None = None,
    end: float | None = None,
    num_elems: int | None = None,
    ctrl_points: npt.NDArray[np.float64] | list[float] | None = None,
) -> BSpline:
    if all((arg is not None for arg in (begin, end, num_elems, ctrl_points))):
        return BSpline(_impl.make_open_uniform(degree, begin, end, num_elems, ctrl_points))
    if all((arg is not None for arg in (begin, end, num_elems))) and ctrl_points is None:
        return BSpline(_impl.make_open_uniform(degree, begin, end, num_elems))
    if all((arg is None for arg in (begin, end, num_elems, ctrl_points))):
        return BSpline(_impl.make_open_uniform(degree))

    error = """
    This function has three possible overloads:
    - make_open_uniform(degree)
    - make_open_uniform(degree, begin, end, num_elems)
    - make_open_uniform(degree, begin, end, num_elems, ctrl_points)
    """

    raise ValueError(dedent(error))
