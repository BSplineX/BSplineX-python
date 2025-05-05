# NOTE: revert back to `.`
from . import _bsplinex_impl as _impl

from typing import overload
from abc import ABC
from textwrap import dedent

import numpy as np
import numpy.typing as npt


class BSpline(ABC):
    def __init__(self, bspline: _impl.OpenUniform | _impl.OpenNonUniform | None) -> None:
        super().__init__()
        if bspline is not None:
            self._bspline = bspline

    @overload
    def evaluate(self, x: float, derivative_order: int = 0) -> float: ...
    @overload
    def evaluate(self, x: npt.NDArray[np.float64], derivative_order: int = 0) -> npt.NDArray[np.float64]: ...

    def evaluate(
        self, x: npt.NDArray[np.float64] | float, derivative_order: int = 0
    ) -> npt.NDArray[np.float64] | float:
        """evaluate the b-spline (derivative) at the given value(s)

        :param x: value(s) to evaluate the b-spline at
        :param derivative_order: order of the derivative, default 0
        :return: the value(s) of the b-spline
        """

        return self._bspline.evaluate(x, derivative_order)


class OpenUniform(BSpline):
    @overload
    def __init__(self, degree: int) -> None:
        """Create a uniform b-spline with open boundary condition

        > NOTE: useful constructor for interpolating points

        :param degree: degree of the b-spline
        :return: the open uniform b-spline
        """
        ...

    @overload
    def __init__(self, degree: int, begin: float, end: float, num_elems: int) -> None:
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
    def __init__(
        self, degree: int, begin: float, end: float, num_elems: int, ctrl_points: npt.NDArray[np.float64] | list[float]
    ) -> None:
        """Create a uniform b-spline with open boundary condition

        :param degree: degree of the b-spline
        :param begin: starting knot
        :param end: end knot
        :param num_elems: number of knots
        :param ctrl_points: control points
        :return: the open uniform b-spline
        """
        ...

    def __init__(
        self,
        degree: int,
        begin: float | None = None,
        end: float | None = None,
        num_elems: int | None = None,
        ctrl_points: npt.NDArray[np.float64] | list[float] | None = None,
    ) -> None:
        """Create a uniform b-spline with open boundary condition

        This function has three possible overloads:

        - make_open_uniform(degree)
        - make_open_uniform(degree, begin, end, num_elems)
        - make_open_uniform(degree, begin, end, num_elems, ctrl_points)

        Note that only `make_open_uniform(degree, begin, end, num_elems, ctrl_points)`
        generates a valid b-spline, the other overloads are meant to
        be used for fitting (`make_open_uniform(degree, begin, end, num_elems)`)
        and interpolation (`make_open_uniform(degree))`

        :param degree: degree of the b-spline
        :param begin: starting knot
        :param end: end knot
        :param num_elems: number of knots
        :param ctrl_points: control points
        :return: the open uniform b-spline
        """

        if all((arg is not None for arg in (begin, end, num_elems, ctrl_points))):
            super().__init__(_impl.make_open_uniform(degree, begin, end, num_elems, ctrl_points))
            return
        if all((arg is not None for arg in (begin, end, num_elems))) and ctrl_points is None:
            super().__init__(_impl.make_open_uniform(degree, begin, end, num_elems))
            return
        if all((arg is None for arg in (begin, end, num_elems, ctrl_points))):
            super().__init__(_impl.make_open_uniform(degree))
            return

        error = """
        This function has three possible overloads:
        - make_open_uniform(degree)
        - make_open_uniform(degree, begin, end, num_elems)
        - make_open_uniform(degree, begin, end, num_elems, ctrl_points)
        """

        raise ValueError(dedent(error))


class OpenNonUniform(BSpline):
    pass


@overload
def make_open_uniform(degree: int) -> OpenUniform: ...


@overload
def make_open_uniform(degree: int, begin: float, end: float, num_elems: int) -> OpenUniform: ...


@overload
def make_open_uniform(
    degree: int, begin: float, end: float, num_elems: int, ctrl_points: npt.NDArray[np.float64] | list[float]
) -> OpenUniform: ...


def make_open_uniform(
    degree: int,
    begin: float | None = None,
    end: float | None = None,
    num_elems: int | None = None,
    ctrl_points: npt.NDArray[np.float64] | list[float] | None = None,
) -> OpenUniform:
    """Create a uniform b-spline with open boundary condition

    This function has three possible overloads:

    - make_open_uniform(degree)
    - make_open_uniform(degree, begin, end, num_elems)
    - make_open_uniform(degree, begin, end, num_elems, ctrl_points)

    Note that only `make_open_uniform(degree, begin, end, num_elems, ctrl_points)`
    generates a valid b-spline, the other overloads are meant to
    be used for fitting (`make_open_uniform(degree, begin, end, num_elems)`)
    and interpolation (`make_open_uniform(degree))`

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :param ctrl_points: control points
    :return: the open uniform b-spline
    """

    if all((arg is not None for arg in (begin, end, num_elems, ctrl_points))):
        return OpenUniform(_impl.make_open_uniform(degree, begin, end, num_elems, ctrl_points))
    if all((arg is not None for arg in (begin, end, num_elems))) and ctrl_points is None:
        return OpenUniform(_impl.make_open_uniform(degree, begin, end, num_elems))
    if all((arg is None for arg in (begin, end, num_elems, ctrl_points))):
        return OpenUniform(_impl.make_open_uniform(degree))

    error = """
    This function has three possible overloads:
    - make_open_uniform(degree)
    - make_open_uniform(degree, begin, end, num_elems)
    - make_open_uniform(degree, begin, end, num_elems, ctrl_points)
    """

    raise ValueError(dedent(error))


@overload
def make_open_nonuniform(degree: int) -> OpenNonUniform: ...


@overload
def make_open_nonuniform(
    degree: int,
    knots: npt.NDArray[np.float64] | list[float] | None = None,
) -> OpenNonUniform: ...


@overload
def make_open_nonuniform(
    degree: int,
    knots: npt.NDArray[np.float64] | list[float] | None = None,
    ctrl_points: npt.NDArray[np.float64] | list[float] | None = None,
) -> OpenNonUniform: ...


def make_open_nonuniform(
    degree: int,
    knots: npt.NDArray[np.float64] | list[float] | None = None,
    ctrl_points: npt.NDArray[np.float64] | list[float] | None = None,
) -> OpenNonUniform:
    """Create a nonuniform b-spline with open boundary condition

    This function has three possible overloads:

    - make_open_nonuniform(degree)
    - make_open_nonuniform(degree, knots)
    - make_open_nonuniform(degree, knots, ctrl_points)

    Note that only `make_open_nonuniform(degree, knots, ctrl_points)`
    generates a valid b-spline, the other overloads are meant to
    be used for fitting (`make_open_uniform(degree, knots)`)
    and interpolation (`make_open_uniform(degree))`

    :param degree: degree of the b-spline
    :param knots: knots
    :param ctrl_points: control points
    :return: the open nonuniform b-spline
    """

    if all((arg is not None for arg in (knots, ctrl_points))):
        return OpenNonUniform(_impl.make_open_nonuniform(degree, knots, ctrl_points))
    if knots is not None and ctrl_points is None:
        return OpenNonUniform(_impl.make_open_nonuniform(degree, knots))
    if all((arg is None for arg in (knots, ctrl_points))):
        return OpenNonUniform(_impl.make_open_nonuniform(degree))

    error = """
    This function has three possible overloads:
    - make_open_nonuniform(degree)
    - make_open_nonuniform(degree, knots)
    - make_open_nonuniform(degree, knots, ctrl_points)
    """

    raise ValueError(dedent(error))
