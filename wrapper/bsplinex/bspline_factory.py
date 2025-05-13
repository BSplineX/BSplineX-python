# NOTE: revert back to `.`
from bsplinex import _bsplinex_impl as _impl
from .bspline import BSpline

from typing import overload
from textwrap import dedent

import numpy.typing as npt


@overload
def make_open_uniform(degree: int) -> BSpline:
    """Create a uniform b-spline with open boundary condition

    > NOTE: useful for interpolating points
    >>> bspline = make_open_uniform(3)
    >>> bspline.interpolate(x, y, additional_conditions)

    :param degree: degree of the b-spline
    :return: the open uniform b-spline
    """
    ...


@overload
def make_open_uniform(degree: int, begin: float, end: float, num_elems: int) -> BSpline:
    """Create a uniform b-spline with open boundary condition

    > NOTE: useful for fitting points
    >>> bspline = make_open_uniform(3, 0.0, 10.0, 10)
    >>> bspline.fit(x, y)


    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :return: the open uniform b-spline
    """
    ...


@overload
def make_open_uniform(degree: int, begin: float, end: float, num_elems: int, ctrl_points: npt.ArrayLike) -> BSpline:
    """Create a uniform b-spline with open boundary condition

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :param ctrl_points: control points of the b-spline
    :return: the open uniform b-spline
    """
    ...


def make_open_uniform(
    degree: int,
    begin: float | None = None,
    end: float | None = None,
    num_elems: int | None = None,
    ctrl_points: npt.ArrayLike | None = None,
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


@overload
def make_open_nonuniform(degree: int) -> BSpline:
    """Create a non-uniform b-spline with open boundary condition

    > NOTE: useful for interpolating points
    >>> bspline = make_open_nonuniform(3)
    >>> bspline.interpolate(x, y, additional_conditions)

    :param degree: degree of the b-spline
    :return: the open non-uniform b-spline
    """
    ...


@overload
def make_open_nonuniform(degree: int, knots: npt.ArrayLike) -> BSpline:
    """Create a non-uniform b-spline with open boundary condition

    > NOTE: useful for fitting points
    >>> bspline = make_open_nonuniform(3, knots)
    >>> bspline.fit(x, y)

    :param degree: degree of the b-spline
    :param knots: knots of the b-spline
    :return: the open non-uniform b-spline
    """
    ...


@overload
def make_open_nonuniform(degree: int, knots: npt.ArrayLike, ctrl_points: npt.ArrayLike) -> BSpline:
    """Create a non-uniform b-spline with open boundary condition

    :param degree: degree of the b-spline
    :param knots: knots of the b-spline
    :param ctrl_points: control points of the b-spline
    :return: the open non-uniform b-spline
    """
    ...


def make_open_nonuniform(
    degree: int, knots: npt.ArrayLike | None = None, ctrl_points: npt.ArrayLike | None = None
) -> BSpline:
    if all((arg is not None for arg in (knots, ctrl_points))):
        return BSpline(_impl.make_open_nonuniform(degree, knots, ctrl_points))
    if knots is not None and ctrl_points is None:
        return BSpline(_impl.make_open_nonuniform(degree, knots))
    if all((arg is None for arg in (knots, ctrl_points))):
        return BSpline(_impl.make_open_nonuniform(degree))

    error = """
    This function has three possible overloads:
    - make_open_nonuniform(degree)
    - make_open_nonuniform(degree, knots)
    - make_open_nonuniform(degree, knots, ctrl_points)
    """

    raise ValueError(dedent(error))


@overload
def make_open_uniform_constant(degree: int) -> BSpline:
    """Create a uniform b-spline with open boundary condition and constant extrapolation

    > NOTE: useful for interpolating points
    >>> bspline = make_open_uniform_constant(3)
    >>> bspline.interpolate(x, y, additional_conditions)

    :param degree: degree of the b-spline
    :return: the open uniform b-spline with constant extrapolation
    """
    ...


@overload
def make_open_uniform_constant(degree: int, begin: float, end: float, num_elems: int) -> BSpline:
    """Create a uniform b-spline with open boundary condition and constant extrapolation

    > NOTE: useful for fitting points
    >>> bspline = make_open_uniform_constant(3, 0.0, 10.0, 10)
    >>> bspline.fit(x, y)

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :return: the open uniform b-spline with constant extrapolation
    """
    ...


@overload
def make_open_uniform_constant(
    degree: int, begin: float, end: float, num_elems: int, ctrl_points: npt.ArrayLike
) -> BSpline:
    """Create a uniform b-spline with open boundary condition and constant extrapolation

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :param ctrl_points: control points of the b-spline
    :return: the open uniform b-spline with constant extrapolation
    """
    ...


def make_open_uniform_constant(
    degree: int,
    begin: float | None = None,
    end: float | None = None,
    num_elems: int | None = None,
    ctrl_points: npt.ArrayLike | None = None,
) -> BSpline:
    if all((arg is not None for arg in (begin, end, num_elems, ctrl_points))):
        return BSpline(_impl.make_open_uniform_constant(degree, begin, end, num_elems, ctrl_points))
    if all((arg is not None for arg in (begin, end, num_elems))) and ctrl_points is None:
        return BSpline(_impl.make_open_uniform_constant(degree, begin, end, num_elems))
    if all((arg is None for arg in (begin, end, num_elems, ctrl_points))):
        return BSpline(_impl.make_open_uniform_constant(degree))

    error = """
    This function has three possible overloads:
    - make_open_uniform_constant(degree)
    - make_open_uniform_constant(degree, begin, end, num_elems)
    - make_open_uniform_constant(degree, begin, end, num_elems, ctrl_points)
    """

    raise ValueError(dedent(error))


@overload
def make_open_nonuniform_constant(degree: int) -> BSpline:
    """Create a non-uniform b-spline with open boundary condition and constant extrapolation

    > NOTE: useful for interpolating points
    >>> bspline = make_open_nonuniform_constant(3)
    >>> bspline.interpolate(x, y, additional_conditions)

    :param degree: degree of the b-spline
    :return: the open non-uniform b-spline with constant extrapolation
    """
    ...


@overload
def make_open_nonuniform_constant(degree: int, knots: npt.ArrayLike) -> BSpline:
    """Create a non-uniform b-spline with open boundary condition and constant extrapolation

    > NOTE: useful for fitting points
    >>> bspline = make_open_nonuniform_constant(3, knots)
    >>> bspline.fit(x, y)

    :param degree: degree of the b-spline
    :param knots: knots of the b-spline
    :return: the open non-uniform b-spline with constant extrapolation
    """
    ...


@overload
def make_open_nonuniform_constant(degree: int, knots: npt.ArrayLike, ctrl_points: npt.ArrayLike) -> BSpline:
    """Create a non-uniform b-spline with open boundary condition and constant extrapolation

    :param degree: degree of the b-spline
    :param knots: knots of the b-spline
    :param ctrl_points: control points of the b-spline
    :return: the open non-uniform b-spline with constant extrapolation
    """
    ...


def make_open_nonuniform_constant(
    degree: int, knots: npt.ArrayLike | None = None, ctrl_points: npt.ArrayLike | None = None
) -> BSpline:
    if all((arg is not None for arg in (knots, ctrl_points))):
        return BSpline(_impl.make_open_nonuniform_constant(degree, knots, ctrl_points))
    if knots is not None and ctrl_points is None:
        return BSpline(_impl.make_open_nonuniform_constant(degree, knots))
    if all((arg is None for arg in (knots, ctrl_points))):
        return BSpline(_impl.make_open_nonuniform_constant(degree))

    error = """
    This function has three possible overloads:
    - make_open_nonuniform_constant(degree)
    - make_open_nonuniform_constant(degree, knots)
    - make_open_nonuniform_constant(degree, knots, ctrl_points)
    """

    raise ValueError(dedent(error))


@overload
def make_clamped_uniform(degree: int) -> BSpline:
    """Create a uniform b-spline with clamped boundary condition

    > NOTE: useful for interpolating points
    >>> bspline = make_clamped_uniform(3)
    >>> bspline.interpolate(x, y, additional_conditions)

    :param degree: degree of the b-spline
    :return: the clamped uniform b-spline
    """
    ...


@overload
def make_clamped_uniform(degree: int, begin: float, end: float, num_elems: int) -> BSpline:
    """Create a uniform b-spline with clamped boundary condition

    > NOTE: useful for fitting points
    >>> bspline = make_clamped_uniform(3, 0.0, 10.0, 10)
    >>> bspline.fit(x, y)

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :return: the clamped uniform b-spline
    """
    ...


@overload
def make_clamped_uniform(degree: int, begin: float, end: float, num_elems: int, ctrl_points: npt.ArrayLike) -> BSpline:
    """Create a uniform b-spline with clamped boundary condition

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :param ctrl_points: control points of the b-spline
    :return: the clamped uniform b-spline
    """
    ...


def make_clamped_uniform(
    degree: int,
    begin: float | None = None,
    end: float | None = None,
    num_elems: int | None = None,
    ctrl_points: npt.ArrayLike | None = None,
) -> BSpline:
    if all((arg is not None for arg in (begin, end, num_elems, ctrl_points))):
        return BSpline(_impl.make_clamped_uniform(degree, begin, end, num_elems, ctrl_points))
    if all((arg is not None for arg in (begin, end, num_elems))) and ctrl_points is None:
        return BSpline(_impl.make_clamped_uniform(degree, begin, end, num_elems))
    if all((arg is None for arg in (begin, end, num_elems, ctrl_points))):
        return BSpline(_impl.make_clamped_uniform(degree))

    error = """
    This function has three possible overloads:
    - make_clamped_uniform(degree)
    - make_clamped_uniform(degree, begin, end, num_elems)
    - make_clamped_uniform(degree, begin, end, num_elems, ctrl_points)
    """

    raise ValueError(dedent(error))


@overload
def make_clamped_nonuniform(degree: int) -> BSpline:
    """Create a non-uniform b-spline with clamped boundary condition

    > NOTE: useful for interpolating points
    >>> bspline = make_clamped_nonuniform(3)
    >>> bspline.interpolate(x, y, additional_conditions)

    :param degree: degree of the b-spline
    :return: the clamped non-uniform b-spline
    """
    ...


@overload
def make_clamped_nonuniform(degree: int, knots: npt.ArrayLike) -> BSpline:
    """Create a non-uniform b-spline with clamped boundary condition

    > NOTE: useful for fitting points
    >>> bspline = make_clamped_nonuniform(3, knots)
    >>> bspline.fit(x, y)

    :param degree: degree of the b-spline
    :param knots: knots of the b-spline
    :return: the clamped non-uniform b-spline
    """
    ...


@overload
def make_clamped_nonuniform(degree: int, knots: npt.ArrayLike, ctrl_points: npt.ArrayLike) -> BSpline:
    """Create a non-uniform b-spline with clamped boundary condition

    :param degree: degree of the b-spline
    :param knots: knots of the b-spline
    :param ctrl_points: control points of the b-spline
    :return: the clamped non-uniform b-spline
    """
    ...


def make_clamped_nonuniform(
    degree: int, knots: npt.ArrayLike | None = None, ctrl_points: npt.ArrayLike | None = None
) -> BSpline:
    if all((arg is not None for arg in (knots, ctrl_points))):
        return BSpline(_impl.make_clamped_nonuniform(degree, knots, ctrl_points))
    if knots is not None and ctrl_points is None:
        return BSpline(_impl.make_clamped_nonuniform(degree, knots))
    if all((arg is None for arg in (knots, ctrl_points))):
        return BSpline(_impl.make_clamped_nonuniform(degree))

    error = """
    This function has three possible overloads:
    - make_clamped_nonuniform(degree)
    - make_clamped_nonuniform(degree, knots)
    - make_clamped_nonuniform(degree, knots, ctrl_points)
    """

    raise ValueError(dedent(error))


@overload
def make_clamped_uniform_constant(degree: int) -> BSpline:
    """Create a uniform b-spline with clamped boundary condition and constant extrapolation

    > NOTE: useful for interpolating points
    >>> bspline = make_clamped_uniform_constant(3)
    >>> bspline.interpolate(x, y, additional_conditions)

    :param degree: degree of the b-spline
    :return: the clamped uniform b-spline with constant extrapolation
    """
    ...


@overload
def make_clamped_uniform_constant(degree: int, begin: float, end: float, num_elems: int) -> BSpline:
    """Create a uniform b-spline with clamped boundary condition and constant extrapolation

    > NOTE: useful for fitting points
    >>> bspline = make_clamped_uniform_constant(3, 0.0, 10.0, 10)
    >>> bspline.fit(x, y)

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :return: the clamped uniform b-spline with constant extrapolation
    """
    ...


@overload
def make_clamped_uniform_constant(
    degree: int, begin: float, end: float, num_elems: int, ctrl_points: npt.ArrayLike
) -> BSpline:
    """Create a uniform b-spline with clamped boundary condition and constant extrapolation

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :param ctrl_points: control points of the b-spline
    :return: the clamped uniform b-spline with constant extrapolation
    """
    ...


def make_clamped_uniform_constant(
    degree: int,
    begin: float | None = None,
    end: float | None = None,
    num_elems: int | None = None,
    ctrl_points: npt.ArrayLike | None = None,
) -> BSpline:
    if all((arg is not None for arg in (begin, end, num_elems, ctrl_points))):
        return BSpline(_impl.make_clamped_uniform_constant(degree, begin, end, num_elems, ctrl_points))
    if all((arg is not None for arg in (begin, end, num_elems))) and ctrl_points is None:
        return BSpline(_impl.make_clamped_uniform_constant(degree, begin, end, num_elems))
    if all((arg is None for arg in (begin, end, num_elems, ctrl_points))):
        return BSpline(_impl.make_clamped_uniform_constant(degree))

    error = """
    This function has three possible overloads:
    - make_clamped_uniform_constant(degree)
    - make_clamped_uniform_constant(degree, begin, end, num_elems)
    - make_clamped_uniform_constant(degree, begin, end, num_elems, ctrl_points)
    """

    raise ValueError(dedent(error))


@overload
def make_clamped_nonuniform_constant(degree: int) -> BSpline:
    """Create a non-uniform b-spline with clamped boundary condition and constant extrapolation

    > NOTE: useful for interpolating points
    >>> bspline = make_clamped_nonuniform_constant(3)
    >>> bspline.interpolate(x, y, additional_conditions)

    :param degree: degree of the b-spline
    :return: the clamped non-uniform b-spline with constant extrapolation
    """
    ...


@overload
def make_clamped_nonuniform_constant(degree: int, knots: npt.ArrayLike) -> BSpline:
    """Create a non-uniform b-spline with clamped boundary condition and constant extrapolation

    > NOTE: useful for fitting points
    >>> bspline = make_clamped_nonuniform_constant(3, knots)
    >>> bspline.fit(x, y)

    :param degree: degree of the b-spline
    :param knots: knots of the b-spline
    :return: the clamped non-uniform b-spline with constant extrapolation
    """
    ...


@overload
def make_clamped_nonuniform_constant(degree: int, knots: npt.ArrayLike, ctrl_points: npt.ArrayLike) -> BSpline:
    """Create a non-uniform b-spline with clamped boundary condition and constant extrapolation

    :param degree: degree of the b-spline
    :param knots: knots of the b-spline
    :param ctrl_points: control points of the b-spline
    :return: the clamped non-uniform b-spline with constant extrapolation
    """
    ...


def make_clamped_nonuniform_constant(
    degree: int, knots: npt.ArrayLike | None = None, ctrl_points: npt.ArrayLike | None = None
) -> BSpline:
    if all((arg is not None for arg in (knots, ctrl_points))):
        return BSpline(_impl.make_clamped_nonuniform_constant(degree, knots, ctrl_points))
    if knots is not None and ctrl_points is None:
        return BSpline(_impl.make_clamped_nonuniform_constant(degree, knots))
    if all((arg is None for arg in (knots, ctrl_points))):
        return BSpline(_impl.make_clamped_nonuniform_constant(degree))

    error = """
    This function has three possible overloads:
    - make_clamped_nonuniform_constant(degree)
    - make_clamped_nonuniform_constant(degree, knots)
    - make_clamped_nonuniform_constant(degree, knots, ctrl_points)
    """

    raise ValueError(dedent(error))


@overload
def make_periodic_uniform(degree: int) -> BSpline:
    """Create a uniform b-spline with periodic boundary condition

    > NOTE: useful for interpolating points
    >>> bspline = make_periodic_uniform(3)
    >>> bspline.interpolate(x, y)

    :param degree: degree of the b-spline
    :return: the periodic uniform b-spline
    """
    ...


@overload
def make_periodic_uniform(degree: int, begin: float, end: float, num_elems: int) -> BSpline:
    """Create a uniform b-spline with periodic boundary condition

    > NOTE: useful for fitting points
    >>> bspline = make_periodic_uniform(3, 0.0, 10.0, 10)
    >>> bspline.fit(x, y)

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :return: the periodic uniform b-spline
    """
    ...


@overload
def make_periodic_uniform(degree: int, begin: float, end: float, num_elems: int, ctrl_points: npt.ArrayLike) -> BSpline:
    """Create a uniform b-spline with periodic boundary condition

    :param degree: degree of the b-spline
    :param begin: starting knot
    :param end: end knot
    :param num_elems: number of knots
    :param ctrl_points: control points of the b-spline
    :return: the periodic uniform b-spline
    """
    ...


def make_periodic_uniform(
    degree: int,
    begin: float | None = None,
    end: float | None = None,
    num_elems: int | None = None,
    ctrl_points: npt.ArrayLike | None = None,
) -> BSpline:
    if all((arg is not None for arg in (begin, end, num_elems, ctrl_points))):
        return BSpline(_impl.make_periodic_uniform(degree, begin, end, num_elems, ctrl_points))
    if all((arg is not None for arg in (begin, end, num_elems))) and ctrl_points is None:
        return BSpline(_impl.make_periodic_uniform(degree, begin, end, num_elems))
    if all((arg is None for arg in (begin, end, num_elems, ctrl_points))):
        return BSpline(_impl.make_periodic_uniform(degree))

    error = """
    This function has three possible overloads:
    - make_periodic_uniform(degree)
    - make_periodic_uniform(degree, begin, end, num_elems)
    - make_periodic_uniform(degree, begin, end, num_elems, ctrl_points)
    """

    raise ValueError(dedent(error))


@overload
def make_periodic_nonuniform(degree: int) -> BSpline:
    """Create a non-uniform b-spline with periodic boundary condition

    > NOTE: useful for interpolating points
    >>> bspline = make_periodic_nonuniform(3)
    >>> bspline.interpolate(x, y)

    :param degree: degree of the b-spline
    :return: the periodic non-uniform b-spline
    """
    ...


@overload
def make_periodic_nonuniform(degree: int, knots: npt.ArrayLike) -> BSpline:
    """Create a non-uniform b-spline with periodic boundary condition

    > NOTE: useful for fitting points
    >>> bspline = make_periodic_nonuniform(3, knots)
    >>> bspline.fit(x, y)

    :param degree: degree of the b-spline
    :param knots: knots of the b-spline
    :return: the periodic non-uniform b-spline
    """
    ...


@overload
def make_periodic_nonuniform(degree: int, knots: npt.ArrayLike, ctrl_points: npt.ArrayLike) -> BSpline:
    """Create a non-uniform b-spline with periodic boundary condition

    :param degree: degree of the b-spline
    :param knots: knots of the b-spline
    :param ctrl_points: control points of the b-spline
    :return: the periodic non-uniform b-spline
    """
    ...


def make_periodic_nonuniform(
    degree: int, knots: npt.ArrayLike | None = None, ctrl_points: npt.ArrayLike | None = None
) -> BSpline:
    if all((arg is not None for arg in (knots, ctrl_points))):
        return BSpline(_impl.make_periodic_nonuniform(degree, knots, ctrl_points))
    if knots is not None and ctrl_points is None:
        return BSpline(_impl.make_periodic_nonuniform(degree, knots))
    if all((arg is None for arg in (knots, ctrl_points))):
        return BSpline(_impl.make_periodic_nonuniform(degree))

    error = """
    This function has three possible overloads:
    - make_periodic_nonuniform(degree)
    - make_periodic_nonuniform(degree, knots)
    - make_periodic_nonuniform(degree, knots, ctrl_points)
    """

    raise ValueError(dedent(error))
