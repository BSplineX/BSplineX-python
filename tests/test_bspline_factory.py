"""Tests for the InterpolantCondition wrapper class"""

import numpy as np
import numpy.typing as npt

import bsplinex as bs
import bsplinex._bsplinex_impl as _bs

from . import utils

DEGREE: int = 3
POINTS: utils.Points = utils.points()


def get_additional_conditions(degree: int, x: npt.NDArray[np.float64]) -> list[bs.InterpolantCondition]:
    conditions = []
    for i in range(degree - 1):
        conditions.append(bs.InterpolantCondition(x_value=x[-(i % 2)], y_value=0.0, derivative_order=i + 1))

    return conditions


def test_make_open_uniform():
    bspline_data = utils.uniform_data(DEGREE, utils.BoundaryCondition.OPEN)
    additional_conditions = get_additional_conditions(DEGREE, POINTS.x)

    bspline = bs.make_open_uniform(DEGREE)
    _bspline = _bs.make_open_uniform(DEGREE)

    bspline.interpolate(POINTS.x, POINTS.y, additional_conditions)
    _bspline.interpolate(POINTS.x, POINTS.y, additional_conditions)

    eval_x = np.mean(POINTS.x)
    assert bspline.evaluate(eval_x) == _bspline.evaluate(eval_x)
