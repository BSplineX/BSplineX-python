"""
Tests that SWIG interface works as expected.
"""

import BSplineX as bs
import numpy as np
import pytest


def test_exceptions():
    """
    Test that exceptions are propagated correctly.
    """
    with pytest.raises(RuntimeError):
        bs.make_open_uniform(0, 0.0, 1.0, 2, [0.0, 1.0])


@pytest.mark.parametrize(
    "factory_method",
    [
        bs.make_open_uniform,
        bs.make_clamped_uniform,
        bs.make_periodic_uniform,
    ],
)
def test_factory_methods_uniform_without_control_points(factory_method):
    degree = 3
    knots_begin = 0.1
    knots_end = 13.2
    n_knots = 100
    bspline = factory_method(degree, knots_begin, knots_end, n_knots)
    assert bspline.evaluate(5.0) == 0.0


@pytest.mark.parametrize(
    "factory_method",
    [
        bs.make_open_nonuniform,
        bs.make_clamped_nonuniform,
        bs.make_periodic_nonuniform,
    ],
)
def test_factory_methods_nonuniform_without_control_points(factory_method):
    degree = 3
    knots = np.linspace(0.1, 13.2, 100)
    bspline = factory_method(degree, knots)
    assert bspline.evaluate(5.0) == 0.0


@pytest.mark.parametrize(
    "factory_method, num_control_points_lambda",
    [
        (bs.make_open_uniform, lambda n_knots, degree: n_knots - degree - 1),
        (bs.make_clamped_uniform, lambda n_knots, degree: n_knots + degree - 1),
        (bs.make_periodic_uniform, lambda n_knots, degree: n_knots - 1),
    ],
)
def test_factory_methods_uniform_with_control_points(factory_method, num_control_points_lambda):
    degree = 3
    knots_begin = 0.1
    knots_end = 13.2
    n_knots = 100
    n_control_points = num_control_points_lambda(n_knots, degree)
    control_points = list(map(lambda x: (50 - x) ** 2, range(n_control_points)))
    bspline = factory_method(degree, knots_begin, knots_end, n_knots, control_points)
    assert bspline.evaluate(5.0) > 0.0


@pytest.mark.parametrize(
    "factory_method, num_control_points_lambda",
    [
        (bs.make_open_nonuniform, lambda n_knots, degree: n_knots - degree - 1),
        (bs.make_clamped_nonuniform, lambda n_knots, degree: n_knots + degree - 1),
        (bs.make_periodic_nonuniform, lambda n_knots, degree: n_knots - 1),
    ],
)
def test_factory_methods_nonuniform_with_control_points(factory_method, num_control_points_lambda):
    degree = 3
    knots = np.linspace(0.1, 13.2, 100)
    n_knots = len(knots)
    n_control_points = num_control_points_lambda(n_knots, degree)
    control_points = list(map(lambda x: (50 - x) ** 2, range(n_control_points)))
    bspline = factory_method(degree, knots, control_points)
    assert bspline.evaluate(5.0) > 0.0
