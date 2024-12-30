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
        bs.open_uniform(0, 0., 1., 2, [0., 1.])


@pytest.mark.parametrize("factory_method", [
    bs.open_uniform,
    bs.clamped_uniform,
    bs.periodic_uniform,
])
def test_factory_methods_uniform_without_control_points(factory_method):
    degree = 3
    knots_begin = 0.1
    knots_end = 13.2
    n_knots = 100
    bspline = factory_method(degree, knots_begin, knots_end, n_knots)
    assert bspline.evaluate(5.) == 0.


@pytest.mark.parametrize("factory_method", [
    bs.open_nonuniform,
    bs.clamped_nonuniform,
    bs.periodic_nonuniform,
])
def test_factory_methods_nonuniform_without_control_points(factory_method):
    degree = 3
    knots = np.linspace(0.1, 13.2, 100)
    bspline = factory_method(degree, knots)
    assert bspline.evaluate(5.) == 0.


@pytest.mark.parametrize("factory_method, num_control_points_lambda", [
    (bs.open_uniform, lambda n_knots, degree: n_knots - degree - 1),
    (bs.clamped_uniform, lambda n_knots, degree: n_knots + degree - 1),
    (bs.periodic_uniform, lambda n_knots, degree: n_knots - 1),
])
def test_factory_methods_uniform_with_control_points(factory_method, num_control_points_lambda):
    degree = 3
    knots_begin = 0.1
    knots_end = 13.2
    n_knots = 100
    n_control_points = num_control_points_lambda(n_knots, degree)
    control_points = list(map(lambda x: (50 - x) ** 2, range(n_control_points)))
    bspline = factory_method(degree, knots_begin, knots_end, n_knots, control_points)
    assert bspline.evaluate(5.) > 0.


@pytest.mark.parametrize("factory_method, num_control_points_lambda", [
    (bs.open_nonuniform, lambda n_knots, degree: n_knots - degree - 1),
    (bs.clamped_nonuniform, lambda n_knots, degree: n_knots + degree - 1),
    (bs.periodic_nonuniform, lambda n_knots, degree: n_knots - 1),
])
def test_factory_methods_nonuniform_with_control_points(factory_method, num_control_points_lambda):
    degree = 3
    knots = np.linspace(0.1, 13.2, 100)
    n_knots = len(knots)
    n_control_points = num_control_points_lambda(n_knots, degree)
    control_points = list(map(lambda x: (50 - x) ** 2, range(n_control_points)))
    bspline = factory_method(degree, knots, control_points)
    assert bspline.evaluate(5.) > 0.
