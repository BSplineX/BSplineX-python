import BSplineX as bs
import numpy as np
import pytest

from reference import OpenBSpline, Extrapolation, ClampedBSpline, PeriodicBSpline
from utils import num_control_points_clamped, num_control_points_periodic, num_control_points_open, x_values_clamped, \
    x_values_open, x_values_periodic


def ref_open(degree: int, knots: np.ndarray, control_points: np.ndarray):
    return OpenBSpline(knots, control_points, degree, Extrapolation.NONE)


def ref_clamped(degree: int, knots: np.ndarray, control_points: np.ndarray):
    return ClampedBSpline(knots, control_points, degree, Extrapolation.NONE)


def ref_periodic(degree: int, knots: np.ndarray, control_points: np.ndarray):
    return PeriodicBSpline(knots, control_points, degree, Extrapolation.PERIODIC)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def degree() -> int:
    return 3


@pytest.fixture
def knots() -> np.ndarray:
    return np.arange(0., 10., 0.1)


@pytest.mark.parametrize("bspline_factory, ref_factory, x_values_factory, num_control_points_lambda", [
    (bs.open_nonuniform, ref_open, x_values_open, num_control_points_open),
    (bs.clamped_nonuniform, ref_clamped, x_values_clamped, num_control_points_clamped),
    (bs.periodic_nonuniform, ref_periodic, x_values_periodic, num_control_points_periodic),
])
def test_evaluate(bspline_factory, ref_factory, x_values_factory, num_control_points_lambda, rng, degree, knots):
    control_points = rng.uniform(-1., 1., num_control_points_lambda(len(knots), degree))
    bspline = bspline_factory(degree, knots, control_points)
    ref_bspline = ref_factory(degree, knots, control_points)

    x_values = x_values_factory(knots, degree)
    y_values = ref_bspline.evaluate(x_values)

    y_computed = [bspline.evaluate(x) for x in x_values]
    assert np.allclose(y_values, y_computed)


@pytest.mark.parametrize("bspline_factory, x_values_factory, num_control_points_lambda", [
    (bs.open_nonuniform, x_values_open, num_control_points_open),
    (bs.clamped_nonuniform, x_values_clamped, num_control_points_clamped),
    (bs.periodic_nonuniform, x_values_periodic, num_control_points_periodic),
])
def test_basis(bspline_factory, x_values_factory, num_control_points_lambda, rng, degree, knots):
    control_points = rng.uniform(-1., 1., num_control_points_lambda(len(knots), degree))
    bspline = bspline_factory(degree, knots, control_points)

    x_values = x_values_factory(knots, degree)
    for x in x_values:
        basis = bspline.basis(x)
        full_control_points = bspline.get_control_points()

        assert len(basis) == len(full_control_points)
        assert np.isclose(np.dot(basis, full_control_points), bspline.evaluate(x))


@pytest.mark.parametrize("bspline_factory, x_values_factory, num_control_points_lambda", [
    (bs.open_nonuniform, x_values_open, num_control_points_open),
    (bs.clamped_nonuniform, x_values_clamped, num_control_points_clamped),
    (bs.periodic_nonuniform, x_values_periodic, num_control_points_periodic),
])
def test_fit(bspline_factory, x_values_factory, num_control_points_lambda, rng, degree, knots):
    x_values = x_values_factory(knots, degree)
    bspline = bspline_factory(degree, knots, rng.uniform(-1., 1., num_control_points_lambda(len(knots), degree)))
    control_points = bspline.get_control_points()
    y_values = [bspline.evaluate(x) for x in x_values]

    bspline2 = bspline_factory(degree, knots)
    bspline2.fit(x_values, y_values)
    control_points2 = bspline2.get_control_points()

    control_points = np.array(control_points)
    control_points2 = np.array(control_points2)

    assert len(control_points) == len(control_points2)
    assert np.allclose(control_points, control_points2)
    for x in x_values:
        assert np.isclose(bspline.evaluate(x), bspline2.evaluate(x))
