import sys
import os.path
import BSplineX as bs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, "tests")))
from tests.reference import ClampedBSpline, PeriodicBSpline
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Main function."""

    # Points to fit
    x_fit = np.linspace(0, 2 * np.pi, 2000)
    y_fit = np.sin(x_fit)

    # Initialize a cubic, uniform, periodic B-spline curve where knots are [0.1, 1.1, ..., 10.1, 11.1]
    degree = 3
    knots_begin = 0
    knots_end = 2 * np.pi
    num_knots = 1000
    knots = np.linspace(knots_begin, knots_end, num_knots, dtype=np.float64)
    ctrl_pts = np.zeros(PeriodicBSpline.required_control_points(num_knots, degree))

    bspline = PeriodicBSpline.from_data(knots, ctrl_pts, degree)
    bsplinex: bs.PeriodicUniform = bs.make_periodic_uniform(3, knots_begin, knots_end, num_knots, ctrl_pts)

    # Fit the curve to the points
    bspline.fit(x_fit, y_fit)
    bsplinex.fit(x_fit, y_fit)

    print(bspline.control_points)
    print(np.array(bsplinex.get_knots()))
    print(np.array(bsplinex.get_control_points()))
    assert (bspline.knots == bsplinex.get_knots()).all()
    assert np.allclose(bspline.control_points, bsplinex.get_control_points(), rtol=1e-12, atol=1e-12)

    # Evaluate the curve at some points. Since the curve is periodic, the evaluation can done at any point
    eval_x = np.linspace(-10, 2 * np.pi + 10, 10000)
    assert np.allclose(bspline.evaluate(eval_x), bsplinex.evaluate(eval_x), rtol=1e-12, atol=1e-12)
    eval_y = np.array(bspline.evaluate(eval_x))
    evalx_y = np.array(bsplinex.evaluate(eval_x))

    plt.figure()
    plt.scatter(x_fit, y_fit)
    plt.plot(eval_x, eval_y)
    plt.plot(eval_x, evalx_y)
    plt.show()


if __name__ == "__main__":
    main()
