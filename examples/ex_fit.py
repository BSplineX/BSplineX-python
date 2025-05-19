"""Create and fit a uniform periodic B-Spline."""

import bsplinex as bs
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Points to fit
    x_fit = [2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9]
    y_fit = [11.2, 22.3, 13.4, 14.5, 25.6, 36.7, 17.8]

    # Initialize a cubic, uniform, periodic B-Spline curve where knots are
    # linearly spaced from 2.0 to 9.0
    degree = 3
    knots_begin = 2.0
    knots_end = 9.0
    num_knots = 12

    bspline = bs.make_periodic_uniform(degree, knots_begin, knots_end, num_knots)

    # Fit the curve to the points
    bspline.fit(x_fit, y_fit)

    # Evaluate the curve at some points
    eval_x = np.linspace(2.0, 9.0, 1000)
    eval_y = bspline.evaluate(eval_x)

    plt.figure("Periodic B-Spline fitting")
    plt.scatter(x_fit, y_fit, label="Original points")
    plt.plot(eval_x, eval_y, label="Fitted B-Spline")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
