"""Create and interpolate a uniform clamped B-Spline."""

import bsplinex as bs
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Points to interpolate
    x_int = [2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9]
    y_int = [11.2, 22.3, 13.4, 14.5, 25.6, 36.7, 17.8]

    # Initialize a cubic, uniform, clamped B-Spline curve
    degree = 3

    bspline = bs.make_clamped_uniform(degree)

    # Since we are using a cubic clamped B-Spline, we can give two (degree - 1)
    # additional conditions the B-Spline has to respect. For example, let's
    # make sure the first derivative at x_int[0] is 0.0 and at x_int[-1] is 2.0
    additional_conditions = [
        bs.InterpolantCondition(x_int[0], 0.0, 1),
        bs.InterpolantCondition(x_int[-1], 2.0, 1),
    ]

    # Interpolate the curve to the points
    bspline.interpolate(x_int, y_int, additional_conditions)

    # Evaluate the curve at some points
    eval_x = np.linspace(x_int[0], x_int[-1], 1000)
    eval_y = bspline.evaluate(eval_x)
    eval_dy = bspline.evaluate(eval_x, 1)

    plt.figure("Clamped B-Spline interpolation")
    plt.scatter(x_int, y_int, label="Original points")
    plt.plot(eval_x, eval_y, label="Interpolated B-Spline")
    plt.plot(eval_x, eval_dy, label="First derivative of the interpolated B-Spline")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
