"""Create and evaluate a non-uniform open B-Spline."""

import bsplinex as bs


def main() -> None:
    # Initialize a cubic, non-uniform, open B-Spline curve with given knots and
    # control points
    degree = 3
    knots = [0.1, 1.3, 2.2, 2.2, 4.9, 6.3, 6.3, 6.3, 13.2]
    ctrl_points = [0.1, 1.3, 2.2, 4.9, 13.2]

    bspline = bs.make_open_nonuniform(degree, knots, ctrl_points)

    # Evaluate the curve at some points
    eval_x = [3.0, 3.4, 5.1, 6.2]
    for val in eval_x:
        print(f"Evaluate bspline at {val}:")
        print(f"bspline                   = {bspline.evaluate(val)}")
        print(f"bspline first derivative  = {bspline.evaluate(val, 1)}")
        print(f"bspline second derivative = {bspline.evaluate(val, 2)}")


if __name__ == "__main__":
    main()
