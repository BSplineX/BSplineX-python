import numpy as np


def num_control_points_open(n_knots, degree):
    return n_knots - degree - 1


def num_control_points_clamped(n_knots, degree):
    return n_knots + degree - 1


def num_control_points_periodic(n_knots, degree):
    return n_knots - 1


def x_values_open(knots, degree):
    return np.linspace(
        knots[degree],
        knots[-degree - 1] - 1e-6,
        1000,
    )


def x_values_clamped(knots, _):
    return np.linspace(
        knots[0],
        knots[-1] - 1e-6,
        1000,
    )


def x_values_periodic(knots, degree):
    period = knots[-degree - 1] - knots[degree]
    return np.linspace(
        knots[degree] - 3 * period,
        knots[-degree - 1] + 3 * period,
        3000,
    )
