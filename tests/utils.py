import numpy as np
from numpy import typing as npt


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


FloatArray = npt.NDArray[np.float64]
