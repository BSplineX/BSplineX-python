from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


KNOTS_SIZE: int = 200
POINTS_SIZE: int = 300


class BoundaryCondition(Enum):
    OPEN = 0
    CLAMPED = 1
    PERIODIC = 2


@dataclass
class UniformKnots:
    start: float
    stop: float
    num_elems: int


@dataclass
class UniformData:
    knots: UniformKnots
    control_points: npt.NDArray[np.float64]


@dataclass
class NonUniformData:
    knots: npt.NDArray[np.float64]
    control_points: npt.NDArray[np.float64]


@dataclass
class Points:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]


def _control_points_size(degree: int, boundary_condition: BoundaryCondition) -> int:
    match boundary_condition:
        case BoundaryCondition.OPEN:
            return KNOTS_SIZE - degree - 1
        case BoundaryCondition.CLAMPED:
            return KNOTS_SIZE - degree - 1
        case BoundaryCondition.PERIODIC:
            return KNOTS_SIZE - degree - 1
        case _:
            raise ValueError(f"Unknown boundary condition {boundary_condition.value}")


def uniform_data(degree: int, boundary_condition: BoundaryCondition) -> UniformData:
    knots = UniformKnots(start=1.0, stop=21.0, num_elems=KNOTS_SIZE)
    control_points_size = _control_points_size(degree, boundary_condition)
    control_points = np.sin(np.linspace(knots.start, knots.stop, control_points_size))
    return UniformData(knots=knots, control_points=control_points)


def nonuniform_data(degree: int, boundary_condition: BoundaryCondition) -> NonUniformData:
    knots = np.pow(np.linspace(2.0, 10.73, KNOTS_SIZE), 2)
    control_points_size = _control_points_size(degree, boundary_condition)
    t = np.linspace(0.0, 1.0, control_points_size)
    control_points = np.sin(t)
    return NonUniformData(knots=knots, control_points=control_points)


def points() -> Points:
    t = np.linspace(0.0, 1.0, POINTS_SIZE)
    x = np.cos(t)
    y = np.sin(t)
    return Points(x=x, y=y)
