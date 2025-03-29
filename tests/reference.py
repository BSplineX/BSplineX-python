"""
Reference BSpline implementation.
"""

import argparse
from abc import abstractmethod, ABC
from enum import Enum
from typing import Iterable, cast

import numpy as np
from scipy.interpolate import BSpline as BSpline_, make_lsq_spline, make_interp_spline

from utils import (
    num_control_points_periodic,
    num_control_points_clamped,
    num_control_points_open,
    x_values_open,
    x_values_clamped,
    x_values_periodic,
    FloatArray,
)

DIST_R = 1e-20


class Extrapolation(str, Enum):
    NONE = "none"
    PERIODIC = "periodic"


class BoundaryCondition(str, Enum):
    OPEN = "open"
    CLAMPED = "clamped"
    PERIODIC = "periodic"


class Curve(str, Enum):
    UNIFORM = "uniform"
    NON_UNIFORM = "non-uniform"


class BSpline(ABC):
    def __init__(self, bspline: BSpline_ | None = None):
        super().__init__()
        if bspline is not None:
            self.bspline = bspline

    @classmethod
    def from_data(cls, knots: FloatArray, control_points: FloatArray, degree: int) -> "BSpline":
        knots = cls._pad_knots(knots, degree)
        control_points = cls._pad_control_points(control_points, degree)

        return cls(BSpline_(knots, control_points, degree, extrapolate=cls._extrapolation()))  # pyright: ignore

    @classmethod
    def empty(cls, degree: int) -> "BSpline":
        """Returns an empty BSpline object."""
        knots, control_points = cls._empty_data(degree)
        return cls.from_data(knots, control_points, degree)

    @property
    def knots(self) -> FloatArray:
        return self.bspline.t

    @property
    def control_points(self) -> FloatArray:
        return cast(FloatArray, self.bspline.c)

    @property
    def degree(self) -> int:
        return self.bspline.k

    @property
    def extrapolation(self) -> bool:
        return self.bspline.extrapolate

    @property
    def bspline(self) -> BSpline_:
        return self._bspline

    @bspline.setter
    def bspline(self, bspline: BSpline_):
        self._bspline = bspline
        n = len(self.control_points)
        self._basis: list[BSpline_] = [
            BSpline_(
                self.knots,
                (np.arange(n) == i).astype(float),
                self.degree,
                extrapolate=self.extrapolation,
            )
            for i in range(n)
        ]

    @property
    @abstractmethod
    def domain(self) -> tuple[float, float]:
        pass

    def evaluate(self, x: Iterable[float], derivative_order: int = 0) -> FloatArray:
        return cast(FloatArray, self.bspline(x, nu=derivative_order))

    def derivative(self, derivative_order=1) -> "BSpline":
        return self.__class__(self.bspline.derivative(nu=derivative_order))

    def fit(self, x: Iterable[float], y: Iterable[float]) -> None:
        self.bspline = make_lsq_spline(x, y, self.knots, self.degree)

    def interpolate(
        self,
        x: Iterable[float],
        y: Iterable[float],
        conditions: tuple[list[tuple[int, float]], list[tuple[int, float]]],
    ) -> None:
        conds_left, conds_right = conditions
        conds_left = conds_left or None
        conds_right = conds_right or None
        # if not conds_left and not conds_right:
        #     # conditions = None
        #     pass
        # else:
        conds = (conds_left, conds_right)

        self.bspline = make_interp_spline(x, y, k=self.degree, bc_type=conds)

    def basis(self, x: float | Iterable[float], derivative_order: int = 1) -> FloatArray:
        return cast(FloatArray, np.array([b(x, nu=derivative_order) for b in self._basis]).transpose())

    def copy(self) -> "BSpline":
        return self.__class__(self.bspline)

    @staticmethod
    @abstractmethod
    def _extrapolation() -> Extrapolation:
        pass

    @staticmethod
    @abstractmethod
    def _pad_knots(knots: Iterable[float], degree: int) -> FloatArray:
        pass

    @staticmethod
    @abstractmethod
    def _pad_control_points(control_points: FloatArray, degree: int) -> FloatArray:
        pass

    @property
    @abstractmethod
    def boundary_condition(self) -> BoundaryCondition:
        pass

    @staticmethod
    @abstractmethod
    def _empty_data(degree: int) -> tuple[FloatArray, FloatArray]:
        pass


class OpenBSpline(BSpline):
    @staticmethod
    def _extrapolation() -> Extrapolation:
        return Extrapolation.NONE

    @staticmethod
    def _pad_knots(knots: Iterable[float], degree: int) -> FloatArray:
        return np.array(knots)

    @staticmethod
    def _pad_control_points(control_points: FloatArray, degree: int) -> FloatArray:
        return np.array(control_points)

    @property
    def boundary_condition(self) -> BoundaryCondition:
        return BoundaryCondition.OPEN

    @staticmethod
    def _empty_data(degree: int) -> tuple[FloatArray, FloatArray]:
        n_knots = 2 * degree + 2
        n_control_points = num_control_points_open(n_knots, degree)
        return np.linspace(0, 1, n_knots, dtype=np.float64), np.zeros(n_control_points)

    @property
    def domain(self) -> tuple[float, float]:
        return self.knots[self.degree].item(), self.knots[-self.degree - 1].item()


class ClampedBSpline(BSpline):
    @staticmethod
    def _extrapolation() -> Extrapolation:
        return Extrapolation.NONE

    @staticmethod
    def _pad_knots(knots: Iterable[float], degree: int) -> FloatArray:
        return np.pad(np.array(knots), (degree, degree), mode="edge")

    @staticmethod
    def _pad_control_points(control_points: FloatArray, degree: int) -> FloatArray:
        return np.array(control_points)

    @property
    def boundary_condition(self) -> BoundaryCondition:
        return BoundaryCondition.CLAMPED

    @staticmethod
    def _empty_data(degree: int) -> tuple[FloatArray, FloatArray]:
        n_knots = 2 * degree + 2
        n_control_points = num_control_points_clamped(n_knots, degree)
        return np.linspace(0, 1, n_knots, dtype=np.float64), np.zeros(n_control_points)

    @property
    def domain(self) -> tuple[float, float]:
        return self.knots[0].item(), self.knots[-1].item()


class PeriodicBSpline(BSpline):
    @staticmethod
    def _extrapolation() -> Extrapolation:
        return Extrapolation.PERIODIC

    @staticmethod
    def _pad_knots(knots: Iterable[float], degree: int) -> FloatArray:
        knots = np.array(knots)
        period = knots[-1] - knots[0]
        pad_left = knots[-degree - 1 : -1] - period
        pad_right = knots[1 : degree + 1] + period
        return np.concatenate((pad_left, knots, pad_right))

    @staticmethod
    def _pad_control_points(control_points: FloatArray, degree: int) -> FloatArray:
        return np.pad(control_points, (0, degree), mode="wrap")

    @property
    def boundary_condition(self) -> BoundaryCondition:
        return BoundaryCondition.PERIODIC

    @staticmethod
    def _empty_data(degree: int) -> tuple[FloatArray, FloatArray]:
        n_knots = 2 * degree + 2
        n_control_points = num_control_points_periodic(n_knots, degree)
        return np.linspace(0, 1, n_knots, dtype=np.float64), np.zeros(n_control_points)

    @property
    def domain(self) -> tuple[float, float]:
        return -np.inf, np.inf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reference data for the De Boor algorithm.")
    parser.add_argument("--knots", type=float, nargs="+", help="The knots vector.", required=True)
    parser.add_argument("--control_points", type=float, nargs="+", help="The control points vector.", required=True)
    parser.add_argument("--degree", type=int, help="The degree of the spline.", required=True)
    parser.add_argument(
        "--type",
        choices=[t.value for t in BoundaryCondition],
        help="The type of the spline",
        default=BoundaryCondition.OPEN.value,
    )
    parser.add_argument(
        "--extrapolate",
        choices=[e.value for e in Extrapolation],
        help="The type of extrapolation to use",
        default=Extrapolation.NONE.value,
    )
    parser.add_argument("--plot", action="store_true", help="Whether to plot the reference data or not.")
    parser.add_argument(
        "--print",
        choices=["C++", "Python"],
        help="Print the reference data in the specified language.",
    )
    return parser.parse_args()


BSPLINE_TYPE_MAP = {
    BoundaryCondition.OPEN: OpenBSpline,
    BoundaryCondition.CLAMPED: ClampedBSpline,
    BoundaryCondition.PERIODIC: PeriodicBSpline,
}

X_VALUES_GENERATOR_MAP = {
    BoundaryCondition.OPEN: x_values_open,
    BoundaryCondition.CLAMPED: x_values_clamped,
    BoundaryCondition.PERIODIC: x_values_periodic,
}


def main() -> None:
    args = parse_args()
    bspline_type = BoundaryCondition(args.type)
    spline_cls = BSPLINE_TYPE_MAP[bspline_type]

    x_values_generator = X_VALUES_GENERATOR_MAP[bspline_type]

    spline = spline_cls.from_data(args.knots, args.control_points, args.degree)

    x_values = x_values_generator(args.knots, args.degree)
    y_values = spline.evaluate(x_values)

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(x_values, np.nan_to_num(y_values))
        plt.grid()
        plt.show()

    match args.print:
        case "Python":
            print(f"knots = {args.knots}")
            print(f"control_points = {args.control_points}")
            print(f"x_values = {x_values}")
            print(f"y_values = {y_values}")
        case "C++":
            print(f"knots = {{{', '.join([str(k) for k in args.knots])}}};")
            print(f"control_points = {{{', '.join([str(c) for c in args.control_points])}}};")
            print(f"std::vector<double> x_values{{{', '.join([str(x) for x in x_values])}}};")
            print(f"std::vector<double> y_values{{{', '.join([str(y) for y in y_values])}}};")


if __name__ == "__main__":
    main()
