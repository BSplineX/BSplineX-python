"""
Reference BSpline implementation.
"""
import argparse
from abc import abstractmethod, ABC
from enum import Enum
from typing import Iterable

import numpy as np
from scipy.interpolate import BSpline as BSpline_, make_lsq_spline

from utils import x_values_open, x_values_clamped, x_values_periodic

DIST_R = 1e-20


class Extrapolation(str, Enum):
    NONE = "none"
    PERIODIC = "periodic"


class BSplineType(str, Enum):
    OPEN = "open"
    CLAMPED = "clamped"
    PERIODIC = "periodic"


class BSpline(ABC):
    def __init__(self, knots: Iterable[float], control_points: Iterable[float], degree: int,
                 extrapolation: Extrapolation) -> None:
        super().__init__()
        knots = self._pad_knots(knots, degree)
        control_points = self._pad_control_points(control_points, degree)
        if not self._supports_extrapolation(extrapolation):
            raise ValueError(f"Invalid extrapolation {extrapolation} for BSpline type {self.__class__}")
        extrapolation = False if extrapolation is Extrapolation.NONE else extrapolation.value
        self.bspline = BSpline_(knots, control_points, degree, extrapolate=extrapolation)

    @property
    def knots(self) -> np.ndarray:
        return self.bspline.t

    @property
    def control_points(self) -> np.ndarray:
        return self.bspline.c

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
        self._basis = [
            BSpline_(self.knots, (np.arange(n) == i).astype(float), self.degree, extrapolate=self.extrapolation)
            for i in range(n)
        ]

    def evaluate(self, x: Iterable[float]) -> np.ndarray:
        return self.bspline(x)

    def fit(self, x: Iterable[float], y: Iterable[float]) -> None:
        self.bspline = make_lsq_spline(x, y, self.knots, self.degree)

    def basis(self, x: float) -> np.ndarray:
        return np.array([b(x) for b in self._basis])

    @staticmethod
    @abstractmethod
    def _supports_extrapolation(extrapolation: Extrapolation) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def _pad_knots(knots: Iterable[float], degree: int) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def _pad_control_points(control_points: Iterable[float], degree: int) -> np.ndarray:
        pass


class OpenBSpline(BSpline):
    @staticmethod
    def _supports_extrapolation(extrapolation: Extrapolation) -> bool:
        return extrapolation is Extrapolation.NONE

    @staticmethod
    def _pad_knots(knots: Iterable[float], degree: int) -> np.ndarray:
        return np.array(knots)

    @staticmethod
    def _pad_control_points(control_points: Iterable[float], degree: int) -> np.ndarray:
        return np.array(control_points)


class ClampedBSpline(BSpline):
    @staticmethod
    def _supports_extrapolation(extrapolation: Extrapolation) -> bool:
        return extrapolation is Extrapolation.NONE

    @staticmethod
    def _pad_knots(knots: Iterable[float], degree: int) -> np.ndarray:
        return np.pad(knots, (degree, degree), mode='edge')

    @staticmethod
    def _pad_control_points(control_points: Iterable[float], degree: int) -> np.ndarray:
        return np.array(control_points)


class PeriodicBSpline(BSpline):
    @staticmethod
    def _supports_extrapolation(extrapolation: Extrapolation) -> bool:
        return extrapolation is Extrapolation.PERIODIC

    @staticmethod
    def _pad_knots(knots: Iterable[float], degree: int) -> np.ndarray:
        knots = np.array(knots)
        period = knots[-1] - knots[0]
        pad_left = knots[-degree - 1: -1] - period
        pad_right = knots[1: degree + 1] + period
        return np.concatenate((pad_left, knots, pad_right))

    @staticmethod
    def _pad_control_points(control_points: Iterable[float], degree: int) -> np.ndarray:
        return np.pad(control_points, (0, degree), mode='wrap')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reference data for the De Boor algorithm.")
    parser.add_argument("--knots", type=float, nargs="+",
                        help="The knots vector.", required=True)
    parser.add_argument("--control_points", type=float, nargs="+",
                        help="The control points vector.", required=True)
    parser.add_argument("--degree", type=int,
                        help="The degree of the spline.", required=True)
    parser.add_argument("--type", choices=[t.value for t in BSplineType],
                        help="The type of the spline", default=BSplineType.OPEN.value)
    parser.add_argument("--extrapolate", choices=[e.value for e in Extrapolation],
                        help="The type of extrapolation to use", default=Extrapolation.NONE.value)
    parser.add_argument("--plot", action="store_true",
                        help="Whether to plot the reference data or not.")
    parser.add_argument("--print", choices=["C++", "Python"],
                        help="Print the reference data in the specified language.")
    return parser.parse_args()


BSPLINE_TYPE_MAP = {
    BSplineType.OPEN: OpenBSpline,
    BSplineType.CLAMPED: ClampedBSpline,
    BSplineType.PERIODIC: PeriodicBSpline
}

X_VALUES_GENERATOR_MAP = {
    BSplineType.OPEN: x_values_open,
    BSplineType.CLAMPED: x_values_clamped,
    BSplineType.PERIODIC: x_values_periodic
}


def main() -> None:
    args = parse_args()
    bspline_type = BSplineType(args.type)
    spline_cls = BSPLINE_TYPE_MAP[bspline_type]

    extrapolation_mode = Extrapolation(args.extrapolate)
    x_values_generator = X_VALUES_GENERATOR_MAP[bspline_type]

    spline = spline_cls(args.knots, args.control_points, args.degree, extrapolation_mode)

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
