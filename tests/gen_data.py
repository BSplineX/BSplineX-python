import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

from reference import Curve, BoundaryCondition, OpenBSpline, BSpline
from utils import FloatArray, num_control_points_open

DENSE_NUM_KNOTS = 50
SPARSE_NUM_KNOTS = 1000


@dataclass
class BSplineData:
    boundary_condition: BoundaryCondition
    curve: Curve
    degree: int
    knots: list[float]
    ctrl: list[float]
    domain: tuple[float, float]
    y_eval: list[float]


@dataclass
class TestData:
    x: list[float]  # used to fit and to interpolate
    y: list[float]  # used to fit and to interpolate
    x_eval: list[float]  # used to eval
    conditions_interp: tuple[list[tuple[int, float]], list[tuple[int, float]]]
    bspline: BSplineData
    derivatives: list[BSplineData]
    bspline_fit: BSplineData
    bspline_interp: BSplineData


def get_bspline_data(bspline: BSpline, curve: Curve, x_eval: FloatArray) -> BSplineData:
    return BSplineData(
        boundary_condition=bspline.boundary_condition,
        curve=curve,
        degree=bspline.degree,
        knots=bspline.knots.tolist(),
        ctrl=bspline.control_points.tolist(),
        domain=bspline.domain,
        y_eval=bspline.evaluate(x_eval).tolist(),
    )


def get_additional_conditions(
    num: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    num_left = num // 2
    num_right = num - num_left

    left = [(i + 1, 0.0) for i in range(num_left)]
    right = [(i + 1, 0.0) for i in range(num_right)]

    return left, right


def get_sorted_array(
    rng: np.random.Generator,
    curve: Curve,
    start: float,
    end: float,
    size: int,
) -> FloatArray:
    x = np.linspace(start, end, size, dtype=np.float64)
    if curve == Curve.NON_UNIFORM:
        # NOTE: we do in this way because knots must not be too close to each other
        # for scipy to work properly
        noise = (end - start) / size * 0.3
        x += rng.uniform(-noise, noise, size)
        x.sort()

    return x


def get_x_y_sin(rng: np.random.Generator, curve: Curve, num_points: int) -> tuple[FloatArray, FloatArray]:
    x = get_sorted_array(rng, curve, 0, 2 * np.pi, num_points)
    y = np.sin(x)

    return x, y


def make_open_data(rng: np.random.Generator, degree: int, curve: Curve, num_knots: int) -> TestData:
    print(f"Generating test data for degree={degree}, curve={curve}, num_knots={num_knots}")
    x, y = get_x_y_sin(rng, curve, num_knots * 2)
    knots = get_sorted_array(rng, curve, x[0].item(), x[-1].item(), num_knots)
    ctrl = rng.uniform(size=num_control_points_open(num_knots, degree))
    bspline = OpenBSpline.from_data(knots, ctrl, degree)
    domain_left, domain_right = bspline.domain

    x_eval = rng.uniform(domain_left, domain_right, 10 * num_knots)

    bspline_fit = bspline.copy()
    mask = (x >= domain_left) & (x < domain_right)
    bspline_fit.fit(x[mask], y[mask])

    conditions_interp = get_additional_conditions(degree - 1)
    bspline_interp = OpenBSpline.empty(degree)
    bspline_interp.interpolate(x, y, conditions_interp)

    return TestData(
        x=x.tolist(),
        y=y.tolist(),
        x_eval=x_eval.tolist(),
        conditions_interp=conditions_interp,
        bspline=get_bspline_data(bspline, curve, x_eval),
        derivatives=[get_bspline_data(bspline.derivative(i), curve, x_eval) for i in range(1, degree + 1)],
        bspline_fit=get_bspline_data(bspline_fit, curve, x_eval),
        bspline_interp=get_bspline_data(bspline_interp, curve, x_eval),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        "Generate a JSON with test data for open, clamped, and periodic uniform/non-uniform BSplines"
    )
    parser.add_argument("--degrees", required=True, nargs="+", type=int, help="list[int] BSpline degrees")
    parser.add_argument("--output-dir", required=True, type=str, help="str output directory")

    return parser.parse_args()


def main():
    args = parse_args()

    rng = np.random.default_rng(42)
    os.makedirs(args.output_dir, exist_ok=True)

    for bc, gen_fn in [(BoundaryCondition.OPEN, make_open_data)]:
        data = []
        for degree in args.degrees:
            data.extend(
                [
                    make_open_data(rng, degree, Curve.UNIFORM, DENSE_NUM_KNOTS),
                    make_open_data(rng, degree, Curve.NON_UNIFORM, DENSE_NUM_KNOTS),
                    make_open_data(rng, degree, Curve.UNIFORM, SPARSE_NUM_KNOTS),
                    make_open_data(rng, degree, Curve.NON_UNIFORM, SPARSE_NUM_KNOTS),
                ]
            )

        with open(os.path.join(args.output_dir, f"{bc.value}.json"), "w") as f:
            json.dump(data, f, default=vars)


if __name__ == "__main__":
    main()
