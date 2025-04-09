import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Type

import numpy as np
from reference import BoundaryCondition, BSpline, ClampedBSpline, Curve, OpenBSpline, PeriodicBSpline
from utils import FloatArray

DENSE_NUM_KNOTS = 50
SPARSE_NUM_KNOTS = 1000


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)


@dataclass
class BSplineData:
    boundary_condition: BoundaryCondition
    curve: Curve
    degree: int
    knots: FloatArray
    ctrl: FloatArray
    domain: tuple[float, float]
    y_eval: FloatArray
    # for each derivative degree (starting from zero):
    #   for each x_eval:
    #       the values of degree+1 non-zero basis, given as a tuple (start_index, list of values)
    nnz_basis: list[list[tuple[int, FloatArray]]]


@dataclass
class TestData:
    x: FloatArray  # used to fit and to interpolate
    y: FloatArray  # used to fit and to interpolate
    x_eval: FloatArray  # used to eval
    conditions_interp: tuple[list[tuple[int, float]], list[tuple[int, float]]]
    bspline: BSplineData
    derivatives: list[BSplineData]
    bspline_fit: BSplineData
    bspline_interp: BSplineData


def get_bspline_data(bspline: BSpline, curve: Curve, x_eval: FloatArray) -> BSplineData:
    domain_left, domain_right = bspline.domain
    mask = (x_eval >= domain_left) & (x_eval <= domain_right)
    x_nnz = x_eval[mask]

    return BSplineData(
        boundary_condition=bspline.boundary_condition,
        curve=curve,
        degree=bspline.degree,
        knots=bspline.knots,
        ctrl=bspline.control_points,
        domain=bspline.domain,
        y_eval=bspline.evaluate(x_eval),
        nnz_basis=[bspline.nnz_basis(x_nnz, derivative_order=i) for i in range(bspline.degree + 1)],
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
        noise = (end - start) / size * 0.2
        # noise = 0
        x += rng.uniform(-noise, noise, size)
        x = x.clip(min=start, max=end)
        x.sort()

    x[0] = start
    x[-1] = end

    return x


def get_x_y_sin(rng: np.random.Generator, curve: Curve, num_points: int) -> tuple[FloatArray, FloatArray]:
    x = get_sorted_array(rng, curve, 0, 2 * np.pi, num_points)
    y = np.sin(x)

    return x, y


def make_bspline_data(
    bspline_cls: Type[BSpline], rng: np.random.Generator, degree: int, curve: Curve, num_knots: int
) -> TestData:
    print(
        f"Generating test data for {bspline_cls().boundary_condition}, degree={degree}, curve={curve}, num_knots={num_knots}"
    )
    x, y = get_x_y_sin(rng, curve, num_knots * 2)
    knots = get_sorted_array(rng, curve, x[0].item(), x[-1].item(), num_knots)
    ctrl = rng.uniform(size=bspline_cls.required_control_points(num_knots, degree))
    bspline = bspline_cls.from_data(knots, ctrl, degree)
    domain_left, domain_right = bspline.domain
    edomain_left, edomain_right = bspline.eval_domain
    x_eval = rng.uniform(edomain_left, edomain_right, 10 * num_knots)

    bspline_fit = bspline.copy()
    mask = (x >= domain_left) & (x <= domain_right)
    bspline_fit.fit(x[mask], y[mask])

    conditions_interp = get_additional_conditions(bspline_cls.required_additional_conditions(degree))
    bspline_interp = bspline_cls.empty(degree)
    bspline_interp.interpolate(x, y, conditions_interp)

    return TestData(
        x=x,
        y=y,
        x_eval=x_eval,
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

    for curve, bspline_cls in [
        (Curve.UNIFORM, OpenBSpline),
        (Curve.NON_UNIFORM, OpenBSpline),
        (Curve.UNIFORM, ClampedBSpline),
        (Curve.NON_UNIFORM, ClampedBSpline),
        (Curve.UNIFORM, PeriodicBSpline),
        (Curve.NON_UNIFORM, PeriodicBSpline),
    ]:
        data = []
        for degree in args.degrees:
            data.extend(
                [
                    asdict(make_bspline_data(bspline_cls, rng, degree, curve, DENSE_NUM_KNOTS)),
                    asdict(make_bspline_data(bspline_cls, rng, degree, curve, SPARSE_NUM_KNOTS)),
                ]
            )
        bc = bspline_cls().boundary_condition
        p = os.path.join(args.output_dir, bc.value)
        os.makedirs(p, exist_ok=True)

        with open(os.path.join(p, f"{curve.value}.json"), "w") as f:
            json.dump(data, f, cls=NpEncoder)


if __name__ == "__main__":
    main()
