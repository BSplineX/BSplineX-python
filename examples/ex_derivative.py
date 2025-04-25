import BSplineX as bs
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    xx = np.linspace(0, 2 * np.pi, 10000)
    yy = np.sin(xx)

    bspline: bs.PeriodicUniform = bs.make_periodic_uniform(3)
    bspline.interpolate(xx, yy, [])

    eval_x = xx
    eval_y = np.array(bspline.evaluate(eval_x))
    eval_y_d1 = np.array(bspline.evaluate(eval_x, 1))
    eval_y_d2 = np.array(bspline.evaluate(eval_x, 2))
    eval_y_d3 = np.array(bspline.evaluate(eval_x, 3))

    try:
        bspline.evaluate(eval_x, 4)
        assert False
    except RuntimeError:
        print("4th derivative correctly raised exception")

    assert np.allclose(eval_y**2 + eval_y_d1**2, 1.0)
    assert np.allclose(eval_y + eval_y_d2, 0.0, atol=1e-7)

    plt.figure()
    plt.plot(eval_x, eval_y, label="f(x) = sin(x)")
    plt.plot(eval_x, eval_y_d1, label="f'(x) = cos(x)")
    plt.plot(eval_x, eval_y_d2, label="f''(x) = -sin(x)")
    plt.plot(eval_x, eval_y_d3, label="f'''(x) = -cos(x)")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
