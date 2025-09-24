import math
import operator
from collections.abc import Callable, Iterable
from functools import reduce

import numpy as np
import pytest
from sleipnir import autodiff
from sleipnir.autodiff import Gradient, Hessian, Jacobian, Variable, VariableMatrix


def prod[T](iterable: Iterable[T]) -> T:
    return reduce(operator.mul, iterable)


def test_linear() -> None:
    # y = x
    x = VariableMatrix(1)
    x[0].set_value(3)
    y = x[0]

    # dy/dx = 1
    g = Gradient(y, x[0]).value()[0, 0]
    assert g == 1.0

    # d²y/dx² = 0
    H = Hessian(y, x)
    assert H.get().value()[0, 0] == 0.0
    assert H.value()[0, 0] == 0.0


def test_quadratic() -> None:
    # y = x²
    x = VariableMatrix(1)
    x[0].set_value(3)
    y = x[0] * x[0]

    # dy/dx = 2x = 6
    g = Gradient(y, x[0]).value()[0, 0]
    assert g == 6.0

    # d²y/dx² = 2
    H = Hessian(y, x)
    assert H.get().value()[0, 0] == 2.0
    assert H.value()[0, 0] == 2.0


def test_cubic() -> None:
    # y = x³
    x = VariableMatrix(1)
    x[0].set_value(3)
    y = x[0] * x[0] * x[0]

    # dy/dx = 3x² = 27
    g = Gradient(y, x[0]).value()[0, 0]
    assert g == 27.0

    # d²y/dx² = 6x = 18
    H = Hessian(y, x)
    assert H.get().value()[0, 0] == 18.0
    assert H.value()[0, 0] == 18.0


def test_quartic() -> None:
    # y = x⁴
    x = VariableMatrix(1)
    x[0].set_value(3)
    y = x[0] * x[0] * x[0] * x[0]

    # dy/dx = 4x³ = 108
    g = Gradient(y, x[0]).value()[0, 0]
    assert g == 108.0

    # d²y/dx² = 12x² = 108
    H = Hessian(y, x)
    assert H.get().value()[0, 0] == 108.0
    assert H.value()[0, 0] == 108.0


def test_sum() -> None:
    y = Variable()
    x = VariableMatrix(5)
    for i in range(5):
        x[i].set_value(i + 1)

    y = sum(x, Variable(0.0))
    assert y.value() == 15.0

    g = Gradient(y, x)
    assert (g.get().value() == np.full((5, 1), 1.0)).all()
    assert (g.value().todense() == np.full((5, 1), 1.0)).all()

    H = Hessian(y, x)
    assert (H.get().value() == np.zeros((5, 5))).all()
    assert (H.value().todense() == np.zeros((5, 5))).all()


def test_sum_of_products() -> None:
    x = VariableMatrix(5)
    for i in range(5):
        x[i].set_value(i + 1)

    # y = ||x||²
    y = (x.T @ x)[0, 0]
    assert y.value() == sum([x * x for x in range(1, 6)])

    g = Gradient(y, x)
    assert (g.get().value() == 2 * x.value()).all()
    assert (g.value().todense() == 2 * x.value()).all()

    H = Hessian(y, x)

    expected_H = np.diag([2.0] * 5)
    assert (H.get().value() == expected_H).all()
    assert (H.value().todense() == expected_H).all()


def test_product_of_sines() -> None:
    x = VariableMatrix(5)
    for i in range(5):
        x[i].set_value(i + 1)

    # y = prod(sin(x))
    y = prod(autodiff.sin(x))
    assert y.value() == pytest.approx(prod(math.sin(x) for x in range(1, 6)), abs=1e-15)

    g = Gradient(y, x)
    for i in range(x.rows()):
        assert g.get().value()[i, 0] == pytest.approx(
            y.value() / math.tan(x[i].value()), abs=1e-15
        )
        assert g.value()[i, 0] == pytest.approx(
            y.value() / math.tan(x[i].value()), abs=1e-15
        )

    H = Hessian(y, x)

    expected_H = np.empty((5, 5))
    for i in range(x.rows()):
        for j in range(x.rows()):
            if i == j:
                expected_H[i, j] = -y.value()
            else:
                expected_H[i, j] = y.value() / (
                    math.tan(x[i].value()) * math.tan(x[j].value())
                )
    np.testing.assert_allclose(H.get().value(), expected_H)
    np.testing.assert_allclose(H.value().todense(), expected_H)


def test_sum_of_squared_residuals() -> None:
    x = VariableMatrix(5)
    for i in range(5):
        x[i].set_value(1)

    # y = sum(diff(x).^2)
    y = sum((x[:4, :1] - x[1:5, :1]) ** 2, Variable(0.0))
    g = Gradient(y, x).value()

    assert y.value() == 0.0
    assert g[0, 0] == 2 * x[0].value() - 2 * x[1].value()
    assert g[1, 0] == -2 * x[0].value() + 4 * x[1].value() - 2 * x[2].value()
    assert g[2, 0] == -2 * x[1].value() + 4 * x[2].value() - 2 * x[3].value()
    assert g[3, 0] == -2 * x[2].value() + 4 * x[3].value() - 2 * x[4].value()
    assert g[4, 0] == -2 * x[3].value() + 2 * x[4].value()

    H = Hessian(y, x)

    expected_H = np.array(
        [
            [2.0, -2.0, 0.0, 0.0, 0.0],
            [-2.0, 4.0, -2.0, 0.0, 0.0],
            [0.0, -2.0, 4.0, -2.0, 0.0],
            [0.0, 0.0, -2.0, 4.0, -2.0],
            [0.0, 0.0, 0.0, -2.0, 2.0],
        ]
    )
    assert (H.get().value() == expected_H).all()
    assert (H.value().todense() == expected_H).all()


def test_sum_of_squares() -> None:
    r = VariableMatrix(4)
    r.set_value(np.array([[25.0], [10.0], [5.0], [0.0]]))

    x = VariableMatrix(4)
    for i in range(4):
        x[i].set_value(0.0)

    J = sum(((r[i] - x[i]) * (r[i] - x[i]) for i in range(4)), Variable(0.0))
    H = Hessian(J, x)

    expected_H = np.diag([2.0] * 4)
    assert (H.get().value() == expected_H).all()
    assert (H.value().todense() == expected_H).all()


def test_nested_powers() -> None:
    x0 = 3.0

    x = Variable()
    x.set_value(x0)

    y = (x**2) ** 2

    J = Jacobian(y, x).value()
    assert J[0, 0] == pytest.approx(4 * x0 * x0 * x0, abs=1e-12)

    H = Hessian(y, x).value()
    assert H[0, 0] == pytest.approx(12 * x0 * x0, abs=1e-12)


def check_unary_hessian(
    f: Callable[[Variable], Variable],
    df: Callable[[float], float],
    d2f: Callable[[float], float],
) -> None:
    # y = f(u) where u = x₀x₁
    #
    # ∂²y/∂x₀² = f″(u) x₁²
    # ∂²y/∂x₀∂x₁ = f″(u) x₀x₁ + f′(u)
    # ∂²y/∂x₁² = f″(u) x₀²
    x = VariableMatrix(2)
    x[0].set_value(0.6)
    x[1].set_value(0.7)
    hessian = Hessian(f(x[0] * x[1]), x)
    symbolic_H = hessian.get()

    # Check symbolic Hessian at a point other than where it was built too
    for x0, x1 in [(0.6, 0.7), (0.3, 0.7)]:
        x[0].set_value(x0)
        x[1].set_value(x1)

        u = x0 * x1
        mixed = d2f(u) * u + df(u)
        expected = np.array([[d2f(u) * x1 * x1, mixed], [mixed, d2f(u) * x0 * x0]])
        np.testing.assert_allclose(symbolic_H.value(), expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            hessian.value().todense(), expected, rtol=0.0, atol=1e-12
        )


def test_division() -> None:
    # y = x₀/x₁
    #
    # ∂²y/∂x₀² = 0
    # ∂²y/∂x₀∂x₁ = −1/x₁²
    # ∂²y/∂x₁² = 2x₀/x₁³
    x = VariableMatrix(2)
    hessian = Hessian(x[0] / x[1], x)
    symbolic_H = hessian.get()

    for x0, x1 in [(3.0, 2.0), (-1.5, 0.5)]:
        x[0].set_value(x0)
        x[1].set_value(x1)

        mixed = -1.0 / (x1 * x1)
        expected = np.array([[0.0, mixed], [mixed, 2.0 * x0 / (x1 * x1 * x1)]])
        np.testing.assert_allclose(symbolic_H.value(), expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            hessian.value().todense(), expected, rtol=0.0, atol=1e-12
        )


def test_unary_minus() -> None:
    check_unary_hessian(lambda u: -u, lambda u: -1.0, lambda u: 0.0)


def test_abs() -> None:
    check_unary_hessian(autodiff.abs, lambda u: 1.0, lambda u: 0.0)


def test_acos() -> None:
    check_unary_hessian(
        autodiff.acos,
        lambda u: -1.0 / math.sqrt(1.0 - u * u),
        lambda u: -u / math.sqrt(1.0 - u * u) ** 3,
    )


def test_asin() -> None:
    check_unary_hessian(
        autodiff.asin,
        lambda u: 1.0 / math.sqrt(1.0 - u * u),
        lambda u: u / math.sqrt(1.0 - u * u) ** 3,
    )


def test_atan() -> None:
    check_unary_hessian(
        autodiff.atan,
        lambda u: 1.0 / (1.0 + u * u),
        lambda u: -2.0 * u / ((1.0 + u * u) * (1.0 + u * u)),
    )


def test_atan2() -> None:
    # z = atan2(y, x)
    #
    # ∂²z/∂y² = −2xy/(x² + y²)²
    # ∂²z/∂y∂x = (y² − x²)/(x² + y²)²
    # ∂²z/∂x² = 2xy/(x² + y²)²
    input = VariableMatrix(2)
    hessian = Hessian(autodiff.atan2(input[0], input[1]), input)
    symbolic_H = hessian.get()

    for y, x in [(1.5, 0.5), (-0.7, 1.3)]:
        input[0].set_value(y)
        input[1].set_value(x)

        d = x * x + y * y
        mixed = (y * y - x * x) / (d * d)
        expected = np.array(
            [[-2.0 * x * y / (d * d), mixed], [mixed, 2.0 * x * y / (d * d)]]
        )
        np.testing.assert_allclose(symbolic_H.value(), expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            hessian.value().todense(), expected, rtol=0.0, atol=1e-12
        )


def test_cbrt() -> None:
    check_unary_hessian(
        autodiff.cbrt,
        lambda u: 1.0 / (3.0 * math.cbrt(u) * math.cbrt(u)),
        lambda u: -2.0 / (9.0 * u * math.cbrt(u) * math.cbrt(u)),
    )


def test_cos() -> None:
    check_unary_hessian(autodiff.cos, lambda u: -math.sin(u), lambda u: -math.cos(u))


def test_cosh() -> None:
    check_unary_hessian(autodiff.cosh, math.sinh, math.cosh)


def test_erf() -> None:
    check_unary_hessian(
        autodiff.erf,
        lambda u: 2.0 / math.sqrt(math.pi) * math.exp(-u * u),
        lambda u: -4.0 / math.sqrt(math.pi) * u * math.exp(-u * u),
    )


def test_exp() -> None:
    check_unary_hessian(autodiff.exp, math.exp, math.exp)


def hypot_expected_H(v: np.ndarray) -> np.ndarray:
    # y = ‖v‖
    #
    # ∂²y/∂vᵢ∂vⱼ = (‖v‖²δᵢⱼ − vᵢvⱼ)/‖v‖³
    r2 = v @ v
    return (r2 * np.eye(len(v)) - np.outer(v, v)) / (r2 * math.sqrt(r2))


def test_hypot_two_arguments() -> None:
    x = VariableMatrix(2)
    hessian = Hessian(autodiff.hypot(x[0], x[1]), x)
    symbolic_H = hessian.get()

    for x0, x1 in [(1.3, 2.3), (-0.5, 1.5)]:
        x[0].set_value(x0)
        x[1].set_value(x1)

        expected = hypot_expected_H(np.array([x0, x1]))
        np.testing.assert_allclose(symbolic_H.value(), expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            hessian.value().todense(), expected, rtol=0.0, atol=1e-12
        )


def test_hypot_three_arguments() -> None:
    x = VariableMatrix(3)
    hessian = Hessian(autodiff.hypot(x[0], x[1], x[2]), x)
    symbolic_H = hessian.get()

    for v in [np.array([1.3, 2.3, 3.3]), np.array([-0.5, 1.5, -2.5])]:
        x[0].set_value(v[0])
        x[1].set_value(v[1])
        x[2].set_value(v[2])

        expected = hypot_expected_H(v)
        np.testing.assert_allclose(symbolic_H.value(), expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            hessian.value().todense(), expected, rtol=0.0, atol=1e-12
        )


def test_hypot_repeated_argument() -> None:
    # y = hypot(a, b, a) = √(2a² + b²)
    #
    # ∂²y/∂a² = (2y² − 4a²)/y³
    # ∂²y/∂a∂b = −2ab/y³
    # ∂²y/∂b² = (y² − b²)/y³
    x = VariableMatrix(2)
    hessian = Hessian(autodiff.hypot(x[0], x[1], x[0]), x)
    symbolic_H = hessian.get()

    for a, b in [(1.3, 2.3), (-0.5, 1.5)]:
        x[0].set_value(a)
        x[1].set_value(b)

        y2 = 2.0 * a * a + b * b
        y3 = y2 * math.sqrt(y2)
        mixed = -2.0 * a * b / y3
        expected = np.array(
            [[(2.0 * y2 - 4.0 * a * a) / y3, mixed], [mixed, (y2 - b * b) / y3]]
        )
        np.testing.assert_allclose(symbolic_H.value(), expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            hessian.value().todense(), expected, rtol=0.0, atol=1e-12
        )


def test_log() -> None:
    check_unary_hessian(autodiff.log, lambda u: 1.0 / u, lambda u: -1.0 / (u * u))


def test_log10() -> None:
    check_unary_hessian(
        autodiff.log10,
        lambda u: 1.0 / (math.log(10.0) * u),
        lambda u: -1.0 / (math.log(10.0) * u * u),
    )


def test_max_and_min() -> None:
    input = VariableMatrix(2)
    expected_y = np.array([[0.0, 0.0], [0.0, 2.0]])
    expected_x = np.array([[2.0, 0.0], [0.0, 0.0]])

    def check(f: Variable, x0: float, y0: float, x1: float, y1: float) -> None:
        hessian = Hessian(f, input)
        symbolic_H = hessian.get()

        input[0].set_value(x0)
        input[1].set_value(y0)
        np.testing.assert_array_equal(symbolic_H.value(), expected_y)
        np.testing.assert_array_equal(hessian.value().todense(), expected_y)

        input[0].set_value(x1)
        input[1].set_value(y1)
        np.testing.assert_array_equal(symbolic_H.value(), expected_x)
        np.testing.assert_array_equal(hessian.value().todense(), expected_x)

        input[0].set_value(2.0)
        input[1].set_value(2.0)
        np.testing.assert_array_equal(symbolic_H.value(), expected_x)
        np.testing.assert_array_equal(hessian.value().todense(), expected_x)

    x_squared = input[0] * input[0]
    y_squared = input[1] * input[1]
    check(autodiff.max(x_squared, y_squared), 1.0, 2.0, 3.0, 2.0)
    check(autodiff.min(x_squared, y_squared), 2.0, 1.0, 2.0, 3.0)


def test_pow() -> None:
    input = VariableMatrix(2)
    f = autodiff.pow(input[0] + 1.0, input[1])
    hessian = Hessian(f, input)

    symbolic_H = hessian.get()

    input[0].set_value(2.0)
    input[1].set_value(3.0)
    mixed = 9.0 * (1.0 + 3.0 * math.log(3.0))
    expected = np.array([[18.0, mixed], [mixed, 27.0 * math.log(3.0) ** 2]])
    np.testing.assert_allclose(symbolic_H.value(), expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        hessian.value().todense(), expected, rtol=0.0, atol=1e-12
    )

    input[0].set_value(3.0)
    input[1].set_value(2.0)
    mixed = 4.0 * (1.0 + 2.0 * math.log(4.0))
    expected = np.array([[2.0, mixed], [mixed, 16.0 * math.log(4.0) ** 2]])
    np.testing.assert_allclose(symbolic_H.value(), expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        hessian.value().todense(), expected, rtol=0.0, atol=1e-12
    )


def test_sign() -> None:
    check_unary_hessian(autodiff.sign, lambda u: 0.0, lambda u: 0.0)


def test_sin() -> None:
    check_unary_hessian(autodiff.sin, math.cos, lambda u: -math.sin(u))


def test_sinh() -> None:
    check_unary_hessian(autodiff.sinh, math.cosh, math.sinh)


def test_sqrt() -> None:
    check_unary_hessian(
        autodiff.sqrt,
        lambda u: 1.0 / (2.0 * math.sqrt(u)),
        lambda u: -1.0 / (4.0 * u * math.sqrt(u)),
    )


def test_tan() -> None:
    check_unary_hessian(
        autodiff.tan,
        lambda u: 1.0 / (math.cos(u) * math.cos(u)),
        lambda u: 2.0 * math.tan(u) / (math.cos(u) * math.cos(u)),
    )


def test_tanh() -> None:
    check_unary_hessian(
        autodiff.tanh,
        lambda u: 1.0 / (math.cosh(u) * math.cosh(u)),
        lambda u: -2.0 * math.tanh(u) / (math.cosh(u) * math.cosh(u)),
    )


def test_rosenbrock() -> None:
    # z = (1 − x)² + 100(y − x²)²
    #   = 100(−x² + y)² + (−x + 1)²
    #
    # ∂z/∂x = 200(−x² + y)⋅−2x + 2(−x + 1)⋅−1
    #       = −400x(−x² + y) − 2(−x + 1)
    #       = 400x³ − 400xy + 2x − 2
    #
    # ∂z/∂y = 200(−x² + y)
    #
    # ∂²z/∂x² = 1200x² − 400y + 2
    # ∂²z/∂xy = −400x
    # ∂²z/∂y² = 200

    input = VariableMatrix(2)
    x = input[0]
    y = input[1]
    hessian = Hessian((1 - x) ** 2 + 100 * (y - x**2) ** 2, input)

    for x0 in np.arange(-2.5, 2.5, 0.1):
        for y0 in np.arange(-2.5, 2.5, 0.1):
            x.set_value(x0)
            y.set_value(y0)

            H = hessian.value().todense()
            assert H[0, 0] == pytest.approx(1200 * x0**2 - 400 * y0 + 2, abs=1e-11)
            assert H[0, 1] == -400 * x0
            assert H[1, 0] == -400 * x0
            assert H[1, 1] == 200


def test_edge_pushing_wang_example_1() -> None:
    # See example 1 of [1]
    #
    # [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
    #     efficient Hessian computation via automatic differentiation", 2016.
    #     https://sci-hub.st/10.1007/s12532-016-0100-3

    x = VariableMatrix(2)
    x[0].set_value(3)
    x[1].set_value(4)

    # y = (x₀sin(x₁)) x₀
    y = (x[0] * autodiff.sin(x[1])) * x[0]

    # dy/dx = [2x₀sin(x₁)  x₀²cos(x₁)]
    # dy/dx = [ 6sin(4)     9cos(4)  ]
    J = Jacobian(y, x)
    expected_J = np.array([[6.0 * math.sin(4.0), 9.0 * math.cos(4.0)]])
    assert (J.get().value() == expected_J).all()
    assert (J.value().todense() == expected_J).all()

    #           [ 2sin(x₁)    2x₀cos(x₁)]
    # d²y/dx² = [2x₀cos(x₁)  −x₀²sin(x₁)]
    #
    #           [2sin(4)   6cos(4)]
    # d²y/dx² = [6cos(4)  −9sin(4)]
    H = Hessian(y, x)
    expected_H = np.array(
        [
            [2.0 * math.sin(4.0), 6.0 * math.cos(4.0)],
            [6.0 * math.cos(4.0), -9.0 * math.sin(4.0)],
        ]
    )
    assert (H.get().value() == expected_H).all()
    assert (H.value().todense() == expected_H).all()


def test_edge_pushing_petro_figure_1() -> None:
    # See figure 1 of [1]
    #
    # [1] Petro, C. G., et al. "On efficient Hessian computation using the edge
    #     pushing algorithm in Julia", 2017.
    #     https://mlubin.github.io/pdf/edge_pushing_julia.pdf

    # y = p₁ log(x₁x₂)
    p_1 = Variable(2.0)
    x = VariableMatrix(2)
    x[0].set_value(2)
    x[1].set_value(3)
    y = p_1 * autodiff.log(x[0] * x[1])

    # d²y/dx² = [−p₁/x₁²     0   ]
    #           [   0     −p₁/x₂²]
    H = Hessian(y, x)
    expected_H = np.diag(
        [-p_1.value() / x[0].value() ** 2, -p_1.value() / x[1].value() ** 2]
    )
    assert (H.get().value() == expected_H).all()
    assert (H.value().todense() == expected_H).all()


def test_linear_node_elimination() -> None:
    x = VariableMatrix(3)

    # Each term exercises a different elimination case:
    #
    #   sin(x₀ + x₁ − x₂): chain of sums with one parent each
    #   exp(2x₁) cos(2x₁): constant multiple with two parents
    #   (x₀ + x₂)²: sum with two parents and two children (not eliminated)
    #   (3x₂)²: eliminated node as both arguments of a product
    #   exp(x₀ − x₀ + x₁): expansion terms that cancel
    #   log(x₁ + x₀/4): quotient by a constant
    two_x1 = 2.0 * x[1]
    x0_plus_x2 = x[0] + x[2]
    three_x2 = 3.0 * x[2]
    f = (
        autodiff.sin(x[0] + x[1] - x[2])
        + autodiff.exp(two_x1) * autodiff.cos(two_x1)
        + x0_plus_x2 * x0_plus_x2
        + three_x2 * three_x2
        + autodiff.exp(x[0] - x[0] + x[1])
        + autodiff.log(x[1] + x[0] / 4.0)
    )

    hessian = Hessian(f, x)

    for point in [(1.0, 2.0, 3.0), (-0.5, 0.25, 1.5)]:
        x.set_value(np.array(point).reshape(3, 1))
        x0, x1, x2 = point

        v = np.array([1.0, 1.0, -1.0])
        w = np.array([0.25, 1.0, 0.0])
        s = x1 + x0 / 4.0

        expected_H = -math.sin(x0 + x1 - x2) * np.outer(v, v) - np.outer(w, w) / (s * s)
        expected_H[1, 1] += -8.0 * math.exp(2.0 * x1) * math.sin(2.0 * x1)
        expected_H[1, 1] += math.exp(x1)
        expected_H[0, 0] += 2.0
        expected_H[0, 2] += 2.0
        expected_H[2, 0] += 2.0
        expected_H[2, 2] += 2.0 + 18.0

        np.testing.assert_allclose(
            hessian.value().todense(), expected_H, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            hessian.get().value(), expected_H, rtol=0.0, atol=1e-12
        )


def test_constant_contributions() -> None:
    x = VariableMatrix(2)

    # f = x₀x₁ + 3x₁² − x₀x₀ + sin(x₀)
    #
    # The products have constant adjoints and second partial derivatives, so
    # their contributions are constant and folded into the results's initial
    # values. sin(x₀) isn't, so it's recomputed on each evaluation.
    f = x[0] * x[1] + 3.0 * x[1] * x[1] - x[0] * x[0] + autodiff.sin(x[0])

    hessian = Hessian(f, x)
    symbolic_H = hessian.get()

    for x0 in [1.0, -2.0, 0.5]:
        x[0].set_value(x0)
        x[1].set_value(2.0 * x0)

        expected_H = np.array([[-2.0 - math.sin(x0), 1.0], [1.0, 6.0]])
        np.testing.assert_allclose(
            hessian.value().todense(), expected_H, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(symbolic_H.value(), expected_H, rtol=0.0, atol=1e-12)


def test_pow_with_power_of_half() -> None:
    x = VariableMatrix(1)
    x[0].set_value(2.0)

    # pow(x, 0.5) is rewritten as sqrt(x)
    y = autodiff.pow(x[0], 0.5)
    assert y.value() == pytest.approx(2.0**0.5, abs=1e-15)

    # d²/dx² √x = −1/4 x^(−3/2)
    H = Hessian(y, x)
    assert H.value()[0, 0] == pytest.approx(-0.25 * 2.0**-1.5, abs=1e-15)


def test_variable_reuse() -> None:
    y = Variable()
    x = VariableMatrix(1)

    # y = x³
    x[0].set_value(1)
    y = x[0] * x[0] * x[0]

    hessian = Hessian(y, x)

    # d²y/dx² = 6x
    # H = 6
    H = hessian.value()

    assert H.shape == (1, 1)
    assert H[0, 0] == 6.0

    x[0].set_value(2)
    # d²y/dx² = 6x
    # H = 12
    H = hessian.value()

    assert H.shape == (1, 1)
    assert H[0, 0] == 12.0
