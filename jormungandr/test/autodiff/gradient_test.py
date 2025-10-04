import math

import numpy as np
import pytest

import jormungandr.autodiff as autodiff
from jormungandr.autodiff import Gradient, Variable, VariableMatrix


def test_trivial_case():
    a = Variable()
    a.set_value(10)
    b = Variable()
    b.set_value(20)
    c = a

    assert Gradient(a, a).value()[0, 0] == 1.0
    assert Gradient(a, b).value()[0, 0] == 0.0
    assert Gradient(c, a).value()[0, 0] == 1.0
    assert Gradient(c, b).value()[0, 0] == 0.0


def test_unary_plus():
    a = Variable()
    a.set_value(10)
    c = +a

    assert c.value() == a.value()
    assert Gradient(c, a).value()[0, 0] == 1.0


def test_unary_minus():
    a = Variable()
    a.set_value(10)
    c = -a

    assert c.value() == -a.value()
    assert Gradient(c, a).value()[0, 0] == -1.0


def test_identical_variables():
    a = Variable()
    a.set_value(10)
    x = a
    c = a * a + x

    assert c.value() == a.value() * a.value() + x.value()
    assert Gradient(c, a).value()[0, 0] == 2 * a.value() + Gradient(x, a).value()[0, 0]
    assert (
        Gradient(c, x).value()[0, 0] == 2 * a.value() * Gradient(a, x).value()[0, 0] + 1
    )


def test_elementary():
    a = Variable()
    a.set_value(1.0)
    b = Variable()
    b.set_value(2.0)

    c = -2 * a
    assert Gradient(c, a).value()[0, 0] == -2.0

    c = a / 3.0
    assert Gradient(c, a).value()[0, 0] == 1.0 / 3.0

    a.set_value(100.0)
    b.set_value(200.0)

    c = a + b
    assert Gradient(c, a).value()[0, 0] == 1.0
    assert Gradient(c, b).value()[0, 0] == 1.0

    c = a - b
    assert Gradient(c, a).value()[0, 0] == 1.0
    assert Gradient(c, b).value()[0, 0] == -1.0

    c = -a + b
    assert Gradient(c, a).value()[0, 0] == -1.0
    assert Gradient(c, b).value()[0, 0] == 1.0

    c = a + 1
    assert Gradient(c, a).value()[0, 0] == 1.0


def test_comparison():
    x = Variable()
    x.set_value(10.0)
    a = Variable()
    a.set_value(10.0)
    b = Variable()
    b.set_value(200.0)

    assert a.value() == a.value()
    assert a.value() == x.value()
    assert a.value() == 10.0
    assert 10.0 == a.value()

    assert a.value() != b.value()
    assert a.value() != 20.0
    assert 20.0 != a.value()

    assert a.value() < b.value()
    assert a.value() < 20.0

    assert b.value() > a.value()
    assert 20.0 > a.value()

    assert a.value() <= a.value()
    assert a.value() <= x.value()
    assert a.value() <= b.value()
    assert a.value() <= 10.0
    assert a.value() <= 20.0

    assert a.value() >= a.value()
    assert x.value() >= a.value()
    assert b.value() >= a.value()
    assert 10.0 >= a.value()
    assert 20.0 >= a.value()

    # Comparison between variables and expressions
    assert a.value() == a.value() / a.value() * a.value()
    assert a.value() / a.value() * a.value() == a.value()

    assert a.value() != (a - a).value()
    assert (a - a).value() != a.value()

    assert (a - a).value() < a.value()
    assert a.value() < (a + a).value()

    assert (a + a).value() > a.value()
    assert a.value() > (a - a).value()

    assert a.value() <= (a - a + a).value()
    assert (a - a + a).value() <= a.value()

    assert a.value() <= (a + a).value()
    assert (a - a).value() <= a.value()

    assert a.value() >= (a - a + a).value()
    assert (a - a + a).value() >= a.value()

    assert (a + a).value() >= a.value()
    assert a.value() >= (a - a).value()


def test_trigonometry():
    x = Variable()
    x.set_value(0.5)

    # sin(x)
    assert autodiff.sin(x).value() == math.sin(x.value())

    g = Gradient(autodiff.sin(x), x)
    assert g.get().value()[0, 0] == math.cos(x.value())
    assert g.value()[0, 0] == math.cos(x.value())

    # cos(x)
    assert autodiff.cos(x).value() == math.cos(x.value())

    g = Gradient(autodiff.cos(x), x)
    assert g.get().value()[0, 0] == -math.sin(x.value())
    assert g.value()[0, 0] == -math.sin(x.value())

    # tan(x)
    assert autodiff.tan(x).value() == math.tan(x.value())

    g = Gradient(autodiff.tan(x), x)
    assert g.get().value()[0, 0] == 1.0 / (math.cos(x.value()) * math.cos(x.value()))
    assert g.value()[0, 0] == 1.0 / (math.cos(x.value()) * math.cos(x.value()))

    # asin(x)
    assert autodiff.asin(x).value() == math.asin(x.value())

    g = Gradient(autodiff.asin(x), x)
    assert g.get().value()[0, 0] == 1.0 / math.sqrt(1 - x.value() * x.value())
    assert g.value()[0, 0] == 1.0 / math.sqrt(1 - x.value() * x.value())

    # acos(x)
    assert autodiff.acos(x).value() == math.acos(x.value())

    g = Gradient(autodiff.acos(x), x)
    assert g.get().value()[0, 0] == -1.0 / math.sqrt(1 - x.value() * x.value())
    assert g.value()[0, 0] == -1.0 / math.sqrt(1 - x.value() * x.value())

    # atan(x)
    assert autodiff.atan(x).value() == math.atan(x.value())

    g = Gradient(autodiff.atan(x), x)
    assert g.get().value()[0, 0] == 1.0 / (1 + x.value() * x.value())
    assert g.value()[0, 0] == 1.0 / (1 + x.value() * x.value())


def test_hyperbolic():
    x = Variable()
    x.set_value(1.0)

    # sinh(x)
    assert autodiff.sinh(x).value() == math.sinh(x.value())

    g = Gradient(autodiff.sinh(x), x)
    assert g.get().value()[0, 0] == math.cosh(x.value())
    assert g.value()[0, 0] == math.cosh(x.value())

    # cosh(x)
    assert autodiff.cosh(x).value() == math.cosh(x.value())

    g = Gradient(autodiff.cosh(x), x)
    assert g.get().value()[0, 0] == math.sinh(x.value())
    assert g.value()[0, 0] == math.sinh(x.value())

    # tanh(x)
    assert autodiff.tanh(x).value() == math.tanh(x.value())

    g = Gradient(autodiff.tanh(x), x)
    assert g.get().value()[0, 0] == 1.0 / (math.cosh(x.value()) * math.cosh(x.value()))
    assert g.value()[0, 0] == 1.0 / (math.cosh(x.value()) * math.cosh(x.value()))


def test_exponential():
    x = Variable()
    x.set_value(1.0)

    # log(x)
    assert autodiff.log(x).value() == math.log(x.value())

    g = Gradient(autodiff.log(x), x)
    assert g.get().value()[0, 0] == 1.0 / x.value()
    assert g.value()[0, 0] == 1.0 / x.value()

    # log10(x)
    assert autodiff.log10(x).value() == math.log10(x.value())

    g = Gradient(autodiff.log10(x), x)
    assert g.get().value()[0, 0] == 1.0 / (math.log(10.0) * x.value())
    assert g.value()[0, 0] == 1.0 / (math.log(10.0) * x.value())

    # exp(x)
    assert autodiff.exp(x).value() == math.exp(x.value())

    g = Gradient(autodiff.exp(x), x)
    assert g.get().value()[0, 0] == math.exp(x.value())
    assert g.value()[0, 0] == math.exp(x.value())


def test_power():
    x = Variable()
    x.set_value(1.0)
    a = Variable()
    a.set_value(2.0)
    y = 2 * a

    # sqrt(x)
    assert autodiff.sqrt(x).value() == math.sqrt(x.value())

    g = Gradient(autodiff.sqrt(x), x)
    assert autodiff.sqrt(x).value() == math.sqrt(x.value())
    assert g.get().value()[0, 0] == 0.5 / math.sqrt(x.value())
    assert g.value()[0, 0] == 0.5 / math.sqrt(x.value())

    # sqrt(a)
    assert autodiff.sqrt(a).value() == math.sqrt(a.value())

    g = Gradient(autodiff.sqrt(a), a)
    assert autodiff.sqrt(a).value() == math.sqrt(a.value())
    assert g.get().value()[0, 0] == 0.5 / math.sqrt(a.value())
    assert g.value()[0, 0] == 0.5 / math.sqrt(a.value())

    # cbrt(x)
    assert autodiff.cbrt(x).value() == math.cbrt(x.value())

    g = Gradient(autodiff.cbrt(x), x)
    assert autodiff.cbrt(x).value() == math.cbrt(x.value())
    assert g.get().value()[0, 0] == 1.0 / (
        3.0 * math.cbrt(x.value()) * math.cbrt(x.value())
    )
    assert g.value()[0, 0] == 1.0 / (3.0 * math.cbrt(x.value()) * math.cbrt(x.value()))

    # cbrt(a)
    assert autodiff.cbrt(a).value() == math.cbrt(a.value())

    g = Gradient(autodiff.cbrt(a), a)
    assert autodiff.cbrt(a).value() == math.cbrt(a.value())
    assert g.get().value()[0, 0] == 1.0 / (
        3.0 * math.cbrt(a.value()) * math.cbrt(a.value())
    )
    assert g.value()[0, 0] == 1.0 / (3.0 * math.cbrt(a.value()) * math.cbrt(a.value()))

    # x²
    assert autodiff.pow(x, 2.0).value() == math.pow(x.value(), 2.0)

    g = Gradient(autodiff.pow(x, 2.0), x)
    assert g.get().value()[0, 0] == 2.0 * x.value()
    assert g.value()[0, 0] == 2.0 * x.value()

    # 2ˣ
    assert autodiff.pow(2.0, x).value() == math.pow(2.0, x.value())

    g = Gradient(autodiff.pow(2.0, x), x)
    assert g.get().value()[0, 0] == math.log(2.0) * math.pow(2.0, x.value())
    assert g.value()[0, 0] == math.log(2.0) * math.pow(2.0, x.value())

    # xˣ
    assert autodiff.pow(x, x).value() == math.pow(x.value(), x.value())

    g = Gradient(autodiff.pow(x, x), x)
    assert g.get().value()[0, 0] == (math.log(x.value()) + 1) * math.pow(
        x.value(), x.value()
    )
    assert g.value()[0, 0] == (math.log(x.value()) + 1) * math.pow(x.value(), x.value())

    # y(a)
    assert y.value() == 2 * a.value()

    g = Gradient(y, a)
    assert g.get().value()[0, 0] == 2.0
    assert g.value()[0, 0] == 2.0

    # xʸ(x)
    assert autodiff.pow(x, y).value() == math.pow(x.value(), y.value())

    g = Gradient(autodiff.pow(x, y), x)
    assert g.get().value()[0, 0] == y.value() / x.value() * math.pow(
        x.value(), y.value()
    )
    assert g.value()[0, 0] == y.value() / x.value() * math.pow(x.value(), y.value())

    # xʸ(a)
    assert autodiff.pow(x, y).value() == math.pow(x.value(), y.value())

    g = Gradient(autodiff.pow(x, y), a)
    assert g.get().value()[0, 0] == math.pow(x.value(), y.value()) * (
        y.value() / x.value() * Gradient(x, a).value()[0, 0]
        + math.log(x.value()) * Gradient(y, a).value()[0, 0]
    )
    assert g.value()[0, 0] == math.pow(x.value(), y.value()) * (
        y.value() / x.value() * Gradient(x, a).value()[0, 0]
        + math.log(x.value()) * Gradient(y, a).value()[0, 0]
    )

    # xʸ(y)
    assert autodiff.pow(x, y).value() == math.pow(x.value(), y.value())

    g = Gradient(autodiff.pow(x, y), y)
    assert g.get().value()[0, 0] == math.log(x.value()) * math.pow(x.value(), y.value())
    assert g.value()[0, 0] == math.log(x.value()) * math.pow(x.value(), y.value())


def test_abs():
    x = Variable()
    g = Gradient(autodiff.abs(x), x)

    x.set_value(1.0)
    assert autodiff.abs(x).value() == abs(x.value())
    assert g.get().value()[0, 0] == 1.0
    assert g.value()[0, 0] == 1.0

    x.set_value(-1.0)
    assert autodiff.abs(x).value() == abs(x.value())
    assert g.get().value()[0, 0] == -1.0
    assert g.value()[0, 0] == -1.0

    x.set_value(0.0)
    assert autodiff.abs(x).value() == abs(x.value())
    assert g.get().value()[0, 0] == 0.0
    assert g.value()[0, 0] == 0.0


def test_atan2():
    x = Variable()
    y = Variable()

    # Testing atan2 function on (double, var)
    x.set_value(1.0)
    y.set_value(0.9)
    assert autodiff.atan2(2.0, x).value() == math.atan2(2.0, x.value())

    g = Gradient(autodiff.atan2(2.0, x), x)
    assert g.get().value()[0, 0] == pytest.approx(
        -2.0 / (2 * 2 + x.value() * x.value()), abs=1e-15
    )
    assert g.value()[0, 0] == pytest.approx(
        -2.0 / (2 * 2 + x.value() * x.value()), abs=1e-15
    )

    # Testing atan2 function on (var, double)
    x.set_value(1.0)
    y.set_value(0.9)
    assert autodiff.atan2(x, 2.0).value() == math.atan2(x.value(), 2.0)

    g = Gradient(autodiff.atan2(x, 2.0), x)
    assert g.get().value()[0, 0] == pytest.approx(
        (2.0 / (2 * 2 + x * x)).value(), abs=1e-15
    )
    assert g.value()[0, 0] == pytest.approx(
        2.0 / (2 * 2 + x.value() * x.value()), abs=1e-15
    )

    # Testing atan2 function on (var, var)
    x.set_value(1.1)
    y.set_value(0.9)
    assert autodiff.atan2(y, x).value() == math.atan2(y.value(), x.value())

    g = Gradient(autodiff.atan2(y, x), y)
    assert g.get().value()[0, 0] == pytest.approx(
        (x / (x * x + y * y)).value(), abs=1e-15
    )
    assert g.value()[0, 0] == pytest.approx(
        x.value() / (x.value() * x.value() + y.value() * y.value()), abs=1e-15
    )

    g = Gradient(autodiff.atan2(y, x), x)
    assert g.get().value()[0, 0] == pytest.approx(
        -y.value() / (x.value() * x.value() + y.value() * y.value()), abs=1e-15
    )
    assert g.value()[0, 0] == pytest.approx(
        -y.value() / (x.value() * x.value() + y.value() * y.value()), abs=1e-15
    )

    # Testing atan2 function on (expr, expr)
    assert 3 * autodiff.atan2(autodiff.sin(y), 2 * x + 1).value() == 3 * math.atan2(
        math.sin(y.value()), 2 * x.value() + 1
    )

    g = Gradient(3 * autodiff.atan2(autodiff.sin(y), 2 * x + 1), y)
    assert g.get().value()[0, 0] == pytest.approx(
        3
        * (2 * x.value() + 1)
        * math.cos(y.value())
        / (
            (2 * x.value() + 1) * (2 * x.value() + 1)
            + math.sin(y.value()) * math.sin(y.value())
        ),
        abs=1e-15,
    )
    assert g.value()[0, 0] == pytest.approx(
        3
        * (2 * x.value() + 1)
        * math.cos(y.value())
        / (
            (2 * x.value() + 1) * (2 * x.value() + 1)
            + math.sin(y.value()) * math.sin(y.value())
        ),
        abs=1e-15,
    )

    g = Gradient(3 * autodiff.atan2(autodiff.sin(y), 2 * x + 1), x)
    assert g.get().value()[0, 0] == pytest.approx(
        3
        * -2
        * math.sin(y.value())
        / (
            (2 * x.value() + 1) * (2 * x.value() + 1)
            + math.sin(y.value()) * math.sin(y.value())
        ),
        abs=1e-15,
    )
    assert g.value()[0, 0] == pytest.approx(
        3
        * -2
        * math.sin(y.value())
        / (
            (2 * x.value() + 1) * (2 * x.value() + 1)
            + math.sin(y.value()) * math.sin(y.value())
        ),
        abs=1e-15,
    )


def test_hypot():
    x = Variable()
    y = Variable()

    # Testing hypot function on (var, double)
    x.set_value(1.8)
    y.set_value(1.5)
    assert autodiff.hypot(x, 2.0).value() == math.hypot(x.value(), 2.0)

    g = Gradient(autodiff.hypot(x, 2.0), x)
    assert g.get().value()[0, 0] == x.value() / math.hypot(x.value(), 2.0)
    assert g.value()[0, 0] == x.value() / math.hypot(x.value(), 2.0)

    # Testing hypot function on (double, var)
    assert autodiff.hypot(2.0, y).value() == math.hypot(2.0, y.value())

    g = Gradient(autodiff.hypot(2.0, y), y)
    assert g.get().value()[0, 0] == y.value() / math.hypot(2.0, y.value())
    assert g.value()[0, 0] == y.value() / math.hypot(2.0, y.value())

    # Testing hypot function on (var, var)
    x.set_value(1.3)
    y.set_value(2.3)
    assert autodiff.hypot(x, y).value() == math.hypot(x.value(), y.value())

    g = Gradient(autodiff.hypot(x, y), x)
    assert g.get().value()[0, 0] == x.value() / math.hypot(x.value(), y.value())
    assert g.value()[0, 0] == x.value() / math.hypot(x.value(), y.value())

    g = Gradient(autodiff.hypot(x, y), y)
    assert g.get().value()[0, 0] == y.value() / math.hypot(x.value(), y.value())
    assert g.value()[0, 0] == y.value() / math.hypot(x.value(), y.value())

    # Testing hypot function on (expr, expr)
    x.set_value(1.3)
    y.set_value(2.3)
    assert autodiff.hypot(2.0 * x, 3.0 * y).value() == math.hypot(
        2.0 * x.value(), 3.0 * y.value()
    )

    g = Gradient(autodiff.hypot(2.0 * x, 3.0 * y), x)
    assert g.get().value()[0, 0] == 4.0 * x.value() / math.hypot(
        2.0 * x.value(), 3.0 * y.value()
    )
    assert g.value()[0, 0] == 4.0 * x.value() / math.hypot(
        2.0 * x.value(), 3.0 * y.value()
    )

    g = Gradient(autodiff.hypot(2.0 * x, 3.0 * y), y)
    assert g.get().value()[0, 0] == 9.0 * y.value() / math.hypot(
        2.0 * x.value(), 3.0 * y.value()
    )
    assert g.value()[0, 0] == 9.0 * y.value() / math.hypot(
        2.0 * x.value(), 3.0 * y.value()
    )

    # Testing hypot function on (var, var, var)
    z = Variable()
    x.set_value(1.3)
    y.set_value(2.3)
    z.set_value(3.3)
    assert autodiff.hypot(x, y, z).value() == math.hypot(
        x.value(), y.value(), z.value()
    )

    g = Gradient(autodiff.hypot(x, y, z), x)
    assert g.get().value()[0, 0] == x.value() / math.hypot(
        x.value(), y.value(), z.value()
    )
    assert g.value()[0, 0] == x.value() / math.hypot(x.value(), y.value(), z.value())

    g = Gradient(autodiff.hypot(x, y, z), y)
    assert g.get().value()[0, 0] == y.value() / math.hypot(
        x.value(), y.value(), z.value()
    )
    assert g.value()[0, 0] == y.value() / math.hypot(x.value(), y.value(), z.value())

    g = Gradient(autodiff.hypot(x, y, z), z)
    assert g.get().value()[0, 0] == z.value() / math.hypot(
        x.value(), y.value(), z.value()
    )
    assert g.value()[0, 0] == z.value() / math.hypot(x.value(), y.value(), z.value())


def test_miscellaneous():
    x = Variable()

    # dx/dx
    x.set_value(3.0)
    assert autodiff.abs(x).value() == abs(x.value())

    g = Gradient(x, x)
    assert g.get().value()[0, 0] == 1.0
    assert g.value()[0, 0] == 1.0

    # erf(x)
    x.set_value(0.5)
    assert autodiff.erf(x).value() == math.erf(x.value())

    g = Gradient(autodiff.erf(x), x)
    assert g.get().value()[0, 0] == 2.0 / math.sqrt(math.pi) * math.exp(
        -x.value() * x.value()
    )
    assert g.value()[0, 0] == 2.0 / math.sqrt(math.pi) * math.exp(
        -x.value() * x.value()
    )


def test_reuse():
    a = Variable()
    a.set_value(10)

    b = Variable()
    b.set_value(20)

    x = a * b

    g = Gradient(x, a)

    assert g.get().value()[0] == 20.0
    assert g.value()[0] == 20.0

    b.set_value(10)
    assert g.get().value()[0] == 10.0
    assert g.value()[0] == 10.0


def test_sign():
    def sign(x):
        if x < 0.0:
            return -1.0
        elif x == 0.0:
            return 0.0
        else:
            return 1.0

    x = Variable()

    # sign(1.0)
    x.set_value(1.0)
    assert sign(x.value()) == autodiff.sign(x).value()

    g = Gradient(autodiff.sign(x), x)
    assert g.get().value()[0, 0] == 0.0
    assert g.value()[0, 0] == 0.0

    # sign(-1.0)
    x.set_value(-1.0)
    assert sign(x.value()) == autodiff.sign(x).value()

    g = Gradient(autodiff.sign(x), x)
    assert g.get().value()[0, 0] == 0.0
    assert g.value()[0, 0] == 0.0

    # sign(0.0)
    x.set_value(0.0)
    assert sign(x.value()) == autodiff.sign(x).value()

    g = Gradient(autodiff.sign(x), x)
    assert g.get().value()[0, 0] == 0.0
    assert g.value()[0, 0] == 0.0


def test_non_scalar():
    x = VariableMatrix(3)
    x[0].set_value(1)
    x[1].set_value(2)
    x[2].set_value(3)

    # y = [x₁ + 3x₂ − 5x₃]
    #
    # dy/dx = [1  3  −5]
    y = x[0] + 3 * x[1] - 5 * x[2]
    g = Gradient(y, x)

    expected_g = np.array([[1.0], [3.0], [-5.0]])

    g_get_value = g.get().value()
    assert g_get_value.shape == (3, 1)
    assert (g_get_value == expected_g).all()

    g_value = g.get().value()
    assert g_value.shape == (3, 1)
    assert (g_value == expected_g).all()
