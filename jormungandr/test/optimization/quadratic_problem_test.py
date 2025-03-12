import numpy as np
import pytest

from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import ExitStatus, Problem


def test_unconstrained1d():
    problem = Problem()

    x = problem.decision_variable()
    x.set_value(2.0)

    problem.minimize(x * x - 6.0 * x)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert x.value() == pytest.approx(3.0, abs=1e-6)


def test_unconstrained2d_1():
    problem = Problem()

    x = problem.decision_variable()
    x.set_value(1.0)
    y = problem.decision_variable()
    y.set_value(2.0)

    problem.minimize(x * x + y * y)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert x.value() == pytest.approx(0.0, abs=1e-6)
    assert y.value() == pytest.approx(0.0, abs=1e-6)


def test_unconstrained2d_2():
    problem = Problem()

    x = problem.decision_variable(2)
    x[0].set_value(1.0)
    x[1].set_value(2.0)

    problem.minimize(x.T @ x)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert x.value(0) == pytest.approx(0.0, abs=1e-6)
    assert x.value(1) == pytest.approx(0.0, abs=1e-6)


# Maximize xy subject to x + 3y = 36.
#
# Maximize f(x,y) = xy
# subject to g(x,y) = x + 3y - 36 = 0
#
#         value func  constraint
#              |          |
#              v          v
# L(x,y,λ) = f(x,y) - λg(x,y)
# L(x,y,λ) = xy - λ(x + 3y - 36)
# L(x,y,λ) = xy - xλ - 3yλ + 36λ
#
# ∇_x,y,λ L(x,y,λ) = 0
#
# ∂L/∂x = y - λ
# ∂L/∂y = x - 3λ
# ∂L/∂λ = -x - 3y + 36
#
#  0x + 1y - 1λ = 0
#  1x + 0y - 3λ = 0
# -1x - 3y + 0λ + 36 = 0
#
# [ 0  1 -1][x]   [  0]
# [ 1  0 -3][y] = [  0]
# [-1 -3  0][λ]   [-36]
#
# Solve with:
# ```python
#   np.linalg.solve(
#     np.array([[0,1,-1],
#               [1,0,-3],
#               [-1,-3,0]]),
#     np.array([[0], [0], [-36]]))
# ```
#
# [x]   [18]
# [y] = [ 6]
# [λ]   [ 6]
def test_equality_constrained_1():
    problem = Problem()

    x = problem.decision_variable()
    y = problem.decision_variable()

    problem.maximize(x * y)

    problem.subject_to(x + 3 * y == 36)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert x.value() == pytest.approx(18.0, abs=1e-5)
    assert y.value() == pytest.approx(6.0, abs=1e-5)


def test_equality_constrained_2():
    problem = Problem()

    x = problem.decision_variable(2)
    x[0].set_value(1.0)
    x[1].set_value(2.0)

    problem.minimize(x.T @ x)

    problem.subject_to(x == np.array([[3.0], [3.0]]))

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert x.value(0) == pytest.approx(3.0, abs=1e-5)
    assert x.value(1) == pytest.approx(3.0, abs=1e-5)


def test_inequality_constrained_2d():
    problem = Problem()

    x = problem.decision_variable()
    x.set_value(5.0)
    y = problem.decision_variable()
    y.set_value(5.0)

    problem.minimize(x * x + y * 2 * y)
    problem.subject_to(y >= -x + 5)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert x.value() == pytest.approx(3.0 + 1.0 / 3.0, abs=1e-6)
    assert y.value() == pytest.approx(1.0 + 2.0 / 3.0, abs=1e-6)
