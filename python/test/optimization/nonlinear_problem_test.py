import math

import numpy as np
import pytest
from sleipnir import autodiff
from sleipnir.autodiff import ExpressionType
from sleipnir.optimization import ExitStatus, Problem, bounds


def test_quartic() -> None:
    problem = Problem()

    x = problem.decision_variable()
    x.set_value(20.0)

    problem.minimize(autodiff.pow(x, 4))

    problem.subject_to(x >= 1)

    assert problem.cost_function_type() == ExpressionType.NONLINEAR
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert x.value() == pytest.approx(1.0, abs=1e-6)


def test_rosenbrock_with_cubic_and_line_constraint() -> None:
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization

    problem = Problem()

    x, y = problem.decision_variable(2)

    problem.minimize(
        100 * autodiff.pow(y - autodiff.pow(x, 2), 2) + autodiff.pow(1 - x, 2)
    )

    problem.subject_to(y >= autodiff.pow(x - 1, 3) + 1)
    problem.subject_to(y <= -x + 2)

    assert problem.cost_function_type() == ExpressionType.NONLINEAR
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONLINEAR

    for x0 in np.arange(-1.5, 1.5, 0.1):
        for y0 in np.arange(-0.5, 2.5, 0.1):
            x.set_value(x0)
            y.set_value(y0)

            assert problem.solve() == ExitStatus.SUCCESS

            # Local minimum at (0.0, 0.0)
            # Global minimum at (1.0, 1.0)
            sol = (x.value(), y.value())
            assert sol == pytest.approx((0.0, 0.0), abs=1e-2) or sol == pytest.approx(
                (1.0, 1.0), abs=1e-2
            )


def test_rosenbrock_with_disk_constraint() -> None:
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization

    problem = Problem()

    x, y = problem.decision_variable(2)

    problem.minimize(
        100 * autodiff.pow(y - autodiff.pow(x, 2), 2) + autodiff.pow(1 - x, 2)
    )

    problem.subject_to(autodiff.pow(x, 2) + autodiff.pow(y, 2) <= 2)

    assert problem.cost_function_type() == ExpressionType.NONLINEAR
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.QUADRATIC

    for x0 in np.arange(-1.5, 1.5, 0.1):
        for y0 in np.arange(-1.5, 1.5, 0.1):
            x.set_value(x0)
            y.set_value(y0)

            assert problem.solve() == ExitStatus.SUCCESS

            sol = (x.value(), y.value())
            assert sol == pytest.approx((1.0, 1.0), abs=1e-3)


def test_gomez_and_levy() -> None:
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization

    problem = Problem()

    x, y = problem.decision_variable(2)

    problem.minimize(
        4 * autodiff.pow(x, 2)
        - 2.1 * autodiff.pow(x, 4)
        + autodiff.pow(x, 6) / 3
        + x * y
        - 4 * autodiff.pow(y, 2)
        + 4 * autodiff.pow(y, 4)
    )

    c = -autodiff.sin(4 * math.pi * x) + 2 * autodiff.pow(
        autodiff.sin(2 * math.pi * y), 2
    )
    problem.subject_to(c <= 1.5)
    problem.subject_to(-1 <= x)
    problem.subject_to(x <= 0.75)
    problem.subject_to(-1 <= y)
    problem.subject_to(y <= 1)

    assert problem.cost_function_type() == ExpressionType.NONLINEAR
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONLINEAR

    # The feasible region is a set of disconnected islands, so each solve
    # converges to one of many local minima. Solving from a grid of initial
    # guesses with one Problem also exercises warm starts from a previous
    # solution far from the new initial guess.
    for x0 in np.arange(-0.95, 0.75, 0.1):
        for y0 in np.arange(-0.95, 1.0, 0.1):
            x.set_value(x0)
            y.set_value(y0)

            assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS, (
                f"(x₀, y₀) = ({x0}, {y0})"
            )

            assert c.value() <= 1.5 + 1e-8
            assert x.value() >= -1 - 1e-8
            assert x.value() <= 0.75 + 1e-8
            assert y.value() >= -1 - 1e-8
            assert y.value() <= 1 + 1e-8


def test_minimum_2d_distance_with_linear_constraint() -> None:
    problem = Problem()

    x, y = problem.decision_variable(2)
    x.set_value(20.0)
    y.set_value(50.0)

    problem.minimize(autodiff.sqrt(x * x + y * y))

    problem.subject_to(y == -x + 5.0)

    assert problem.cost_function_type() == ExpressionType.NONLINEAR
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert x.value() == pytest.approx(2.5, abs=1e-2)
    assert y.value() == pytest.approx(2.5, abs=1e-2)


def test_conflicting_bounds() -> None:
    problem = Problem()

    x, y = problem.decision_variable(2)

    problem.minimize(autodiff.hypot(x, y))

    problem.subject_to(autodiff.hypot(x, y) <= 1)
    problem.subject_to(bounds(0.5, x, -0.5))

    assert problem.cost_function_type() == ExpressionType.NONLINEAR
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONLINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.GLOBALLY_INFEASIBLE


def test_wachter_and_biegler_line_search_failure() -> None:
    # Example 19.2 of [1] exercises a deficiency of line search IPMs that must
    # be overcome by feasibility restoration. Depending on the specific
    # trajectory taken, either success or local infeasibility are expected.
    #
    # [1] Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19.
    #     Springer, 2006.

    problem = Problem()

    x, s1, s2 = problem.decision_variable(3)
    x.set_value(-2)
    s1.set_value(3)
    s2.set_value(1)

    problem.minimize(x)

    problem.subject_to(x**2 - s1 - 1 == 0)
    problem.subject_to(x - s2 - 0.5 == 0)
    problem.subject_to(s1 >= 0)
    problem.subject_to(s2 >= 0)

    assert problem.cost_function_type() == ExpressionType.LINEAR
    assert problem.equality_constraint_type() == ExpressionType.QUADRATIC
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    status = problem.solve(diagnostics=True)
    assert status in (ExitStatus.SUCCESS, ExitStatus.LOCALLY_INFEASIBLE)

    if status == ExitStatus.SUCCESS:
        assert x.value() == pytest.approx(1.0, abs=1e-6)
        assert s1.value() == pytest.approx(0.0, abs=1e-6)
        assert s2.value() == pytest.approx(0.5, abs=1e-6)
    else:
        assert x.value() == pytest.approx(-1.0, abs=1e-6)
        assert s1.value() == pytest.approx(0.0, abs=1e-6)
        assert s2.value() == pytest.approx(0.0, abs=3e-5)


def test_hock_schittkowski_13() -> None:
    # Problem 13 of [1] (see
    # https://vanderbei.princeton.edu/ampl/nlmodels/cute/hs013.mod) has
    # constraint gradients that are linearly dependent at the solution.
    #
    # [1] Hock, W. and Schittkowski, K. "Test Examples for Nonlinear Programming
    #     Codes", Springer, 1981.

    problem = Problem()

    x, y = problem.decision_variable(2)
    x.set_value(-2)
    y.set_value(-2)

    problem.minimize((x - 2) ** 2 + y**2)

    problem.subject_to((1 - x) ** 3 >= y)
    problem.subject_to(x >= 0)
    problem.subject_to(y >= 0)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONLINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    # The solution has no finite Lagrange multipliers, so the iterates only
    # approach it to within the square root of the tolerance
    assert x.value() == pytest.approx(1.0, abs=1e-4)
    assert y.value() == pytest.approx(0.0, abs=1e-4)
