import platform

import numpy as np
import pytest
import sleipnir.autodiff as autodiff
from sleipnir.autodiff import ExpressionType
from sleipnir.optimization import ExitStatus, Problem, bounds


def test_quartic():
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


def test_rosenbrock_with_cubic_and_line_constraint():
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
            assert x.value() == pytest.approx(
                0.0, abs=1e-2
            ) or x.value() == pytest.approx(1.0, abs=1e-2)
            assert y.value() == pytest.approx(
                0.0, abs=1e-2
            ) or y.value() == pytest.approx(1.0, abs=1e-2)


def test_rosenbrock_with_disk_constraint():
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

            assert x.value() == pytest.approx(1.0, abs=1e-3)
            assert y.value() == pytest.approx(1.0, abs=1e-3)


def test_minimum_2d_distance_with_linear_constraint():
    problem = Problem()

    x, y = problem.decision_variable(2)
    x.set_value(20.0)
    y.set_value(50.0)

    problem.minimize(autodiff.sqrt(x * x + y * y))

    problem.subject_to(y == -x + 5.0)

    assert problem.cost_function_type() == ExpressionType.NONLINEAR
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    if platform.system() == "Linux" and platform.machine() == "aarch64":
        # FIXME: Fails on Linux aarch64 with "feasibility restoration failed"
        assert (
            problem.solve(diagnostics=True) == ExitStatus.FEASIBILITY_RESTORATION_FAILED
        )
        return
    else:
        assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert x.value() == pytest.approx(2.5, abs=1e-2)
    assert y.value() == pytest.approx(2.5, abs=1e-2)


def test_conflicting_bounds():
    problem = Problem()

    x, y = problem.decision_variable(2)

    problem.minimize(autodiff.hypot(x, y))

    problem.subject_to(autodiff.hypot(x, y) <= 1)
    problem.subject_to(bounds(0.5, x, -0.5))

    assert problem.cost_function_type() == ExpressionType.NONLINEAR
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONLINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.GLOBALLY_INFEASIBLE


def test_wachter_and_biegler_line_search_failure():
    # See example 19.2 of [1]
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

    # FIXME: Fails with "feasibility restoration failed"
    assert problem.solve(diagnostics=True) == ExitStatus.FEASIBILITY_RESTORATION_FAILED

    # assert x.value() == pytest.approx(1.0, abs=1e-6)
    # assert s1.value() == pytest.approx(0.0, abs=1e-6)
    # assert s2.value() == pytest.approx(0.5, abs=1e-6)
