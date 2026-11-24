import numpy as np
import pytest
from sleipnir.autodiff import (
    Translation2,
)
from sleipnir.optimization import (
    ExitStatus,
    Problem,
)
from tolerance import solution_tolerance

A = np.array([[2.0, -1.0, 0.5], [-1.0, 3.0, 0.25], [0.5, 0.25, 1.0]])


def test_unit_vector_equality_constant_direction() -> None:
    # The direction is nearly opposite the initial value e₁
    d = np.array([-1.0, 0.2, -0.1])

    problem = Problem()
    n = problem.unit_vector_decision_variable(3)
    problem.minimize(n.T @ A @ n)
    problem.subject_to(n == d)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert n.value() == pytest.approx(d / np.linalg.norm(d), abs=solution_tolerance(1))


def test_unit_vector_equality_with_matrix() -> None:
    # A unit vector equals only another unit vector or a constant direction
    problem = Problem()
    n = problem.unit_vector_decision_variable(3)
    x = problem.real_decision_variable(3)
    for compare in [lambda: n == x, lambda: x == n, lambda: n == 0]:
        with pytest.raises(TypeError):
            compare()

    # Directions that need a conversion to a vector still work
    problem.subject_to(n == np.array([0, 0, 1]))

    # Inequalities are componentwise bounds
    problem.subject_to(n >= x)
    problem.subject_to(x <= n)


def test_unit_vector_equality_two_decision_variables() -> None:
    problem = Problem()
    a = problem.unit_vector_decision_variable(3)
    b = problem.unit_vector_decision_variable(3)
    a.set_value(np.array([1.0, 0.3, 0.0]))
    b.set_value(np.array([0.2, 1.0, 0.4]))
    problem.minimize(a.T @ A @ a + b.T @ A @ b)
    problem.subject_to(a == b)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    eigenvalues = np.linalg.eigvalsh(A)
    a_value = a.value()
    b_value = b.value()
    assert a_value @ b_value == pytest.approx(1.0, abs=solution_tolerance(1))
    assert a_value @ A @ a_value == pytest.approx(
        eigenvalues[0], abs=solution_tolerance(1)
    )


def test_distance_tolerances_at_least() -> None:
    problem = Problem()
    p = problem.translation2_decision_variable()
    p.set_value(np.array([0.5, 0.1]))
    problem.minimize(p.squared_distance(Translation2(1.0, 0.0)))
    problem.subject_to(p.squared_distance(Translation2(0.0, 0.0)) >= 4.0)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert p.x().value() == pytest.approx(2.0, abs=solution_tolerance(1))
    assert p.y().value() == pytest.approx(0.0, abs=solution_tolerance(1))
