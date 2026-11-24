import math

import numpy as np
import pytest
from sleipnir.autodiff import (
    ExpressionType,
    RealMatrix,
    UnitVector,
)
from sleipnir.optimization import (
    ExitStatus,
    IterationInfo,
    Problem,
)
from tolerance import rounding_tolerance, solution_tolerance

RAYLEIGH_MATRIX = np.array([[2.0, -1.0, 0.5], [-1.0, 3.0, 0.25], [0.5, 0.25, 1.0]])
"""Symmetric with distinct eigenvalues."""


def check_iterates_on_manifold(problem: Problem, offset: int, size: int) -> None:
    """Checks the manifold block of each iterate stays unit-length."""

    def callback(info: IterationInfo) -> None:
        norm = np.linalg.norm(info.x[offset : offset + size])
        assert norm == pytest.approx(1.0, abs=rounding_tolerance(10))

    problem.add_callback(callback)


@pytest.mark.parametrize(
    "x0",
    [[1.0, 0.0, 0.0], [-1.0, 0.2, 0.1], [0.1, 1.0, -0.3], [0.5, -0.5, 0.7]],
)
def test_sphere_rayleigh_quotient(x0: list[float]) -> None:
    A = RAYLEIGH_MATRIX
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    problem = Problem()
    n = problem.unit_vector_decision_variable(3)
    n.set_value(np.array(x0))

    problem.minimize(n.T @ A @ n)
    check_iterates_on_manifold(problem, 0, 3)

    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.NONE

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    n_value = n.value()
    assert np.linalg.norm(n_value) == pytest.approx(1.0, abs=rounding_tolerance(10))
    assert n_value @ A @ n_value == pytest.approx(
        eigenvalues[0], abs=solution_tolerance(1)
    )
    assert abs(n_value @ eigenvectors[:, 0]) == pytest.approx(
        1.0, abs=solution_tolerance(1)
    )


# Equality constraint (SQP) or inequality constraint (IPM)
@pytest.mark.parametrize("use_inequality", [False, True])
def test_sphere_with_constraints_matches_unit_norm_equality_constraint(
    use_inequality: bool,
) -> None:
    A = RAYLEIGH_MATRIX
    x0 = np.ones((3, 1)) / math.sqrt(3)

    def add_constraint(problem: Problem, n: RealMatrix | UnitVector) -> None:
        if use_inequality:
            problem.subject_to(n[2] >= 0.6)
        else:
            problem.subject_to(n[0] == 0.5)

    # Manifold formulation
    manifold_problem = Problem()
    n = manifold_problem.unit_vector_decision_variable(3)
    n.set_value(x0.ravel())
    manifold_problem.minimize(n.T @ A @ n)
    add_constraint(manifold_problem, n)
    check_iterates_on_manifold(manifold_problem, 0, 3)

    assert manifold_problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    # Unit-norm equality constraint formulation
    constrained_problem = Problem()
    m = constrained_problem.real_decision_variable(3)
    m.set_value(x0)
    constrained_problem.minimize(m.T @ A @ m)
    constrained_problem.subject_to(m.T @ m == 1.0)
    add_constraint(constrained_problem, m)

    assert constrained_problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert n.value() == pytest.approx(m.value().ravel(), abs=solution_tolerance(10))
