import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sleipnir.autodiff import (
    RealMatrix,
    Rotation2,
    Rotation3,
    Translation2,
)
from sleipnir.optimization import (
    ExitStatus,
    Problem,
    bounds,
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


def test_heading_tolerance_wrapping_past_pi() -> None:
    # The allowed headings are 3 ± 0.3 rad, which wraps past π. The cost pulls
    # the heading toward −2.5 rad, past the upper end at 3.3 rad.
    problem = Problem()
    z = problem.rotation2_decision_variable()
    z.set_value(3.0)
    problem.minimize(-(z.cos() * math.cos(-2.5) + z.sin() * math.sin(-2.5)))
    error = z.relative_to(Rotation2(3.0)).angle()
    problem.subject_to(bounds(-0.3, error, 0.3))

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    # The active bound's multiplier is |sin(3.3 + 2.5)| ≈ 0.46, so the
    # interior-point slack at convergence is about tolerance/0.46
    assert z.value() == pytest.approx(3.3 - 2 * math.pi, abs=solution_tolerance(5))


def test_rotation3_and_unit_vector_tolerances() -> None:
    # The cost pulls the rotation toward 2 rad about an axis, and the
    # constraint keeps it within 0.4 rad of the identity
    q_target_axis = np.array([1.0, 1.0, 0.0])
    q_target = Rotation.from_rotvec(2.0 * q_target_axis / np.linalg.norm(q_target_axis))
    target = Rotation3(*q_target.as_quat(scalar_first=True))

    problem = Problem()
    q = problem.rotation3_decision_variable()
    n = problem.unit_vector_decision_variable(3)

    q_dot = (
        q.w() * target.w()
        + q.x() * target.x()
        + q.y() * target.y()
        + q.z() * target.z()
    )
    problem.minimize(-q_dot * q_dot - n[2])
    error = q.relative_to(Rotation3(1.0, 0.0, 0.0, 0.0)).to_vector()
    problem.subject_to(error.T @ error <= 0.4 * 0.4)

    # The cost pulls the vector toward e₃, and the constraint keeps it within
    # 0.3 rad of e₁
    problem.subject_to(n.dot(RealMatrix([[1.0], [0.0], [0.0]])) >= math.cos(0.3))

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert 2 * math.acos(abs(q.value()[0])) == pytest.approx(
        0.4, abs=solution_tolerance(1)
    )
    assert n.value(0) == pytest.approx(math.cos(0.3), abs=solution_tolerance(1))


def test_distance_tolerances_at_least() -> None:
    problem = Problem()
    p = problem.translation2_decision_variable()
    p.set_value(np.array([0.5, 0.1]))
    problem.minimize(p.squared_distance(Translation2(1.0, 0.0)))
    problem.subject_to(p.squared_distance(Translation2(0.0, 0.0)) >= 4.0)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert p.x().value() == pytest.approx(2.0, abs=solution_tolerance(1))
    assert p.y().value() == pytest.approx(0.0, abs=solution_tolerance(1))
