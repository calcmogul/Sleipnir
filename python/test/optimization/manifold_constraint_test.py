import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sleipnir.autodiff import (
    ExpressionType,
    Pose3,
    Rotation2,
    Rotation3,
    Translation2,
    VariableMatrix,
)
from sleipnir.optimization import (
    ExitStatus,
    Problem,
    bounds,
    se2,
    se3,
    so2,
    so3,
    sphere,
)
from tolerance import solution_tolerance


def test_heading_equality_from_the_opposite_heading() -> None:
    problem = Problem()
    pose = problem.decision_variable(se2)

    # The initial heading is 0, opposite the constrained heading
    problem.minimize(pose.translation().squared_distance(Translation2(1.0, 2.0)))
    problem.subject_to(pose.rotation() == Rotation2(math.pi))

    assert problem.equality_constraint_type() == ExpressionType.NONLINEAR
    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert abs(pose.rotation().value()) == pytest.approx(
        math.pi, abs=solution_tolerance(1)
    )
    assert pose.translation().x().value() == pytest.approx(
        1.0, abs=solution_tolerance(1)
    )


def test_pose3_equality() -> None:
    q_target_axis = np.array([1.0, -2.0, 0.5])
    q_target = Rotation.from_rotvec(
        2.9 * q_target_axis / np.linalg.norm(q_target_axis)
    ).as_quat(scalar_first=True)
    target = Pose3(np.array([1.0, -0.5, 2.0]), Rotation3(*q_target))

    problem = Problem()
    pose = problem.decision_variable(se3)
    problem.subject_to(pose == target)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert pose.translation().value() == pytest.approx(
        np.array([1.0, -0.5, 2.0]), abs=solution_tolerance(1)
    )
    assert abs(pose.rotation().value() @ q_target) == pytest.approx(
        1.0, abs=solution_tolerance(1)
    )


A = np.array([[2.0, -1.0, 0.5], [-1.0, 3.0, 0.25], [0.5, 0.25, 1.0]])


def test_unit_vector_equality_constant_direction() -> None:
    # The direction is nearly opposite the initial value e₁
    d = np.array([-1.0, 0.2, -0.1])

    problem = Problem()
    n = problem.decision_variable(sphere, 3)
    problem.minimize(n.T @ A @ n)
    problem.subject_to(n == d)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert n.value() == pytest.approx(d / np.linalg.norm(d), abs=solution_tolerance(1))


def test_unit_vector_equality_two_decision_variables() -> None:
    problem = Problem()
    a = problem.decision_variable(sphere, 3)
    b = problem.decision_variable(sphere, 3)
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
    z = problem.decision_variable(so2)
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
    q = problem.decision_variable(so3)
    n = problem.decision_variable(sphere, 3)

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
    problem.subject_to(n.dot(VariableMatrix([[1.0], [0.0], [0.0]])) >= math.cos(0.3))

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert 2 * math.acos(abs(q.value()[0])) == pytest.approx(
        0.4, abs=solution_tolerance(1)
    )
    assert n.value(0) == pytest.approx(math.cos(0.3), abs=solution_tolerance(1))


def test_distance_tolerances_at_most() -> None:
    problem = Problem()
    pose = problem.decision_variable(se2)
    problem.minimize(pose.translation().squared_distance(Translation2(3.0, 4.0)))
    problem.subject_to(
        pose.translation().squared_distance(Translation2(0.0, 0.0)) <= 1.0
    )

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert pose.translation().x().value() == pytest.approx(
        0.6, abs=solution_tolerance(1)
    )
    assert pose.translation().y().value() == pytest.approx(
        0.8, abs=solution_tolerance(1)
    )


def test_distance_tolerances_at_least() -> None:
    problem = Problem()
    p = Translation2(problem.decision_variable(2))
    p.set_value(np.array([0.5, 0.1]))
    problem.minimize(p.squared_distance(Translation2(1.0, 0.0)))
    problem.subject_to(p.squared_distance(Translation2(0.0, 0.0)) >= 4.0)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert p.x().value() == pytest.approx(2.0, abs=solution_tolerance(1))
    assert p.y().value() == pytest.approx(0.0, abs=solution_tolerance(1))
