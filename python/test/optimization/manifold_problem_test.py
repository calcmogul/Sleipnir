import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sleipnir.autodiff import (
    ExpressionType,
    Pose2,
    Rotation3,
    Translation2,
    Translation3,
    Variable,
    VariableMatrix,
)
from sleipnir.optimization import (
    ExitStatus,
    IterationInfo,
    Problem,
    se2,
    se3,
    so2,
    so3,
    sphere,
)
from tolerance import rounding_tolerance, solution_tolerance

RAYLEIGH_MATRIX = np.array([[2.0, -1.0, 0.5], [-1.0, 3.0, 0.25], [0.5, 0.25, 1.0]])
"""Symmetric with distinct eigenvalues."""


def check_iterates_on_manifold(problem: Problem, offset: int, size: int) -> None:
    """Checks the manifold block of each iterate stays unit-length."""

    def callback(info: IterationInfo) -> bool:
        norm = np.linalg.norm(info.x[offset : offset + size])
        assert norm == pytest.approx(1.0, abs=rounding_tolerance(10))
        return False

    problem.add_callback(callback)


@pytest.mark.parametrize(
    "x0",
    [[1.0, 0.0, 0.0], [-1.0, 0.2, 0.1], [0.1, 1.0, -0.3], [0.5, -0.5, 0.7]],
)
def test_sphere_rayleigh_quotient(x0: list[float]) -> None:
    A = RAYLEIGH_MATRIX
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    problem = Problem()
    n = problem.decision_variable(sphere, 3)
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

    def add_constraint(problem: Problem, n: VariableMatrix) -> None:
        if use_inequality:
            problem.subject_to(n[2] >= 0.6)
        else:
            problem.subject_to(n[0] == 0.5)

    # Manifold formulation
    manifold_problem = Problem()
    n = manifold_problem.decision_variable(sphere, 3)
    n.set_value(x0.ravel())
    manifold_problem.minimize(n.T @ A @ n)
    add_constraint(manifold_problem, n)
    check_iterates_on_manifold(manifold_problem, 0, 3)

    assert manifold_problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    # Unit-norm equality constraint formulation
    constrained_problem = Problem()
    m = constrained_problem.decision_variable(3)
    m.set_value(x0)
    constrained_problem.minimize(m.T @ A @ m)
    constrained_problem.subject_to(m.T @ m == 1.0)
    add_constraint(constrained_problem, m)

    assert constrained_problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert n.value() == pytest.approx(m.value().ravel(), abs=solution_tolerance(10))


@pytest.mark.parametrize("initial_angle", [0.0, 1.0, 3.0])
def test_wahba(initial_angle: float) -> None:
    q_true_axis = np.array([1.0, -2.0, 0.5])
    q_true = Rotation.from_rotvec(2.5 * q_true_axis / np.linalg.norm(q_true_axis))

    body_vectors = [
        np.array([[1.0], [0.0], [0.0]]),
        np.array([[0.0], [1.0], [0.0]]),
        np.array([[0.5], [0.5], [0.7]]),
    ]

    problem = Problem()
    q = problem.decision_variable(so3)
    initial_axis = np.array([0.3, 1.0, 0.0])
    q.set_value(
        *Rotation.from_rotvec(
            initial_angle * initial_axis / np.linalg.norm(initial_axis)
        ).as_quat(scalar_first=True)
    )

    cost = Variable(0.0)
    for v in body_vectors:
        w = q_true.as_matrix() @ v
        error = VariableMatrix(v).rotate_by(q) - w
        cost += error.T @ error
    problem.minimize(cost)
    check_iterates_on_manifold(problem, 0, 4)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    # q and −q are the same rotation
    assert abs(q.value() @ q_true.as_quat(scalar_first=True)) == pytest.approx(
        1.0, abs=solution_tolerance(1)
    )


def test_rotation_averaging() -> None:
    rotations = [
        Rotation.from_rotvec(angle * np.array(axis) / np.linalg.norm(axis))
        for angle, axis in [
            (0.3, [1.0, 0.0, 0.0]),
            (0.5, [0.0, 1.0, 0.2]),
            (-0.4, [0.3, 0.3, 1.0]),
            (0.2, [-1.0, 0.5, 0.5]),
        ]
    ]

    problem = Problem()
    q = problem.decision_variable(so3)

    residuals = []
    cost = Variable(0.0)
    for rotation in rotations:
        r = (Rotation3(*rotation.as_quat(scalar_first=True)).inverse() * q).to_vector()
        cost += r.T @ r
        residuals.append(r)
    problem.minimize(cost)
    check_iterates_on_manifold(problem, 0, 4)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    # The minimizer of Σᵢ ‖log(qᵢ⁻¹q)‖² is the Karcher mean, where
    # Σᵢ log(qᵢ⁻¹q) = 0
    assert sum(r.value() for r in residuals) == pytest.approx(
        np.zeros((3, 1)), abs=solution_tolerance(10)
    )


def test_2d_point_set_registration() -> None:
    # The true angle is near π, and the initial guess is on the other side of
    # ±π, where an angle decision variable would have to wrap
    θ_true = 2.8
    R_true = np.array(
        [[math.cos(θ_true), -math.sin(θ_true)], [math.sin(θ_true), math.cos(θ_true)]]
    )
    t_true = np.array([[1.0], [-0.5]])

    points = [
        np.array([[1.0], [0.0]]),
        np.array([[0.0], [2.0]]),
        np.array([[-1.0], [-1.0]]),
        np.array([[2.0], [1.0]]),
    ]

    problem = Problem()
    z = problem.decision_variable(so2)
    t = problem.decision_variable(2)
    z.set_value(-2.8)

    cost = Variable(0.0)
    for p in points:
        measured = R_true @ p + t_true
        error = VariableMatrix(p).rotate_by(z) + t - measured
        cost += error.T @ error
    problem.minimize(cost)
    check_iterates_on_manifold(problem, 0, 2)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert z.value() == pytest.approx(θ_true, abs=solution_tolerance(1))
    assert t.value() == pytest.approx(t_true, abs=solution_tolerance(1))


def test_se2_point_set_registration() -> None:
    pose_true = Pose2(np.array([1.0, -0.5]), 2.8)

    points = [
        np.array([1.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([-1.0, -1.0]),
        np.array([2.0, 1.0]),
    ]

    problem = Problem()
    pose = problem.decision_variable(se2)
    pose.rotation().set_value(-2.8)

    cost = Variable(0.0)
    for p in points:
        measured = Translation2(Translation2(p).transform_by(pose_true).value())
        cost += Translation2(p).transform_by(pose).squared_distance(measured)
    problem.minimize(cost)
    check_iterates_on_manifold(problem, 2, 2)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert pose.rotation().value() == pytest.approx(2.8, abs=solution_tolerance(1))
    assert pose.translation().value() == pytest.approx(
        np.array([1.0, -0.5]), abs=solution_tolerance(1)
    )


def test_se3_point_set_registration() -> None:
    q_true_axis = np.array([0.2, 1.0, -0.4])
    q_true = Rotation.from_rotvec(2.5 * q_true_axis / np.linalg.norm(q_true_axis))
    t_true = np.array([1.0, -0.5, 2.0])

    points = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
    ]

    problem = Problem()
    pose = problem.decision_variable(se3)

    # The measurements are of the points in the pose's frame
    cost = Variable(0.0)
    for p in points:
        measured = Translation3(q_true.inv().apply(p - t_true))
        cost += Translation3(p).transform_by(pose.inverse()).squared_distance(measured)
    problem.minimize(cost)
    check_iterates_on_manifold(problem, 3, 4)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert pose.translation().value() == pytest.approx(
        t_true, abs=solution_tolerance(1)
    )
    assert abs(
        pose.rotation().value() @ q_true.as_quat(scalar_first=True)
    ) == pytest.approx(1.0, abs=solution_tolerance(1))


def test_pose_estimation_with_inequality_constraints() -> None:
    q_true_axis = np.array([0.2, 1.0, -0.4])
    q_true = Rotation.from_rotvec(1.2 * q_true_axis / np.linalg.norm(q_true_axis))
    t_true = np.array([[1.0], [-0.5], [2.0]])

    points = [
        np.array([[1.0], [0.0], [0.0]]),
        np.array([[0.0], [2.0], [0.0]]),
        np.array([[0.0], [0.0], [-1.0]]),
        np.array([[1.0], [1.0], [1.0]]),
    ]

    problem = Problem()
    q = problem.decision_variable(so3)
    t = problem.decision_variable(3)

    cost = Variable(0.0)
    for p in points:
        measured = q_true.as_matrix() @ p + t_true
        error = VariableMatrix(p).rotate_by(q) + t - measured
        cost += error.T @ error
    problem.minimize(cost)

    # The translation bound is active, so the solution moves off the
    # unconstrained optimum
    problem.subject_to(t[2] <= 1.5)
    problem.subject_to(t[0] >= -5.0)
    check_iterates_on_manifold(problem, 0, 4)

    assert problem.inequality_constraint_type() == ExpressionType.LINEAR
    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert t.value(2) == pytest.approx(1.5, abs=solution_tolerance(1))
    assert np.linalg.norm(q.value()) == pytest.approx(1.0, abs=rounding_tolerance(10))
