import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sleipnir.autodiff import Pose3, Rotation3, Twist3, Variable
from sleipnir.optimization import ExitStatus, Problem, so3
from tolerance import solution_tolerance


def test_twist_between_two_poses() -> None:
    v_true = np.array([1.0, -2.0, 0.5])
    ω_true = np.array([0.3, -1.2, 2.0])

    q_start_axis = np.array([0.0, 1.0, 1.0])
    q_start = Rotation.from_rotvec(1.0 * q_start_axis / np.linalg.norm(q_start_axis))
    start = Pose3(
        np.array([-1.0, 0.5, 2.0]), Rotation3(*q_start.as_quat(scalar_first=True))
    )
    end = start * Twist3(v_true, ω_true).exp()

    problem = Problem()
    twist = Twist3(problem.decision_variable(3), problem.decision_variable(3))
    problem.subject_to(start * twist.exp() == end)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert twist.linear().value() == pytest.approx(
        v_true.reshape(3, 1), abs=solution_tolerance(1)
    )
    assert twist.angular().value() == pytest.approx(
        ω_true.reshape(3, 1), abs=solution_tolerance(1)
    )


def test_rotation_integration() -> None:
    # Rotate from the identity to a target in N steps with the least angular
    # velocity effort, which is a constant angular velocity
    N = 10
    dt = 0.1
    rvec = np.array([0.5, 2.0, -1.0])
    target = Rotation3(rvec)

    problem = Problem()
    q = [problem.decision_variable(so3) for _ in range(N + 1)]
    ω = [problem.decision_variable(3) for _ in range(N)]

    problem.subject_to(q[0] == Rotation3(1.0, 0.0, 0.0, 0.0))
    problem.subject_to(q[N] == target)
    J = Variable(0.0)
    for k in range(N):
        problem.subject_to(q[k + 1] == q[k] * Rotation3(ω[k] * dt))
        J += (ω[k].T @ ω[k])[0]
    problem.minimize(J)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    for k in range(N):
        assert ω[k].value() == pytest.approx(
            (rvec / (N * dt)).reshape(3, 1), abs=solution_tolerance(10)
        )
