import math

import numpy as np
import pytest
from sleipnir.autodiff import Pose2, Real, RealMatrix, Twist2
from sleipnir.optimization import ExitStatus, Problem
from tolerance import solution_tolerance


def test_pose2_integration() -> None:
    # Drive from the origin to a pose facing backward in N steps of a
    # differential drive robot, whose body-frame velocity has no sideways
    # component, with the least effort
    N = 20
    dt = 0.1
    target = Pose2(np.array([2.0, 1.0]), math.pi)

    problem = Problem()
    poses = [problem.pose2_decision_variable() for _ in range(N + 1)]
    v = [problem.real_decision_variable() for _ in range(N)]
    ω = [problem.real_decision_variable() for _ in range(N)]

    problem.subject_to(poses[0] == Pose2(np.zeros(2), 0.0))
    problem.subject_to(poses[N] == target)
    J = Real(0.0)
    for k in range(N):
        twist = Twist2(RealMatrix([[v[k]], [Real(0.0)]]), ω[k])
        problem.subject_to(poses[k + 1] == poses[k] * (twist * dt).exp())
        J += v[k] * v[k] + ω[k] * ω[k]
    problem.minimize(J)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert abs(poses[N].rotation().value()) == pytest.approx(
        math.pi, abs=solution_tolerance(1)
    )
    assert poses[N].translation().x().value() == pytest.approx(
        2.0, abs=solution_tolerance(1)
    )
