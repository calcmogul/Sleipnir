"""This problem tests the case where regularization fails"""

import math

import numpy as np
import sleipnir.autodiff as autodiff
from sleipnir.autodiff import ExpressionType
from sleipnir.optimization import ExitStatus, Problem, bounds


def test_arm_on_elevator_problem():
    N = 800

    ELEVATOR_START_HEIGHT = 1.0  # m
    ELEVATOR_END_HEIGHT = 1.25  # m
    ELEVATOR_MAX_VELOCITY = 1  # m/s
    ELEVATOR_MAX_ACCELERATION = 2.0  # m/s²

    ARM_LENGTH = 1.0  # m
    ARM_START_ANGLE = 0.0  # rad
    ARM_END_ANGLE = math.pi  # rad
    ARM_MAX_VELOCITY = 2.0 * math.pi  # rad/s
    ARM_MAX_ACCELERATION = 4.0 * math.pi  # rad/s²

    END_EFFECTOR_MAX_HEIGHT = 1.8  # m

    TOTAL_TIME = 4.0  # s
    dt = TOTAL_TIME / N

    problem = Problem()

    elevator = problem.decision_variable(2, N + 1)
    elevator_accel = problem.decision_variable(1, N)

    arm = problem.decision_variable(2, N + 1)
    arm_accel = problem.decision_variable(1, N)

    for k in range(N):
        # Elevator dynamics constraints
        problem.subject_to(
            elevator[0, k + 1]
            == elevator[0, k] + elevator[1, k] * dt + 0.5 * elevator_accel[0, k] * dt**2
        )
        problem.subject_to(
            elevator[1, k + 1] == elevator[1, k] + elevator_accel[0, k] * dt
        )

        # Arm dynamics constraints
        problem.subject_to(
            arm[0, k + 1] == arm[0, k] + arm[1, k] * dt + 0.5 * arm_accel[0, k] * dt**2
        )
        problem.subject_to(arm[1, k + 1] == arm[1, k] + arm_accel[0, k] * dt)

    # Elevator start and end conditions
    problem.subject_to(elevator[:, :1] == np.array([[ELEVATOR_START_HEIGHT], [0.0]]))
    problem.subject_to(
        elevator[:, N : N + 1] == np.array([[ELEVATOR_END_HEIGHT], [0.0]])
    )

    # Arm start and end conditions
    problem.subject_to(arm[:, :1] == np.array([[ARM_START_ANGLE], [0.0]]))
    problem.subject_to(arm[:, N : N + 1] == np.array([[ARM_END_ANGLE], [0.0]]))

    # Elevator velocity limits
    problem.subject_to(
        bounds(-ELEVATOR_MAX_VELOCITY, elevator[1:2, :], ELEVATOR_MAX_VELOCITY)
    )

    # Elevator acceleration limits
    problem.subject_to(
        bounds(-ELEVATOR_MAX_ACCELERATION, elevator_accel, ELEVATOR_MAX_ACCELERATION)
    )

    # Arm velocity limits
    problem.subject_to(bounds(-ARM_MAX_VELOCITY, arm[1:2, :], ARM_MAX_VELOCITY))

    # Arm acceleration limits
    problem.subject_to(bounds(-ARM_MAX_ACCELERATION, arm_accel, ARM_MAX_ACCELERATION))

    # Height limit
    heights = elevator[:1, :] + ARM_LENGTH * arm[:1, :].cwise_map(autodiff.sin)
    problem.subject_to(heights <= END_EFFECTOR_MAX_HEIGHT)

    # Cost function
    problem.minimize(
        sum(
            (ELEVATOR_END_HEIGHT - elevator[0, k]) ** 2
            + (ARM_END_ANGLE - arm[0, k]) ** 2
            for k in range(N + 1)
        )
    )

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.NONLINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS
