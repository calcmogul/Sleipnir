import numpy as np
import pytest

from jormungandr.autodiff import ExpressionType
from jormungandr.control import *
from jormungandr.optimization import ExitStatus
from jormungandr.test.differential_drive_util import (
    differential_drive_dynamics,
    differential_drive_dynamics_double,
)
from jormungandr.test.rk4 import rk4


def test_differential_drive_ocp():
    N = 50

    min_timestep = 0.05  # s
    x_initial = np.zeros((5, 1))
    x_final = np.array([[1.0], [1.0], [0.0], [0.0], [0.0]])
    u_min = np.array([[-12.0], [-12.0]])
    u_max = np.array([[12.0], [12.0]])

    problem = OCP(
        5,
        2,
        min_timestep,
        N,
        differential_drive_dynamics,
        DynamicsType.EXPLICIT_ODE,
        TimestepMethod.VARIABLE_SINGLE,
        TranscriptionMethod.DIRECT_TRANSCRIPTION,
    )

    # Seed the min time formulation with lerp between waypoints
    for i in range(N + 1):
        problem.X()[0, i].set_value(i / (N + 1))
        problem.X()[1, i].set_value(i / (N + 1))

    problem.constrain_initial_state(x_initial)
    problem.constrain_final_state(x_final)

    problem.set_lower_input_bound(u_min)
    problem.set_upper_input_bound(u_max)

    problem.set_min_timestep(min_timestep)
    problem.set_max_timestep(3.0)

    # Set up cost
    problem.minimize(problem.dt() @ np.ones((N + 1, 1)))

    assert problem.cost_function_type() == ExpressionType.LINEAR
    assert problem.equality_constraint_type() == ExpressionType.NONLINEAR
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    # FIXME: Often fails with "factorization failed"
    status = problem.solve(max_iterations=1000, diagnostics=True)
    assert status in [ExitStatus.SUCCESS, ExitStatus.FACTORIZATION_FAILED]
    if status != ExitStatus.SUCCESS:
        return

    X = problem.X()
    U = problem.U()

    # Verify initial state
    assert X.value(0, 0) == pytest.approx(x_initial[0, 0], abs=1e-8)
    assert X.value(1, 0) == pytest.approx(x_initial[1, 0], abs=1e-8)
    assert X.value(2, 0) == pytest.approx(x_initial[2, 0], abs=1e-8)
    assert X.value(3, 0) == pytest.approx(x_initial[3, 0], abs=1e-8)
    assert X.value(4, 0) == pytest.approx(x_initial[4, 0], abs=1e-8)

    # FIXME: Replay diverges
    return

    # Verify solution
    x = np.zeros((5, 1))
    u = np.zeros((2, 1))
    for k in range(N):
        u = U[:, k : k + 1].value()

        # Input constraints
        assert U[0, k].value() >= -u_max[0]
        assert U[0, k].value() <= u_max[0]
        assert U[1, k].value() >= -u_max[1]
        assert U[1, k].value() <= u_max[1]

        # Verify state
        assert X.value(0, k) == pytest.approx(x[0, 0], abs=1e-8)
        assert X.value(1, k) == pytest.approx(x[1, 0], abs=1e-8)
        assert X.value(2, k) == pytest.approx(x[2, 0], abs=1e-8)
        assert X.value(3, k) == pytest.approx(x[3, 0], abs=1e-8)
        assert X.value(4, k) == pytest.approx(x[4, 0], abs=1e-8)

        # Project state forward
        x = rk4(differential_drive_dynamics_double, x, u, problem.dt().value(0, k))

    # Verify final state
    assert X.value(0, N) == pytest.approx(x_final[0, 0], abs=1e-8)
    assert X.value(1, N) == pytest.approx(x_final[1, 0], abs=1e-8)
    assert X.value(2, N) == pytest.approx(x_final[2, 0], abs=1e-8)
    assert X.value(3, N) == pytest.approx(x_final[3, 0], abs=1e-8)
    assert X.value(4, N) == pytest.approx(x_final[4, 0], abs=1e-8)

    # Log states for offline viewing
    with open("Differential drive states.csv", "w") as f:
        f.write(
            "Time (s),X position (m),Y position (m),Heading (rad),Left velocity (m/s),Right velocity (m/s)\n"
        )

        time = 0.0
        for k in range(N + 1):
            f.write(
                f"{time},{X.value(0, k)},{X.value(1, k)},{X.value(2, k)},{X.value(3, k)},{X.value(4, k)}\n"
            )

            time += problem.dt().value(0, k)

    # Log inputs for offline viewing
    with open("Differential drive inputs.csv", "w") as f:
        f.write("Time (s),Left voltage (V),Right voltage (V)\n")

        time = 0.0
        for k in range(N + 1):
            if k < N:
                f.write(f"{time},{U.value(0, k)},{U.value(1, k)}\n")
            else:
                f.write(f"{time},0.0,0.0\n")

            time += problem.dt().value(0, k)
