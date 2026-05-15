import math
import platform

import numpy as np
import pytest
from cart_pole_util import (
    cart_pole_dynamics_double,
    cart_pole_dynamics_variable,
)
from rk4 import rk4
from sleipnir.autodiff import ExpressionType, VariableMatrix
from sleipnir.optimization import (
    OCP,
    DynamicsType,
    ExitStatus,
    TimestepMethod,
    TranscriptionMethod,
    bounds,
)


def lerp(a, b, t):
    return a + t * (b - a)


def test_cart_pole_ocp():
    TOTAL_TIME = 5  # s
    dt = 0.05  # s
    N = int(TOTAL_TIME / dt)

    u_max = 20.0  # N
    d_max = 2.0  # m

    x_initial = np.zeros((4, 1))
    x_final = np.array([[1.0], [math.pi], [0.0], [0.0]])

    problem = OCP(
        4,
        1,
        dt,
        N,
        cart_pole_dynamics_variable,
        DynamicsType.EXPLICIT_ODE,
        TimestepMethod.VARIABLE_SINGLE,
        TranscriptionMethod.DIRECT_COLLOCATION,
    )

    # x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
    X = problem.X()

    # Initial guess
    for k in range(N + 1):
        X[0, k].set_value(lerp(x_initial[0, 0], x_final[0, 0], k / N))
        X[1, k].set_value(lerp(x_initial[1, 0], x_final[1, 0], k / N))

    # Initial conditions
    problem.constrain_initial_state(x_initial)

    # Final conditions
    problem.constrain_final_state(x_final)

    # Cart position constraints
    def each(x: VariableMatrix, u: VariableMatrix):
        problem.subject_to(bounds(0.0, x[0], d_max))

    problem.for_each_step(each)

    # Input constraints
    problem.set_lower_input_bound(-u_max)
    problem.set_upper_input_bound(u_max)

    # u = f_x
    U = problem.U()

    # Minimize sum squared inputs
    problem.minimize(sum(U[:, k : k + 1].T @ U[:, k : k + 1] for k in range(N)))

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.NONLINEAR
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    if platform.system() == "Windows" and platform.machine() == "ARM64":
        # FIXME: Fails on Windows aarch64 with "feasibility restoration failed"
        assert (
            problem.solve(diagnostics=True) == ExitStatus.FEASIBILITY_RESTORATION_FAILED
        )
        return
    else:
        assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    # Verify initial state
    assert X.value(0, 0) == pytest.approx(x_initial[0, 0], abs=1e-8)
    assert X.value(1, 0) == pytest.approx(x_initial[1, 0], abs=1e-8)
    assert X.value(2, 0) == pytest.approx(x_initial[2, 0], abs=1e-8)
    assert X.value(3, 0) == pytest.approx(x_initial[3, 0], abs=1e-8)

    # FIXME: Replay diverges
    if 0:
        # Verify solution
        x = np.zeros((4, 1))
        u = np.zeros((1, 1))
        for k in range(N):
            # Cart position constraints
            assert X[0, k] >= 0.0
            assert X[0, k] <= d_max

            # Input constraints
            assert U[0, k] >= -u_max
            assert U[0, k] <= u_max

            # Verify state
            assert X.value(0, k) == pytest.approx(x[0, 0], abs=1e-2)
            assert X.value(1, k) == pytest.approx(x[1, 0], abs=1e-2)
            assert X.value(2, k) == pytest.approx(x[2, 0], abs=1e-2)
            assert X.value(3, k) == pytest.approx(x[3, 0], abs=1e-2)

            # Project state forward
            x = rk4(cart_pole_dynamics_double, x, u, dt)

    # Verify final state
    assert X.value(0, N) == pytest.approx(x_final[0, 0], abs=1e-8)
    assert X.value(1, N) == pytest.approx(x_final[1, 0], abs=1e-8)
    assert X.value(2, N) == pytest.approx(x_final[2, 0], abs=1e-8)
    assert X.value(3, N) == pytest.approx(x_final[3, 0], abs=1e-8)

    # Log states for offline viewing
    with open("Cart-pole states.csv", "w") as f:
        f.write(
            "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),Pole angular velocity (rad/s)\n"
        )

        for k in range(N + 1):
            f.write(
                f"{k * dt},{X.value(0, k)},{X.value(1, k)},{X.value(2, k)},{X.value(3, k)}\n"
            )

    # Log inputs for offline viewing
    with open("Cart-pole inputs.csv", "w") as f:
        f.write("Time (s),Cart force (N)\n")

        for k in range(N + 1):
            if k < N:
                f.write(f"{k * dt},{U.value(0, k)}\n")
            else:
                f.write(f"{k * dt},0.0\n")
