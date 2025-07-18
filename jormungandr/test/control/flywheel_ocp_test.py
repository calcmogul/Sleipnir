import math
from typing import Callable

import numpy as np
import pytest

from jormungandr.autodiff import ExpressionType, VariableMatrix
from jormungandr.control import *
from jormungandr.optimization import ExitStatus


def near(expected, actual, tolerance):
    return abs(expected - actual) < tolerance


def flywheel_test(
    test_name: str,
    A: float,
    B: float,
    F: Callable[[VariableMatrix, VariableMatrix], VariableMatrix],
    dynamics_type: DynamicsType,
    method: TranscriptionMethod,
):
    T = 5.0  # s
    dt = 0.005  # s
    N = int(T / dt)

    # Flywheel model:
    # States: [velocity]
    # Inputs: [voltage]
    A_discrete = math.exp(A * dt)
    B_discrete = (1.0 - A_discrete) * B

    r = 10.0

    problem = OCP(1, 1, dt, N, F, dynamics_type, TimestepMethod.FIXED, method)
    problem.constrain_initial_state(0.0)
    problem.set_upper_input_bound(12)
    problem.set_lower_input_bound(-12)

    # Set up cost
    r_mat = np.full((1, N + 1), r)
    problem.minimize((r_mat - problem.X()) @ (r_mat - problem.X()).T)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    # Voltage for steady-state velocity:
    #
    # rₖ₊₁ = Arₖ + Buₖ
    # uₖ = B⁺(rₖ₊₁ − Arₖ)
    # uₖ = B⁺(rₖ − Arₖ)
    # uₖ = B⁺(I − A)rₖ
    u_ss = 1.0 / B_discrete * (1.0 - A_discrete) * r

    # Verify initial state
    assert problem.X().value(0, 0) == pytest.approx(0.0, abs=1e-8)

    # Verify solution
    x = 0.0
    u = 0.0
    for k in range(N):
        # Verify state
        assert problem.X().value(0, k) == pytest.approx(x, abs=1e-2)

        # Determine expected input for this timestep
        error = r - x
        if error > 1e-2:
            # Max control input until the reference is reached
            u = 12.0
        else:
            # Maintain speed
            u = u_ss

        # Verify input
        if (
            k > 0
            and k < N - 1
            and near(12.0, problem.U().value(0, k - 1), 1e-2)
            and near(u_ss, problem.U().value(0, k + 1), 1e-2)
        ):
            # If control input is transitioning between 12 and u_ss, ensure it's
            # within (u_ss, 12)
            assert problem.U().value(0, k) >= u_ss
            assert problem.U().value(0, k) <= 12.0
        else:
            if method == TranscriptionMethod.DIRECT_COLLOCATION:
                # The tolerance is large because the trajectory is represented by a
                # spline, and splines chatter when transitioning quickly between
                # steady-states.
                assert problem.U().value(0, k) == pytest.approx(u, abs=2.0)
            else:
                assert problem.U().value(0, k) == pytest.approx(u, abs=2e-4)

        # Project state forward
        x = A_discrete * x + B_discrete * u

    # Verify final state
    assert problem.X().value(0, N) == pytest.approx(r, abs=1e-6)

    # Log states for offline viewing
    with open("Flywheel states.csv", "w") as f:
        f.write("Time (s),Velocity (rad/s)\n")

        for k in range(N + 1):
            f.write(f"{k * dt},{problem.X().value(0, k)}\n")

    # Log inputs for offline viewing
    with open("Flywheel inputs.csv", "w") as f:
        f.write("Time (s),Voltage (V)\n")

        for k in range(N + 1):
            if k < N:
                f.write(f"{k * dt},{problem.U().value(0, k)}\n")
            else:
                f.write(f"{k * dt},0.0\n")


def test_flywheel_explicit_ocp():
    A = -1.0
    B = 1.0

    def f_ode(x: VariableMatrix, u: VariableMatrix):
        return A * x + B * u

    flywheel_test(
        "OCP Flywheel Explicit Collocation",
        A,
        B,
        f_ode,
        DynamicsType.EXPLICIT_ODE,
        TranscriptionMethod.DIRECT_COLLOCATION,
    )
    flywheel_test(
        "OCP Flywheel Explicit Transcription",
        A,
        B,
        f_ode,
        DynamicsType.EXPLICIT_ODE,
        TranscriptionMethod.DIRECT_TRANSCRIPTION,
    )
    flywheel_test(
        "OCP Flywheel Explicit Single-Shooting",
        A,
        B,
        f_ode,
        DynamicsType.EXPLICIT_ODE,
        TranscriptionMethod.SINGLE_SHOOTING,
    )


def test_flywheel_discrete_ocp():
    A = -1.0
    B = 1.0
    dt = 0.005  # s

    A_discrete = math.exp(A * dt)
    B_discrete = (1.0 - A_discrete) * B

    def f_discrete(x: VariableMatrix, u: VariableMatrix):
        return A_discrete * x + B_discrete * u

    flywheel_test(
        "OCP Flywheel Discrete Transcription",
        A,
        B,
        f_discrete,
        DynamicsType.DISCRETE,
        TranscriptionMethod.DIRECT_TRANSCRIPTION,
    )
    flywheel_test(
        "OCP Flywheel Discrete Single-Shooting",
        A,
        B,
        f_discrete,
        DynamicsType.DISCRETE,
        TranscriptionMethod.SINGLE_SHOOTING,
    )
