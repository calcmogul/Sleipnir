import math

import numpy as np
import pytest

from jormungandr.autodiff import ExpressionType
from jormungandr.optimization import ExitStatus, Problem


def near(expected, actual, tolerance):
    return abs(expected - actual) < tolerance


def test_flywheel_problem():
    T = 5.0
    dt = 0.005
    N = int(T / dt)

    # Flywheel model:
    # States: [velocity]
    # Inputs: [voltage]
    A = math.exp(-dt)
    B = 1.0 - math.exp(-dt)

    problem = Problem()
    X = problem.decision_variable(1, N + 1)
    U = problem.decision_variable(1, N)

    # Dynamics constraint
    for k in range(N):
        problem.subject_to(
            X[:, k + 1 : k + 2] == A * X[:, k : k + 1] + B * U[:, k : k + 1]
        )

    # State and input constraints
    problem.subject_to(X[0, 0] == 0.0)
    problem.subject_to(-12 <= U)
    problem.subject_to(U <= 12)

    # Cost function - minimize error
    r = np.array([[10.0]])
    J = 0.0
    for k in range(N + 1):
        J += (r - X[:, k : k + 1]).T @ (r - X[:, k : k + 1])
    problem.minimize(J)

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
    u_ss = 1.0 / B * (1.0 - A) * r[0, 0]

    # Verify initial state
    assert X.value(0, 0) == pytest.approx(0.0, abs=1e-8)

    # Verify solution
    x = 0.0
    u = 0.0
    for k in range(N):
        # Verify state
        assert X.value(0, k) == pytest.approx(x, abs=1e-2)

        # Determine expected input for this timestep
        error = r[0, 0] - x
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
            and near(12.0, U.value(0, k - 1), 1e-2)
            and near(u_ss, U.value(0, k + 1), 1e-2)
        ):
            # If control input is transitioning between 12 and u_ss, ensure it's
            # within (u_ss, 12)
            assert U.value(0, k) >= u_ss
            assert U.value(0, k) <= 12.0
        else:
            assert U.value(0, k) == pytest.approx(u, abs=2e-4)

        # Project state forward
        x = A * x + B * u

    # Verify final state
    assert X.value(0, N) == pytest.approx(r[0, 0], abs=1e-7)

    # Log states for offline viewing
    with open("Flywheel states.csv", "w") as f:
        f.write("Time (s),Velocity (rad/s)\n")

        for k in range(N + 1):
            f.write(f"{k * dt},{X.value(0, k)}\n")

    # Log inputs for offline viewing
    with open("Flywheel inputs.csv", "w") as f:
        f.write("Time (s),Voltage (V)\n")

        for k in range(N + 1):
            if k < N:
                f.write(f"{k * dt},{U.value(0, k)}\n")
            else:
                f.write(f"{k * dt},0.0\n")
