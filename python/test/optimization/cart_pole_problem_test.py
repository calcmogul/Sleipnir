import math

import numpy as np
import pytest
from cart_pole_util import (
    cart_pole_dynamics_double,
    cart_pole_dynamics_variable,
)
from rk4 import rk4
from sleipnir.autodiff import ExpressionType
from sleipnir.optimization import ExitStatus, Problem, bounds
from tolerance import SolutionType, solution_tolerance


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


# dt = 0.25 s is sensitive to small solver changes, so a failure here may just
# mean the iterates found a different basin rather than a regression.
@pytest.mark.parametrize("dt", [0.05, 0.125, 0.25])
def test_cart_pole_problem(dt: float) -> None:
    TOTAL_TIME = 5.0  # s
    N = int(TOTAL_TIME / dt)

    u_max = 20.0  # N
    d_max = 2.0  # m

    x_initial = np.zeros((4, 1))
    x_final = np.array([[1.0], [math.pi], [0.0], [0.0]])

    problem = Problem()

    # x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
    X = problem.decision_variable(4, N + 1)

    # Initial guess
    for k in range(N + 1):
        X[0, k].set_value(lerp(x_initial[0, 0], x_final[0, 0], k / N))
        X[1, k].set_value(lerp(x_initial[1, 0], x_final[1, 0], k / N))

    # u = f_x
    U = problem.decision_variable(1, N)

    # Initial conditions
    problem.subject_to(X[:, :1] == x_initial)

    # Final conditions
    problem.subject_to(X[:, N : N + 1] == x_final)

    # Cart position constraints
    problem.subject_to(bounds(0.0, X[:1, :], d_max))

    # Input constraints
    problem.subject_to(bounds(-u_max, U, u_max))

    # Dynamics constraints - RK4 integration
    for k in range(N):
        problem.subject_to(
            X[:, k + 1 : k + 2]
            == rk4(cart_pole_dynamics_variable, X[:, k : k + 1], U[:, k : k + 1], dt)
        )

    # Minimize sum squared inputs
    problem.minimize(sum(U[:, k : k + 1].T @ U[:, k : k + 1] for k in range(N)))

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.NONLINEAR
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    # Verify initial state
    assert X.value(0, 0) == pytest.approx(x_initial[0, 0], abs=solution_tolerance(1))
    assert X.value(1, 0) == pytest.approx(
        x_initial[1, 0], abs=solution_tolerance(1, SolutionType.DEGENERATE)
    )
    assert X.value(2, 0) == pytest.approx(
        x_initial[2, 0], abs=solution_tolerance(1, SolutionType.DEGENERATE)
    )
    assert X.value(3, 0) == pytest.approx(x_initial[3, 0], abs=solution_tolerance(1))

    # Constraint residuals are part of the KKT error, so the solver only guarantees
    # bounds to within its tolerance. For example, X(0, 0) is pinned to its lower
    # bound by the initial state, so its sign is rounding error.
    bound_tolerance = solution_tolerance(1)

    # Verify solution
    for k in range(N):
        # Cart position constraints
        assert X.value(0, k) >= -bound_tolerance
        assert X.value(0, k) <= d_max + bound_tolerance

        # Input constraints
        assert U.value(0, k) >= -u_max - bound_tolerance
        assert U.value(0, k) <= u_max + bound_tolerance

        # Dynamics constraints
        expected_x_k1 = rk4(
            cart_pole_dynamics_double,
            X[:, k : k + 1].value(),
            U[:, k : k + 1].value(),
            dt,
        )
        actual_x_k1 = X[:, k + 1 : k + 2].value()
        for row in range(actual_x_k1.shape[0]):
            assert actual_x_k1[row, 0] == pytest.approx(
                expected_x_k1[row, 0], abs=solution_tolerance(1)
            )

    # Verify final state
    assert X.value(0, N) == pytest.approx(
        x_final[0, 0], abs=solution_tolerance(1, SolutionType.DEGENERATE)
    )
    assert X.value(1, N) == pytest.approx(
        x_final[1, 0], abs=solution_tolerance(1, SolutionType.DEGENERATE)
    )
    assert X.value(2, N) == pytest.approx(
        x_final[2, 0], abs=solution_tolerance(1, SolutionType.DEGENERATE)
    )
    assert X.value(3, N) == pytest.approx(
        x_final[3, 0], abs=solution_tolerance(1, SolutionType.DEGENERATE)
    )

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
