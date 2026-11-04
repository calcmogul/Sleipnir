import math

import numpy as np
import pytest
from cart_pole_util import (
    cart_pole_dynamics_double,
    cart_pole_dynamics_variable,
)
from sleipnir.autodiff import ExpressionType, VariableMatrix
from sleipnir.optimization import (
    OCP,
    DynamicsType,
    ExitStatus,
    TimestepMethod,
    TranscriptionMethod,
    bounds,
)
from tolerance import SolutionType, solution_tolerance


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def test_cart_pole_ocp() -> None:
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
    def each(x: VariableMatrix, u: VariableMatrix) -> None:
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

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    # Verify initial state
    assert X.value(0, 0) == pytest.approx(x_initial[0, 0], abs=solution_tolerance(1))
    assert X.value(1, 0) == pytest.approx(x_initial[1, 0], abs=solution_tolerance(1))
    assert X.value(2, 0) == pytest.approx(x_initial[2, 0], abs=solution_tolerance(1))
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
        #
        # Direct collocation constrains the system dynamics at the midpoint of a
        # cubic Hermite spline through each pair of adjacent states.
        f = cart_pole_dynamics_double
        h = problem.dt().value(0, k)
        x_begin = X[:, k : k + 1].value()
        x_end = X[:, k + 1 : k + 2].value()
        u_begin = U[:, k : k + 1].value()
        u_end = U[:, k + 1 : k + 2].value()

        xdot_begin = f(x_begin, u_begin)
        xdot_end = f(x_end, u_end)
        xdot_c = -3.0 / (2.0 * h) * (x_begin - x_end) - 0.25 * (xdot_begin + xdot_end)

        x_c = 0.5 * (x_begin + x_end) + h / 8.0 * (xdot_begin - xdot_end)
        u_c = 0.5 * (u_begin + u_end)

        expected_xdot_c = f(x_c, u_c)
        for row in range(xdot_c.shape[0]):
            assert xdot_c[row, 0] == pytest.approx(
                expected_xdot_c[row, 0], abs=solution_tolerance(1)
            )

    # Verify final state
    assert X.value(0, N) == pytest.approx(
        x_final[0, 0], abs=solution_tolerance(1, SolutionType.DEGENERATE)
    )
    assert X.value(1, N) == pytest.approx(x_final[1, 0], abs=solution_tolerance(1))
    assert X.value(2, N) == pytest.approx(x_final[2, 0], abs=solution_tolerance(1))
    assert X.value(3, N) == pytest.approx(x_final[3, 0], abs=solution_tolerance(1))

    # Log states for offline viewing
    with open("Cart-pole states.csv", "w") as f:
        f.write(
            "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),Pole angular velocity (rad/s)\n"
        )

        time = 0.0
        for k in range(N + 1):
            f.write(
                f"{time},{X.value(0, k)},{X.value(1, k)},{X.value(2, k)},{X.value(3, k)}\n"
            )

            time += problem.dt().value(0, k)

    # Log inputs for offline viewing
    with open("Cart-pole inputs.csv", "w") as f:
        f.write("Time (s),Cart force (N)\n")

        time = 0.0
        for k in range(N + 1):
            f.write(f"{time},{U.value(0, k)}\n")

            time += problem.dt().value(0, k)
