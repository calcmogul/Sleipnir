#!/usr/bin/env python

import math

import numpy as np
from sleipnir.optimization import Problem, bounds


def main() -> None:
    T = 4.0  # s
    T_horizon = 1.0  # s
    dt = 0.05  # s
    N = int(T_horizon / dt)

    # Flywheel model:
    # States: [velocity]
    # Inputs: [voltage]
    A = math.exp(-dt)
    B = 1.0 - math.exp(-dt)

    # The problem is set up once. Each timestep only changes the values of its
    # parameters, so every solve after the first is warm-started from the
    # previous solve's solution.
    problem = Problem()
    X = problem.decision_variable(1, N + 1)
    U = problem.decision_variable(1, N)

    # Current state
    x_0 = problem.parameter()

    # Reference
    r = problem.parameter()

    # Dynamics constraint
    for k in range(N):
        problem.subject_to(X[0, k + 1] == A * X[0, k] + B * U[0, k])

    # State and input constraints
    problem.subject_to(X[0, 0] == x_0)
    problem.subject_to(bounds(-12, U, 12))

    # Cost function - minimize error
    problem.minimize(sum((r - X[0, k]) * (r - X[0, k]) for k in range(N + 1)))

    x_plant = 0.0

    for t in np.arange(0.0, T, dt):
        # The reference changes halfway through
        r.set_value(10.0 if t < 2.0 else 5.0)

        # A disturbance slows the flywheel down
        if t == 1.0:
            x_plant -= 3.0

        x_0.set_value(x_plant)

        problem.solve()

        # Apply the first input to the plant
        u = U.value(0, 0)
        x_plant = A * x_plant + B * u


if __name__ == "__main__":
    main()
