#!/usr/bin/python3

"""
Solve the guided fuel-optimal landing diversion (G-FOLD) problem using CasADi.

The coordinate system is +X up, +Y east, +Z north.

[1] Açıkmeşe et al., "Lossless Convexification of Nonconvex Control Bound and
    Pointing Constraints of the Soft Landing Optimal Control Problem", 2013.
    http://www.larsblackmore.com/iee_tcst13.pdf
[2] Açıkmeşe et al., "Convex Programming Approach to Powered Descent Guidance
    for Mars Landing", 2007. https://sci-hub.st/10.2514/1.27553
"""

import math
import sys

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm


def lerp(a, b, t):
    return a + t * (b - a)


def discretize_ab(A, B, dt):
    """
    Discretizes the given continuous A and B matrices.

    Parameter ``A``:
        Continuous system matrix.

    Parameter ``B``:
        Continuous input matrix.

    Parameter ``dt``:
        Discretization timestep.

    Returns:
        Discrete system matrix, discrete input matrix.
    """
    states = A.shape[0]
    inputs = B.shape[1]

    # M = [A  B]
    #     [0  0]
    M = expm(
        np.block([[A, B], [np.zeros((inputs, states)), np.zeros((inputs, inputs))]])
        * dt
    )

    # ϕ = eᴹᵀ = [A_d  B_d]
    #           [ 0    I ]
    return M[:states, :states], M[:states, states:]


def main():
    # From section IV of [1]:

    # Initial mass (kg)
    m_0 = 2000.0

    # Final mass (kg)
    m_f = 300.0

    # Maximum thrust (N)
    T_max = 24000

    ρ_1 = 0.2 * T_max
    ρ_2 = 0.8 * T_max

    # Fuel consumption rate (s/m)
    α = 5e-4
    assert α > 0

    # Initial position (m)
    q_0 = np.array([[2400.0, 450.0, -330.0]]).T

    # Initial velocity (m/s)
    v_0 = np.array([[-10.0, -40.0, 10.0]]).T

    # Final position (m)
    q_f = np.array([[0.0, 0.0, 0.0]]).T

    # Final velocity (m/s)
    v_f = np.array([[0.0, 0.0, 0.0]]).T

    # Gravitational acceleration on Mars (m/s²)
    g = np.array([[-3.71, 0.0, 0.0]]).T

    # Constant angular velocity of planet (rad/s)
    ω = np.array([[2.53e-5, 0.0, 6.62e-5]]).T

    # Thrust pointing limit (rad)
    θ = math.radians(90)
    assert θ >= 0 and θ <= math.pi / 2

    # Minimum glide slope
    γ_gs = math.radians(30)
    assert γ_gs >= 0 and γ_gs <= math.pi / 2

    # Maximum velocity magnitude (m/s)
    v_max = 90.0

    # Time between control intervals (s)
    dt = 0.5

    # See equation (2) of [1].

    #     [0   -ω₃  ω₂]
    # S = [ω₃   0  −ω₁]
    #     [−ω₂  ω₁  0 ]
    ω_1 = ω[0, 0]
    ω_2 = ω[1, 0]
    ω_3 = ω[2, 0]
    S = np.array([[0.0, -ω_3, ω_2], [ω_3, 0.0, -ω_1], [-ω_2, ω_1, 0.0]])

    #     [  0        I  ]
    # A = [-S(ω)²  -2S(ω)]
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-S @ S, -2 * S]])

    #     [0]
    # B = [I]
    B = np.block([[np.zeros((3, 3))], [np.eye(3)]])

    A_d, B_d = discretize_ab(A, B, dt)

    # Time horizon bounds (s)
    #
    # See equation (55) of [2].
    t_min = (m_0 - m_f) * norm(v_0) / ρ_2
    t_max = m_f / (α * ρ_1)

    # Number of control intervals
    #
    # See equation (57) of [2].
    N_min = math.ceil(t_min / dt)
    N_max = math.floor(t_max / dt)

    # Bisect to find minimum feasible N
    print(f"Searching N ∈ [{N_min}, {N_max}] for smallest feasible N")
    X_value = np.zeros((6, 2))
    Z_value = np.zeros((1, 2))
    U_value = np.zeros((3, 1))
    σ_value = np.zeros((1, 1))
    found = False
    while not found:
        N = N_min + math.floor((N_max - N_min) / 2)

        # TODO: Minimum feasible N from CasADi. Debugging only. Remove this.
        N = 84

        print(f"Trying N = {N} from [{N_min}, {N_max}]...", end="")
        sys.stdout.flush()

        problem = ca.Opti()

        # x = [position, velocity]ᵀ
        X = problem.variable(6, N + 1)
        # z = ln(m)
        Z = problem.variable(1, N + 1)
        # u = T_c/m
        U = problem.variable(3, N)
        # σ = Γ/m
        σ = problem.variable(1, N)

        q = X[:3, :]
        v = X[3:6, :]

        # Initial position
        problem.subject_to(q[:, :1] == q_0)

        # Initial velocity
        problem.subject_to(v[:, :1] == v_0)

        # Initial ln(mass)
        problem.subject_to(Z[0, 0] == math.log(m_0))

        # Final x position
        problem.subject_to(q[0, -1] == q_f[0, 0])

        # Final velocity
        problem.subject_to(v[:, -1] == v_f)

        # Position and velocity initial guesses
        for k in range(N + 1):
            for i in range(3):
                problem.set_initial(q[i, k], lerp(q_0[i, 0], q_f[i, 0], k / N))
                problem.set_initial(v[i, k], lerp(v_0[i, 0], v_f[i, 0], k / N))

        # End straight
        problem.subject_to(U[0, -1] >= σ[-1])
        problem.subject_to(U[1, -1] == 0)
        problem.subject_to(U[2, -1] == 0)

        # State, input, and dynamics constraints
        for k in range(N + 1):
            t = k * dt

            x_k = X[:, k : k + 1]
            q_k = X[:3, k : k + 1]
            v_k = X[3:6, k : k + 1]
            z_k = Z[:, k : k + 1]

            # Mass limits
            z_min = math.log(m_0 - α * ρ_2 * t)
            z_max = math.log(m_0 - α * ρ_1 * t)
            problem.subject_to(problem.bounded(z_min, z_k, z_max))
            z_estimate = (z_min + z_max) / 2
            problem.set_initial(z_k, z_estimate)

            # Glide slope constraint, which ensure the trajectory isn't too shallow
            # or goes below the target height
            #
            # See equation (12) of [1].
            #
            #       [0  1  0]
            #   E = [0  0  1]
            #
            #                      [1/tan(γ_gs)]
            #   c = e₁/tan(γ_gs) = [     0     ]
            #                      [     0     ]
            #
            #   |E(r - r_f)|₂ - cᵀ(r - r_f) ≤ 0                            (12)
            #
            #   hypot((r − r_f)₂, (r − r_f)₃) − (r − r_f)₁/tan(γ_gs) ≤ 0
            #   hypot((r − r_f)₂, (r − r_f)₃) ≤ (r − r_f)₁/tan(γ_gs)
            #   (r − r_f)₁/tan(γ_gs) ≥ hypot((r − r_f)₂, (r − r_f)₃)
            #   (r − r_f)₁²/tan²(γ_gs) ≥ (r − r_f)₂² + (r − r_f)₃²
            #   (r − r_f)₁² ≥ tan²(γ_gs)((r − r_f)₂² + (r − r_f)₃²)
            problem.subject_to(
                (q_k[0, 0] - q_f[0, 0]) ** 2
                >= math.tan(γ_gs) ** 2
                * ((q_k[1, 0] - q_f[1, 0]) ** 2 + (q_k[2, 0] - q_f[2, 0]) ** 2)
            )

            # Velocity limits
            problem.subject_to(v_k.T @ v_k <= v_max**2)

            if k < N:
                x_k1 = X[:, k + 1 : k + 2]
                z_k1 = Z[:, k + 1 : k + 2]
                u_k = U[:, k : k + 1]
                σ_k = σ[:, k : k + 1]

                # Input initial guess
                #
                #   ρ₁ ≤ |T_c| ≤ ρ₂
                #   ρ₁ ≤ |u| exp(z) ≤ ρ₂
                #   ρ₁/exp(z) ≤ |u| ≤ ρ₂/exp(z)
                u_min = ρ_1 / math.exp(z_estimate)
                u_max = ρ_2 / math.exp(z_estimate)
                problem.set_initial(u_k, np.array([[(u_min + u_max) / 2, 0.0, 0.0]]).T)

                problem.subject_to(σ_k >= 0)

                # Thrust magnitude limit
                #
                # See equation (34) of [1].
                #
                #   |u|₂ ≤ σ
                #   u_x² + u_y² + u_z² ≤ σ²
                problem.subject_to(u_k.T @ u_k <= σ_k**2)

                # Thrust pointing limit
                #
                # See equation (34) of [1].
                #
                #   n̂ᵀu ≥ cos(θ)σ where n̂ = [1  0  0]ᵀ
                #   [1  0  0]u ≥ cos(θ)σ
                #   u_x ≥ cos(θ)σ
                problem.subject_to(u_k[0, 0] >= math.cos(θ) * σ_k)

                # Thrust slack limits
                #
                # See equation (34) of [2].
                z_0 = math.log(m_0 - α * ρ_2 * t)
                μ_1 = ρ_1 * math.exp(-z_0)
                μ_2 = ρ_2 * math.exp(-z_0)
                σ_min = μ_1 * (1 - (z_k - z_0) + 0.5 * (z_k - z_0) ** 2)
                σ_max = μ_2 * (1 - (z_k - z_0))
                problem.subject_to(problem.bounded(σ_min, σ_k, σ_max))

                # σₘᵢₙ = μ₁ (1 - (zₖ - z₀) + 0.5 (zₖ - z₀)²)
                # σₘₐₓ = μ₂ (1 - (zₖ - z₀))
                # (σₘᵢₙ + σₘₐₓ)/2 = (μ₁ (1 - (zₖ - z₀) + 0.5 (zₖ - z₀)²) +
                #                    μ₂ (1 - (zₖ - z₀))
                #                   )/2
                # (σₘᵢₙ + σₘₐₓ)/2 = ((μ₁ + μ₂)(1 - (zₖ - z₀)) + 0.5μ₁ (zₖ - z₀)²)/2
                # (σₘᵢₙ + σₘₐₓ)/2 = 0.5 (μ₁ + μ₂)(1 - (zₖ - z₀)) + 0.25μ₁ (zₖ - z₀)²
                problem.set_initial(
                    σ_k,
                    0.5 * (μ_1 + μ_2) * (1 - (z_estimate - z_0))
                    + 0.25 * μ_1 * (z_estimate - z_0) ** 2,
                )

                # Integrate dynamics
                #
                # See equation (2) of [1].
                #
                #   ẋ = Ax + B(g + u)
                #   ż = −ασ
                #
                #   xₖ₊₁ = A_d xₖ + B_d(g + uₖ)
                #   zₖ₊₁ = zₖ - αTσₖ
                problem.subject_to(x_k1 == A_d @ x_k + B_d @ (g + u_k))
                problem.subject_to(z_k1 == z_k - α * dt * σ_k)

        # Problem 4 from [1]: Minimum fuel
        problem.minimize(sum([σ[k] for k in range(N)]))
        problem.solver("ipopt", {"print_time": 0}, {"print_level": 0, "sb": "yes"})
        try:
            sol = problem.solve()

            print(" feasible")

            X_value = sol.value(X)
            U_value = sol.value(U)
            Z_value = sol.value(Z)
            σ_value = sol.value(σ)

            # Problem is feasible, so try a smaller N
            if N_min < N:
                N_max = N - 1
            else:
                print(f"Smallest feasible N = {N}")
                found = True
        except:
            print(" infeasible")

            # Problem is infeasible, so try a larger N
            if N_min < N_max:
                N_min = N + 1
            else:
                print(f"Smallest feasible N = {N + 1}")
                found = True

        # TODO: Debugging only. Remove this.
        X_value = sol.value(X)
        U_value = sol.value(U)
        Z_value = sol.value(Z)
        σ_value = sol.value(σ)
        break

    N = X_value.shape[1] - 1
    times = [k * dt for k in range(N + 1)]

    plt.figure()
    ax = plt.gca()
    ax.set_title("Position vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.plot(times, X_value[0, :], label="x (up)")
    ax.plot(times, X_value[1, :], label="y (east)")
    ax.plot(times, X_value[2, :], label="z (north)")
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("Velocity vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.plot(times, X_value[3, :], label="x (up)")
    ax.plot(times, X_value[4, :], label="y (east)")
    ax.plot(times, X_value[5, :], label="z (north)")
    ax.plot(times, [norm(X_value[3:6, k]) for k in range(N + 1)], label="Total")
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("Mass vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass (kg)")
    ax.plot(times, [m_0 - α * ρ_2 * k * dt for k in range(N + 1)], label="Min mass")
    ax.plot(times, [m_0 - α * ρ_1 * k * dt for k in range(N + 1)], label="Max mass")
    ax.plot(times, [math.exp(Z_value[k]) for k in range(N + 1)], label="Mass")
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("u vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("u")
    ax.plot(times[:-1], U_value[0, :], label="x (up)")
    ax.plot(times[:-1], U_value[1, :], label="y (east)")
    ax.plot(times[:-1], U_value[2, :], label="z (north)")
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("σ vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("σ")
    ax.plot(times[:-1], σ_value.T)

    # Solve for |T_c|/T_max
    #
    # u = T_c/m
    # T_c = um
    # T_c = u exp(z)
    # |T_c| = |u| exp(z)
    # |T_c|/T_max = |u| exp(z)/T_max
    plt.figure()
    ax = plt.gca()
    ax.set_title("% Throttle vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("% Throttle")
    ax.plot(times[:-1], [ρ_1 / T_max * 100 for k in range(N)], label="Min thrust")
    ax.plot(times[:-1], [ρ_2 / T_max * 100 for k in range(N)], label="Max thrust")
    ax.plot(
        times[:-1],
        [norm(U_value[:, k]) * math.exp(Z_value[k]) / T_max * 100 for k in range(N)],
        label="Thrust",
    )
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("Angle vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.plot(
        times,
        [
            math.degrees(
                math.atan2(X_value[0, k], math.hypot(X_value[1, k], X_value[2, k]))
            )
            for k in range(N + 1)
        ],
        label="Glide angle",
    )
    ax.plot(
        times[:-1],
        [
            math.degrees(
                math.acos(U_value[0, k] / norm(U_value[:, k]))
                if norm(U_value[:, k]) > 1e-2
                else 0
            )
            for k in range(N)
        ],
        label="Thrust angle from vertical",
    )
    ax.legend()

    plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(vertical_axis="x")

    # Minimum glide slope cone
    h = q_0[0, 0] - q_f[0, 0]
    θs, Rs = np.meshgrid(
        np.linspace(0, 2 * math.pi, 100), np.linspace(0, h / math.tan(γ_gs), 100)
    )
    ax.plot_surface(
        Rs * math.tan(γ_gs) + q_f[2, 0],  # x
        Rs * np.sin(θs) + q_f[1, 0],  # y
        Rs * np.cos(θs) + q_f[0, 0],  # z
        alpha=0.2,
        label="Minimum glide slope",
    )

    # Rocket trajectory
    ax.plot(X_value[0, :], X_value[1, :], X_value[2, :], label="Trajectory")

    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
