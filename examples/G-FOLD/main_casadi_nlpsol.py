#!/usr/bin/python3

"""
Solves the guided fuel-optimal landing diversion (G-FOLD) problem using CasADi's
nlpsol() function.

The coordinate system is +X up, +Y east, +Z north.

[1] Açıkmeşe et al., "Lossless Convexification of Nonconvex Control Bound and
    Pointing Constraints of the Soft Landing Optimal Control Problem", 2013.
    http://www.larsblackmore.com/iee_tcst13.pdf
[2] Açıkmeşe et al., "Convex Programming Approach to Powered Descent Guidance
    for Mars Landing", 2007. https://sci-hub.st/10.2514/1.27553
"""

import math

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm


def discretize_ab(A, B, dt):
    """Discretizes the given continuous A and B matrices.

    Keyword arguments:
    A -- continuous system matrix
    B -- continuous input matrix
    dt -- discretization timestep

    Returns:
    discrete system matrix, discrete input matrix
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
θ = math.radians(60)
assert θ > 0 and θ < math.pi / 2

# Minimum glide slope
γ_gs = math.radians(30)
assert γ_gs > 0 and γ_gs < math.pi / 2

# Maximum velocity magnitude (m/s)
v_max = 90.0

# Time between control intervals (s)
dt = 1.0

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
N = int((N_min + N_max) / 2)

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


def plot_solution(X: np.array, Z: np.array, U: np.array):
    times = [k * dt for k in range(N + 1)]

    plt.figure()
    ax = plt.gca()
    ax.set_title("Position vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.plot(times, X[0, :], label="x (up)")
    ax.plot(times, X[1, :], label="y (east)")
    ax.plot(times, X[2, :], label="z (north)")
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("Mass vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass (kg)")
    ax.plot(times, [m_0 - α * ρ_2 * k * dt for k in range(N + 1)], label="Min mass")
    ax.plot(times, [math.exp(Z[0, k]) for k in range(N + 1)], label="Mass")
    ax.plot(times, [m_0 - α * ρ_1 * k * dt for k in range(N + 1)], label="Max mass")
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("Velocity vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.plot(times, X[3, :], label="x (up)")
    ax.plot(times, X[4, :], label="y (east)")
    ax.plot(times, X[5, :], label="z (north)")
    ax.plot(times, [norm(X[3:6, k]) for k in range(N + 1)], label="Total")
    ax.legend()

    # u = T_c/m
    # T_c = um
    # T_c = u exp(z)
    # |T_c| = |u| exp(z)
    plt.figure()
    ax = plt.gca()
    ax.set_title("% Throttle vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("% Throttle")
    ax.plot(times[:-1], [ρ_1 / T_max * 100 for k in range(N)], label="Min thrust")
    ax.plot(
        times[:-1],
        [norm(U[:, k]) * np.exp(Z[:, k]) / T_max * 100 for k in range(N)],
        label="Thrust",
    )
    ax.plot(times[:-1], [ρ_2 / T_max * 100 for k in range(N)], label="Max thrust")
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("Angle vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.plot(
        times,
        [math.atan2(X[0, k], math.hypot(X[1, k], X[2, k])) for k in range(N + 1)],
        label="Glide angle",
    )
    ax.plot(
        times[:-1],
        [
            (math.acos(U[0, k] / norm(U[:, k])) if norm(U[:, k]) > 1e-2 else 0)
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
    ax.plot(X[0, :], X[1, :], X[2, :], label="Trajectory")

    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    plt.show()


def main():
    def solve(problem=3, error_sq=0, initial_guess=None):
        decision_variables = []

        if initial_guess is None:
            initial_guess = [0, 0, 0, 0] * N
        initial_guess_lower_bounds = []
        initial_guess_upper_bounds = []

        constraints = []
        constraint_lower_bounds = []
        constraint_upper_bounds = []

        J_min_fuel = 0

        # Start straight
        # constraints += [u_k[0] + σ_k, u_k[1], u_k[2]]
        # constraint_lower_bounds += [0, 0, 0]
        # constraint_upper_bounds += [0, 0, 0]

        x_k = ca.MX(np.concatenate((q_0, v_0)).T[0])
        z_k = ca.MX([math.log(m_0)])
        for k in range(N):
            t = k * dt

            u_k = ca.MX.sym(f"u_{k}", 3)
            σ_k = ca.MX.sym(f"σ_{k}", 1)

            # Input variables
            decision_variables += [u_k[:3], σ_k]

            # Bounds of u and σ
            initial_guess_lower_bounds += [-ca.inf, -ca.inf, -ca.inf, -1e-3]
            initial_guess_upper_bounds += [ca.inf, ca.inf, ca.inf, ca.inf]

            if k == N - 1:
                # End straight
                constraints += [u_k[0] + σ_k, u_k[1], u_k[2]]
                constraint_lower_bounds += [0, 0, 0]
                constraint_upper_bounds += [0, 0, 0]

                # End with zero thrust
                constraints += [σ_k]
                constraint_lower_bounds += [0]
                constraint_upper_bounds += [0]

            # Thrust magnitude limit
            #
            # See equation (34) of [1].
            #
            #   |u|₂ ≤ σ
            #   u_x² + u_y² + u_z² ≤ σ²
            constraints += [σ_k**2 - u_k.T @ u_k]
            constraint_lower_bounds += [0]
            constraint_upper_bounds += [ca.inf]

            # Thrust pointing limit
            #
            # See equation (34) of [1].
            #
            #   n̂ᵀu ≥ cos(θ)σ where n̂ = [1  0  0]ᵀ
            #   [1  0  0]u ≥ cos(θ)σ
            #   u_x ≥ cos(θ)σ
            constraints += [u_k[0] - ca.cos(θ) * σ_k]
            constraint_lower_bounds += [0]
            constraint_upper_bounds += [ca.inf]

            # Mass limits
            z_min = math.log(m_0 - α * ρ_2 * t)
            z_max = math.log(m_0 - α * ρ_1 * t)
            constraints += [z_k]
            constraint_lower_bounds += [z_min]
            constraint_upper_bounds += [z_max]

            # Thrust slack limits
            #
            # See equation (34) of [2].
            z_0 = math.log(m_0 - α * ρ_2 * t)
            μ_1 = ρ_1 * math.exp(-z_0)
            μ_2 = ρ_2 / (m_0 - α * ρ_2 * t)
            σ_min = μ_1 * (1 - (z_k - z_0) + 0.5 * (z_k - z_0) ** 2)
            σ_max = μ_2 * (1 - (z_k - z_0))
            constraints += [σ_k - σ_min, σ_k - σ_max]
            constraint_lower_bounds += [0, -ca.inf]
            constraint_upper_bounds += [ca.inf, 0]

            # Glide slope constraint, which ensure the trajectory isn't too
            # shallow or goes below the target height
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
            constraints += [
                x_k[0] ** 2 - math.tan(γ_gs) ** 2 * (x_k[1] ** 2 + x_k[2] ** 2)
            ]
            constraint_lower_bounds += [0]
            constraint_upper_bounds += [ca.inf]

            # Velocity limits
            constraints += [x_k[3:6].T @ x_k[3:6]]
            constraint_lower_bounds += [0]
            constraint_upper_bounds += [v_max**2]

            J_min_fuel += σ_k

            # Integrate dynamics
            #
            # See equation (2) of [1].
            #
            #   ẋ = Ax + B(g + u)
            #   ż = −ασ
            #
            #   xₖ₊₁ = A_d xₖ + B_d(g + uₖ)
            #   zₖ₊₁ = zₖ - αTσₖ
            x_k = A_d @ x_k + B_d @ (g + u_k)
            z_k -= α * dt * σ_k

        # Final x position
        constraints += [x_k[0]]
        constraint_lower_bounds += [0]
        constraint_upper_bounds += [0]

        # Final velocity
        constraints += [x_k[3], x_k[4], x_k[5]]
        constraint_lower_bounds += [0, 0, 0]
        constraint_upper_bounds += [0, 0, 0]

        if problem == 3:
            # Problem 3 from [1]: Minimum landing error
            J = x_k[1] ** 2 + x_k[2] ** 2
        elif problem == 4:
            constraints += [(x_k[1] - q_f[1, 0]) ** 2 + (x_k[2] - q_f[2, 0]) ** 2]
            constraint_lower_bounds += [0]
            constraint_upper_bounds += [error_sq]

            # Problem 4 from [1]: Minimum fuel
            J = J_min_fuel

        solver = ca.nlpsol(
            "solver",
            "ipopt",
            {
                "f": J,
                "x": ca.vertcat(*decision_variables),
                "g": ca.vertcat(*constraints),
            },
            {"error_on_fail": False, "ipopt": {"sb": "yes"}},
        )
        sol = solver(
            x0=initial_guess,
            lbx=initial_guess_lower_bounds,
            ubx=initial_guess_upper_bounds,
            lbg=constraint_lower_bounds,
            ubg=constraint_upper_bounds,
        )

        w_opt = sol["x"]
        x_opt = [np.array(np.concatenate((q_0, v_0)))]
        z_opt = [np.array([[math.log(m_0)]])]
        u_opt = []

        for k in range(N):
            u_opt += [w_opt[4 * k : 4 * k + 3]]
            x_k = A_d @ x_opt[-1] + B_d @ (g + u_opt[-1])
            z_k = z_opt[-1] - α * dt * w_opt[4 * k + 3]
            x_opt += [x_k.full()]
            z_opt += [z_k.full()]

        u_opt = np.array(u_opt, dtype=float)
        x_opt = np.array([a.squeeze() for a in x_opt], dtype=float)
        z_opt = np.array([a[0] for a in z_opt], dtype=float)
        u_opt = np.array([a.squeeze() for a in u_opt], dtype=float)
        error_sq = (x_opt[-1][1] - q_f[1, 0]) ** 2

        return x_opt.T, z_opt.T, u_opt.T, error_sq, w_opt

    x_opt, z_opt, u_opt, best_error, initial_guess_out = solve(problem=3)
    if 0:
        x_opt, z_opt, u_opt, best_error, _ = solve(
            problem=4, error_sq=best_error, initial_guess=initial_guess_out
        )

    plot_solution(x_opt, z_opt, u_opt)


if __name__ == "__main__":
    main()
