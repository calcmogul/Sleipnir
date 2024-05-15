// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <sleipnir/optimization/problem.hpp>

#include "cmdline_args.hpp"
#include "rk4.hpp"

slp::VariableMatrix cart_pole_dynamics(const slp::VariableMatrix& x,
                                       const slp::VariableMatrix& u) {
  // https://underactuated.mit.edu/acrobot.html#cart_pole
  //
  // θ is CCW+ measured from negative y-axis.
  //
  // q = [x, θ]ᵀ
  // q̇ = [ẋ, θ̇]ᵀ
  // u = f_x
  //
  // M(q)q̈ + C(q, q̇)q̇ = τ_g(q) + Bu
  // M(q)q̈ = τ_g(q) − C(q, q̇)q̇ + Bu
  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  //
  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  //
  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  //
  //          [     0      ]
  // τ_g(q) = [−m_p gl sinθ]
  //
  //     [1]
  // B = [0]
  constexpr double m_c = 5.0;  // Cart mass (kg)
  constexpr double m_p = 0.5;  // Pole mass (kg)
  constexpr double l = 0.5;    // Pole length (m)
  constexpr double g = 9.806;  // Acceleration due to gravity (m/s²)

  auto q = x.segment(0, 2);
  auto qdot = x.segment(2, 2);
  auto theta = q[1];
  auto thetadot = qdot[1];

  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  slp::VariableMatrix M{2, 2};
  M[0, 0] = m_c + m_p;
  M[0, 1] = m_p * l * cos(theta);  // NOLINT
  M[1, 0] = m_p * l * cos(theta);  // NOLINT
  M[1, 1] = m_p * std::pow(l, 2);

  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  slp::VariableMatrix C{2, 2};
  C[0, 0] = 0;
  C[0, 1] = -m_p * l * thetadot * sin(theta);  // NOLINT
  C[1, 0] = 0;
  C[1, 1] = 0;

  //          [     0      ]
  // τ_g(q) = [-m_p gl sinθ]
  slp::VariableMatrix tau_g{2, 1};
  tau_g[0] = 0;
  tau_g[1] = -m_p * g * l * sin(theta);  // NOLINT

  //     [1]
  // B = [0]
  Eigen::Matrix<double, 2, 1> B{{1}, {0}};

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  slp::VariableMatrix qddot{4, 1};
  qddot.segment(0, 2) = qdot;
  qddot.segment(2, 2) = slp::solve(M, tau_g - C * qdot + B * u);
  return qddot;
}

slp::Problem cart_pole_problem(std::chrono::duration<double> dt, int N) {
  constexpr double u_max = 20.0;  // N
  constexpr double d_max = 2.0;   // m

  constexpr Eigen::Vector<double, 4> x_initial{{0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 4> x_final{{1.0, std::numbers::pi, 0.0, 0.0}};

  slp::Problem problem;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.decision_variable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    X[0, k].set_value(
        std::lerp(x_initial(0), x_final(0), static_cast<double>(k) / N));
    X[1, k].set_value(
        std::lerp(x_initial(1), x_final(1), static_cast<double>(k) / N));
  }

  // u = f_x
  auto U = problem.decision_variable(1, N);

  // Initial conditions
  problem.subject_to(X.col(0) == x_initial);

  // Final conditions
  problem.subject_to(X.col(N) == x_final);

  // Cart position constraints
  problem.subject_to(X.row(0) >= 0.0);
  problem.subject_to(X.row(0) <= d_max);

  // Input constraints
  problem.subject_to(U >= -u_max);
  problem.subject_to(U <= u_max);

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.subject_to(
        X.col(k + 1) ==
        rk4<decltype(cart_pole_dynamics), slp::VariableMatrix,
            slp::VariableMatrix>(cart_pole_dynamics, X.col(k), U.col(k), dt));
  }

  // Minimize sum squared inputs
  slp::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U.col(k).T() * U.col(k);
  }
  problem.minimize(J);

  return problem;
}

int main(int argc, char* argv[]) {
  using namespace std::chrono_literals;

  CmdlineArgs args{argv, argc};
  bool diagnostics = args.contains("--enable-diagnostics");

  auto problem = cart_pole_problem(5s, 300);
  problem.solve({.diagnostics = diagnostics});
}
