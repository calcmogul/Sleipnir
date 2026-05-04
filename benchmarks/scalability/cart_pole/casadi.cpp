// Copyright (c) Sleipnir contributors

#include "casadi.hpp"

#include <cmath>
#include <numbers>

#include <Eigen/Core>

#include "rk4.hpp"

casadi::MX cart_pole_dynamics(const casadi::MX& x, const casadi::MX& u) {
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

  auto q = x(casadi::Slice{0, 2});
  auto qdot = x(casadi::Slice{2, 4});
  auto theta = q(1);
  auto thetadot = qdot(1);

  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  casadi::MX M{2, 2};
  M(0, 0) = m_c + m_p;
  M(0, 1) = m_p * l * cos(theta);
  M(1, 0) = m_p * l * cos(theta);
  M(1, 1) = m_p * std::pow(l, 2);

  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  casadi::MX C{2, 2};
  C(0, 0) = 0;
  C(0, 1) = -m_p * l * thetadot * sin(theta);
  C(1, 0) = 0;
  C(1, 1) = 0;

  //          [     0      ]
  // τ_g(q) = [-m_p gl sinθ]
  casadi::MX tau_g{2, 1};
  tau_g(0) = 0;
  tau_g(1) = -m_p * g * l * sin(theta);

  //     [1]
  // B = [0]
  casadi::MX B{2, 1};
  B(0) = 1.0;
  B(1) = 0.0;

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  casadi::MX qddot{4, 1};
  qddot(casadi::Slice{0, 2}) = qdot;
  qddot(casadi::Slice{2, 4}) =
      solve(M, tau_g - mtimes(C, qdot) + mtimes(B, u), "symbolicqr");
  return qddot;
}

casadi::Opti cart_pole_casadi(std::chrono::duration<double> dt, int N) {
  constexpr double u_max = 20.0;  // N
  constexpr double d_max = 2.0;   // m

  constexpr Eigen::Vector<double, 4> x_initial{{0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 4> x_final{{1.0, std::numbers::pi, 0.0, 0.0}};

  casadi::Opti problem;
  casadi::Slice all;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.variable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    problem.set_initial(X(0, k), std::lerp(x_initial(0), x_final(0),
                                           static_cast<double>(k) / N));
    problem.set_initial(X(1, k), std::lerp(x_initial(1), x_final(1),
                                           static_cast<double>(k) / N));
  }

  // u = f_x
  auto U = problem.variable(1, N);

  // Initial conditions
  problem.subject_to(X(0, 0) == x_initial(0));
  problem.subject_to(X(1, 0) == x_initial(1));
  problem.subject_to(X(2, 0) == x_initial(2));
  problem.subject_to(X(3, 0) == x_initial(3));

  // Final conditions
  problem.subject_to(X(0, N) == x_final(0));
  problem.subject_to(X(1, N) == x_final(1));
  problem.subject_to(X(2, N) == x_final(2));
  problem.subject_to(X(3, N) == x_final(3));

  // Cart position constraints
  problem.subject_to(problem.bounded(0.0, X(0, all), d_max));

  // Input constraints
  problem.subject_to(problem.bounded(-u_max, U, u_max));

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.subject_to(
        X(all, k + 1) ==
        rk4<decltype(cart_pole_dynamics), casadi::MX, casadi::MX>(
            cart_pole_dynamics, X(all, k), U(all, k), dt));
  }

  // Minimize sum squared inputs
  casadi::MX J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U(all, k).T() * U(all, k);
  }
  problem.minimize(J);

  return problem;
}
