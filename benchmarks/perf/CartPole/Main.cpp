// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CmdlineArguments.hpp"
#include "RK4.hpp"

sleipnir::VariableMatrix CartPoleDynamics(const sleipnir::VariableMatrix& x,
                                          const sleipnir::VariableMatrix& u) {
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

  auto q = x.Segment(0, 2);
  auto qdot = x.Segment(2, 2);
  auto theta = q[1];
  auto thetadot = qdot[1];

  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  sleipnir::VariableMatrix M{2, 2};
  M[0, 0] = m_c + m_p;
  M[0, 1] = m_p * l * cos(theta);  // NOLINT
  M[1, 0] = m_p * l * cos(theta);  // NOLINT
  M[1, 1] = m_p * std::pow(l, 2);

  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  sleipnir::VariableMatrix C{2, 2};
  C[0, 0] = 0;
  C[0, 1] = -m_p * l * thetadot * sin(theta);  // NOLINT
  C[1, 0] = 0;
  C[1, 1] = 0;

  //          [     0      ]
  // τ_g(q) = [-m_p gl sinθ]
  sleipnir::VariableMatrix tau_g{2, 1};
  tau_g[0] = 0;
  tau_g[1] = -m_p * g * l * sin(theta);  // NOLINT

  //     [1]
  // B = [0]
  Eigen::Matrix<double, 2, 1> B{{1}, {0}};

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  sleipnir::VariableMatrix qddot{4, 1};
  qddot.Segment(0, 2) = qdot;
  qddot.Segment(2, 2) = sleipnir::Solve(M, tau_g - C * qdot + B * u);
  return qddot;
}

sleipnir::OptimizationProblem CartPoleProblem(std::chrono::duration<double> dt,
                                              int N) {
  constexpr double u_max = 20.0;  // N
  constexpr double d_max = 2.0;   // m

  constexpr Eigen::Vector<double, 4> x_initial{{0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 4> x_final{{1.0, std::numbers::pi, 0.0, 0.0}};

  sleipnir::OptimizationProblem problem;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.DecisionVariable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    X[0, k].SetValue(
        std::lerp(x_initial(0), x_final(0), static_cast<double>(k) / N));
    X[1, k].SetValue(
        std::lerp(x_initial(1), x_final(1), static_cast<double>(k) / N));
  }

  // u = f_x
  auto U = problem.DecisionVariable(1, N);

  // Initial conditions
  problem.SubjectTo(X.Col(0) == x_initial);

  // Final conditions
  problem.SubjectTo(X.Col(N) == x_final);

  // Cart position constraints
  problem.SubjectTo(X.Row(0) >= 0.0);
  problem.SubjectTo(X.Row(0) <= d_max);

  // Input constraints
  problem.SubjectTo(U >= -u_max);
  problem.SubjectTo(U <= u_max);

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.SubjectTo(X.Col(k + 1) ==
                      RK4<decltype(CartPoleDynamics), sleipnir::VariableMatrix,
                          sleipnir::VariableMatrix>(CartPoleDynamics, X.Col(k),
                                                    U.Col(k), dt));
  }

  // Minimize sum squared inputs
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U.Col(k).T() * U.Col(k);
  }
  problem.Minimize(J);

  return problem;
}

int main(int argc, char* argv[]) {
  using namespace std::chrono_literals;

  CmdlineArgs args{argv, argc};
  bool diagnostics = args.Contains("--enable-diagnostics");

  auto problem = CartPoleProblem(5s, 300);
  problem.Solve({.diagnostics = diagnostics});
}
