// Copyright (c) Sleipnir contributors

#include "CartPoleUtil.hpp"

#include <Eigen/QR>

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
// τ_g(q) = [-m_p gl sinθ]
//
//     [1]
// B = [0]

inline constexpr double m_c = 5.0;  // Cart mass (kg)
inline constexpr double m_p = 0.5;  // Pole mass (kg)
inline constexpr double l = 0.5;    // Pole length (m)
inline constexpr double g = 9.806;  // Acceleration due to gravity (m/s²)

Eigen::Vector<double, 4> CartPoleDynamicsDouble(
    const Eigen::Vector<double, 4>& x, const Eigen::Vector<double, 1>& u) {
  auto q = x.segment(0, 2);
  auto qdot = x.segment(2, 2);
  auto theta = q[1];
  auto thetadot = qdot[1];

  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  Eigen::Matrix<double, 2, 2> M{
      {m_c + m_p, m_p * l * cos(theta)},              // NOLINT
      {m_p * l * cos(theta), m_p * std::pow(l, 2)}};  // NOLINT

  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  Eigen::Matrix<double, 2, 2> C{
      {0, -m_p * l * thetadot * sin(theta)},  // NOLINT
      {0, 0}};

  //          [     0      ]
  // τ_g(q) = [-m_p gl sinθ]
  Eigen::Vector<double, 2> tau_g{{0}, {-m_p * g * l * sin(theta)}};  // NOLINT

  //     [1]
  // B = [0]
  constexpr Eigen::Matrix<double, 2, 1> B{{1}, {0}};

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  Eigen::Vector<double, 4> qddot;
  qddot.segment(0, 2) = qdot;
  qddot.segment(2, 2) = M.householderQr().solve(tau_g - C * qdot + B * u);
  return qddot;
}

sleipnir::VariableMatrix CartPoleDynamics(const sleipnir::VariableMatrix& x,
                                          const sleipnir::VariableMatrix& u) {
  auto q = x.Segment(0, 2);
  auto qdot = x.Segment(2, 2);
  auto theta = q[1];
  auto thetadot = qdot[1];

  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  sleipnir::VariableMatrix M{
      {m_c + m_p, m_p * l * cos(theta)},              // NOLINT
      {m_p * l * cos(theta), m_p * std::pow(l, 2)}};  // NOLINT

  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  sleipnir::VariableMatrix C{{0, -m_p * l * thetadot * sin(theta)},  // NOLINT
                             {0, 0}};

  //          [     0      ]
  // τ_g(q) = [-m_p gl sinθ]
  sleipnir::VariableMatrix tau_g{{0}, {-m_p * g * l * sin(theta)}};  // NOLINT

  //     [1]
  // B = [0]
  constexpr Eigen::Matrix<double, 2, 1> B{{1}, {0}};

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  sleipnir::VariableMatrix qddot{4};
  qddot.Segment(0, 2) = qdot;
  qddot.Segment(2, 2) = sleipnir::Solve(M, tau_g - C * qdot + B * u);
  return qddot;
}
