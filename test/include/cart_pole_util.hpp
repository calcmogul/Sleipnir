// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/QR>
#include <sleipnir/autodiff/variable_matrix.hpp>

template <typename Scalar>
class CartPoleUtil {
 public:
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

  static constexpr Scalar m_c{5.0};  // Cart mass (kg)
  static constexpr Scalar m_p{0.5};  // Pole mass (kg)
  static constexpr Scalar l{0.5};    // Pole length (m)
  static constexpr Scalar g{9.806};  // Acceleration due to gravity (m/s²)

  static Eigen::Vector<Scalar, 4> dynamics_scalar(
      const Eigen::Vector<Scalar, 4>& x, const Eigen::Vector<Scalar, 1>& u) {
    using std::cos;
    using std::pow;

    auto q = x.segment(0, 2);
    auto qdot = x.segment(2, 2);
    auto theta = q[1];
    auto thetadot = qdot[1];

    //        [ m_c + m_p  m_p l cosθ]
    // M(q) = [m_p l cosθ    m_p l²  ]
    Eigen::Matrix<Scalar, 2, 2> M{
        {m_c + m_p, m_p * l * cos(theta)},         // NOLINT
        {m_p * l * cos(theta), m_p * pow(l, 2)}};  // NOLINT

    //           [0  −m_p lθ̇ sinθ]
    // C(q, q̇) = [0       0      ]
    Eigen::Matrix<Scalar, 2, 2> C{
        {Scalar(0), -m_p * l * thetadot * sin(theta)},  // NOLINT
        {Scalar(0), Scalar(0)}};

    //          [     0      ]
    // τ_g(q) = [-m_p gl sinθ]
    Eigen::Vector<Scalar, 2> tau_g{{Scalar(0)},
                                   {-m_p * g * l * sin(theta)}};  // NOLINT

    //     [1]
    // B = [0]
    constexpr Eigen::Matrix<Scalar, 2, 1> B{{Scalar(1)}, {Scalar(0)}};

    // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
    Eigen::Vector<Scalar, 4> qddot;
    qddot.segment(0, 2) = qdot;
    qddot.segment(2, 2) = M.householderQr().solve(tau_g - C * qdot + B * u);
    return qddot;
  }

  static slp::VariableMatrix<Scalar> dynamics_variable(
      const slp::VariableMatrix<Scalar>& x,
      const slp::VariableMatrix<Scalar>& u) {
    using std::pow;

    auto q = x.segment(0, 2);
    auto qdot = x.segment(2, 2);
    auto theta = q[1];
    auto thetadot = qdot[1];

    //        [ m_c + m_p  m_p l cosθ]
    // M(q) = [m_p l cosθ    m_p l²  ]
    slp::VariableMatrix<Scalar> M{
        {m_c + m_p, m_p * l * slp::cos(theta)},
        {m_p * l * slp::cos(theta), m_p * pow(l, 2)}};  // NOLINT

    //           [0  −m_p lθ̇ sinθ]
    // C(q, q̇) = [0       0      ]
    slp::VariableMatrix<Scalar> C{
        {Scalar(0), -m_p * l * thetadot * slp::sin(theta)},
        {Scalar(0), Scalar(0)}};

    //          [     0      ]
    // τ_g(q) = [-m_p gl sinθ]
    slp::VariableMatrix<Scalar> tau_g{{Scalar(0)},
                                      {-m_p * g * l * slp::sin(theta)}};

    //     [1]
    // B = [0]
    constexpr Eigen::Matrix<Scalar, 2, 1> B{{Scalar(1)}, {Scalar(0)}};

    // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
    slp::VariableMatrix<Scalar> qddot{4};
    qddot.segment(0, 2) = qdot;
    qddot.segment(2, 2) = slp::solve(M, tau_g - C * qdot + B * u);
    return qddot;
  }
};
