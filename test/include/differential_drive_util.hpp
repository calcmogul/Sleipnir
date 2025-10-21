// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>

#include <Eigen/Core>
#include <sleipnir/autodiff/variable_matrix.hpp>

template <typename Scalar>
class DifferentialDriveUtil {
 public:
  // x = [x, y, heading, left velocity, right velocity]ᵀ
  // u = [left voltage, right voltage]ᵀ

  static constexpr Scalar trackwidth{0.699};    // m
  static constexpr Scalar Kv_linear{3.02};      // V/(m/s)
  static constexpr Scalar Ka_linear{0.642};     // V/(m/s²)
  static constexpr Scalar Kv_angular{1.382};    // V/(m/s)
  static constexpr Scalar Ka_angular{0.08495};  // V/(m/s²)

  static constexpr Scalar A1{
      -(Kv_linear / Ka_linear + Kv_angular / Ka_angular) / Scalar(2.0)};
  static constexpr Scalar A2{
      -(Kv_linear / Ka_linear - Kv_angular / Ka_angular) / Scalar(2.0)};
  static constexpr Scalar B1 =
      Scalar(0.5) / Ka_linear + Scalar(0.5) / Ka_angular;
  static constexpr Scalar B2 =
      Scalar(0.5) / Ka_linear - Scalar(0.5) / Ka_angular;
  static constexpr Eigen::Matrix<Scalar, 2, 2> A{{A1, A2}, {A2, A1}};
  static constexpr Eigen::Matrix<Scalar, 2, 2> B{{B1, B2}, {B2, B1}};

  static Eigen::Vector<Scalar, 5> dynamics_scalar(
      const Eigen::Vector<Scalar, 5>& x, const Eigen::Vector<Scalar, 2>& u) {
    using std::cos;
    using std::sin;

    Eigen::Vector<Scalar, 5> xdot;

    auto v = (x[3] + x[4]) / Scalar(2.0);
    xdot(0) = v * cos(x[2]);  // NOLINT
    xdot(1) = v * sin(x[2]);  // NOLINT
    xdot(2) = (x[4] - x[3]) / trackwidth;
    xdot.segment(3, 2) = A * x.segment(3, 2) + B * u;

    return xdot;
  }

  static slp::VariableMatrix<Scalar> dynamics_variable(
      const slp::VariableMatrix<Scalar>& x,
      const slp::VariableMatrix<Scalar>& u) {
    slp::VariableMatrix<Scalar> xdot{5};

    auto v = (x[3] + x[4]) / Scalar(2.0);
    xdot[0] = v * slp::cos(x[2]);
    xdot[1] = v * slp::sin(x[2]);
    xdot[2] = (x[4] - x[3]) / trackwidth;
    xdot.segment(3, 2) = A * x.segment(3, 2) + B * u;

    return xdot;
  }
};
