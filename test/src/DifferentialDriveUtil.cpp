// Copyright (c) Sleipnir contributors

#include "DifferentialDriveUtil.hpp"

// x = [x, y, heading, left velocity, right velocity]ᵀ
// u = [left voltage, right voltage]ᵀ

inline constexpr double trackwidth = 0.699;    // m
inline constexpr double Kv_linear = 3.02;      // V/(m/s)
inline constexpr double Ka_linear = 0.642;     // V/(m/s²)
inline constexpr double Kv_angular = 1.382;    // V/(m/s)
inline constexpr double Ka_angular = 0.08495;  // V/(m/s²)

inline constexpr double A1 =
    -(Kv_linear / Ka_linear + Kv_angular / Ka_angular) / 2.0;
inline constexpr double A2 =
    -(Kv_linear / Ka_linear - Kv_angular / Ka_angular) / 2.0;
inline constexpr double B1 = 0.5 / Ka_linear + 0.5 / Ka_angular;
inline constexpr double B2 = 0.5 / Ka_linear - 0.5 / Ka_angular;
inline constexpr Eigen::Matrix<double, 2, 2> A{{A1, A2}, {A2, A1}};
inline constexpr Eigen::Matrix<double, 2, 2> B{{B1, B2}, {B2, B1}};

Eigen::Vector<double, 5> DifferentialDriveDynamicsDouble(
    const Eigen::Vector<double, 5>& x, const Eigen::Vector<double, 2>& u) {
  Eigen::Vector<double, 5> xdot;

  auto v = (x[3] + x[4]) / 2.0;
  xdot(0) = v * cos(x[2]);  // NOLINT
  xdot(1) = v * sin(x[2]);  // NOLINT
  xdot(2) = (x[4] - x[3]) / trackwidth;
  xdot.segment(3, 2) = A * x.segment(3, 2) + B * u;

  return xdot;
}

sleipnir::VariableMatrix DifferentialDriveDynamics(
    const sleipnir::VariableMatrix& x, const sleipnir::VariableMatrix& u) {
  sleipnir::VariableMatrix xdot{5};

  auto v = (x[3] + x[4]) / 2.0;
  xdot[0] = v * cos(x[2]);  // NOLINT
  xdot[1] = v * sin(x[2]);  // NOLINT
  xdot[2] = (x[4] - x[3]) / trackwidth;
  xdot.Segment(3, 2) = A * x.Segment(3, 2) + B * u;

  return xdot;
}
