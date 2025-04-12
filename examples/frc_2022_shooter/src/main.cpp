// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <fmt/base.h>
#include <sleipnir/optimization/problem.hpp>

// FRC 2022 shooter trajectory optimization.
//
// This program finds the initial velocity, pitch, and yaw for a game piece to
// hit the 2022 FRC game's target that minimizes either time-to-target or
// initial velocity (see minimize() calls below).
//
// This optimization problem formulation uses direct transcription of the flight
// dynamics, including air resistance.

using Eigen::Vector3d;
using Vector6d = Eigen::Vector<double, 6>;

constexpr double field_width = 8.2296;    // 27 ft -> m
constexpr double field_length = 16.4592;  // 54 ft -> m
constexpr Vector6d target_wrt_field{
    {field_length / 2.0}, {field_width / 2.0}, {2.64}, {0.0}, {0.0}, {0.0}};
[[maybe_unused]]
constexpr double target_radius = 0.61;               // m
constexpr double cone_angle = std::numbers::pi / 4;  // rad
constexpr Eigen::Vector3d g{{0.0}, {0.0}, {9.806}};  // m/s²

slp::VariableMatrix<double> cross(const slp::VariableMatrix<double>& a,
                                  const slp::VariableMatrix<double>& b) {
  return slp::VariableMatrix<double>({{a[1, 0] * b[2, 0] - a[2, 0] * b[1, 0]},
                                      {a[2, 0] * b[0, 0] - a[0, 0] * b[2, 0]},
                                      {a[0, 0] * b[1, 0] - a[1, 0] * b[0, 0]}});
}

slp::VariableMatrix<double> f(const slp::VariableMatrix<double>& x) {
  using namespace slp::slicing;

  // x' = x'
  // y' = y'
  // z' = z'
  // [x"]   [ 0]
  // [y"] = [ 0] − F_D(v)/m v̂ − F_L(v)/m (ω x v)
  // [z"]   [−g]

  // ρ is the fluid density in kg/m³
  // v is the linear velocity in m/s
  // v̂ is the velocity direction unit vector
  // ω is the angular velocity in rad/s
  // A is the cross-sectional area of a circle in m²
  // m is the object mass in kg
  constexpr double ρ = 1.204;  // kg/m³
  auto v = x(slp::Slice{3, 6}, _);
  slp::Variable v2 = v.T() * v;
  auto v_norm = sqrt(v2);
  auto v_hat = v / v_norm;
  constexpr Eigen::Vector3d ω{{0.0}, {-2.0}, {0.0}};  // rad/s
  constexpr double r = 0.15;                          // m
  constexpr double A = std::numbers::pi * r * r;      // m²
  constexpr double m = 0.283;                         // kg

  // Per https://en.wikipedia.org/wiki/Drag_(physics)#The_drag_equation:
  //   F_D(v) = ½ρ|v|²C_D A
  //   C_D is the drag coefficient (dimensionless)
  constexpr double C_D = 0.5;
  auto F_D = 0.5 * ρ * v2 * C_D * A;

  // Magnus force:
  //   F_L(v) = ½ρ|v|C_L A
  //   C_L is the lift coefficient (dimensionless)
  constexpr double C_L = 0.5;
  auto F_L = 0.5 * ρ * v_norm * C_L * A;

  return slp::block<double>(
      {{v}, {-g - F_D / m * v_hat - F_L / m * cross(v, ω)}});
}

#ifndef RUNNING_TESTS
int main() {
  // Robot initial state
  constexpr Vector6d robot_wrt_field{{field_length / 4.0},
                                     {field_width / 4.0},
                                     {0.0},
                                     {1.524},
                                     {-1.524},
                                     {0.0}};

  constexpr double max_initial_velocity = 10.0;  // m/s

  Vector6d shooter_wrt_robot{{0.0}, {0.0}, {1.2}, {0.0}, {0.0}, {0.0}};
  Vector6d shooter_wrt_field = robot_wrt_field + shooter_wrt_robot;

  slp::Problem<double> problem;

  // Set up duration decision variables
  constexpr int N = 50;
  auto T = problem.decision_variable();
  problem.subject_to(T >= 0);
  T.set_value(1);
  auto dt = T / N;

  // Ball state in field frame
  //
  //     [x position]
  //     [y position]
  //     [z position]
  // x = [x velocity]
  //     [y velocity]
  //     [z velocity]
  auto X = problem.decision_variable(6, N);

  auto p = X.block(0, 0, 3, N);
  auto p_x = X.row(0);
  auto p_y = X.row(1);
  auto p_z = X.row(2);

  auto v = X.block(3, 0, 3, N);
  auto v_x = X.row(3);
  auto v_y = X.row(4);
  auto v_z = X.row(5);

  auto v0_wrt_shooter =
      X.block(3, 0, 3, 1) - shooter_wrt_field.segment(3, 3).eval();

  // Position initial guess is linear interpolation between start and end
  // position
  for (int k = 0; k < N; ++k) {
    p_x[k].set_value(std::lerp(shooter_wrt_field(0), target_wrt_field(0),
                               static_cast<double>(k) / N));
    p_y[k].set_value(std::lerp(shooter_wrt_field(1), target_wrt_field(1),
                               static_cast<double>(k) / N));
    p_z[k].set_value(std::lerp(shooter_wrt_field(2), target_wrt_field(2),
                               static_cast<double>(k) / N));
  }

  // Velocity initial guess is max initial velocity toward target
  Vector3d uvec_shooter_to_target =
      (target_wrt_field.segment(0, 3) - shooter_wrt_field.segment(0, 3))
          .normalized();
  for (int k = 0; k < N; ++k) {
    v.col(k).set_value(robot_wrt_field.segment(3, 3) +
                       max_initial_velocity * uvec_shooter_to_target);
  }

  // Shooter initial position
  problem.subject_to(p.col(0) == shooter_wrt_field.block(0, 0, 3, 1));

  // Require initial velocity is below max
  //
  //   √(v_x² + v_y² + v_z²) ≤ vₘₐₓ
  //   v_x² + v_y² + v_z² ≤ vₘₐₓ²
  problem.subject_to(slp::pow(v_x[0] - robot_wrt_field[3], 2) +
                         slp::pow(v_y[0] - robot_wrt_field[4], 2) +
                         slp::pow(v_z[0] - robot_wrt_field[5], 2) <=
                     max_initial_velocity * max_initial_velocity);

  // Keep-out region (cylinder with conic bowl)
  //
  //            r_t
  //           |----|
  //   ___ ___  ____
  //    |   |  |   /
  //    |  h|  |  /
  // z_t|   |  |γ/    z
  //    |  ___ |/     ^
  //    |      |      |
  //   ___     |      ⋅-> x
  //
  // (x_c, y_c, z_c) is the tip of the cone, r_t is target radius, z_t is
  // target height.
  //
  //   tan(γ) = r_t/h
  //   h = r_t/tan(γ)
  //
  //   z_c = z_t - h
  //   z_c = z_t - r_t/tan(γ)
  //
  // Constrain ball to be either outside cylinder or inside cone.
  // First, get cylindrical coordinates with respect to cone origin.
  //
  //   r' = hypot(x - x_c, y - y_c)
  //   z' = z - z_c
  //
  // Cylinder:
  //   r' ≥ r_t
  //   (x - x_c)² + (y - y_c)² ≥ r_t²
  //   (x - x_c)² + (y - y_c)² - r_t² ≥ 0
  //
  // Cone:
  //   z' ≥ r'/tan(γ)
  //   z - z_c ≥ r'/tan(γ)
  //   (z - z_c) tan(γ) ≥ r'
  //   (z - z_c)² tan²(γ) ≥ (x - x_c)² + (y - y_c)²
  //   (z - z_c)² tan²(γ) - (x - x_c)² - (y - y_c)² ≥ 0
  //
  // Ball is in keep-out region if both constraints are violated.
  //
  //   max((x - x_c)² + (y - y_c)² - r_t²,
  //       (z - z_c)² tan²(γ) - (x - x_c)² - (y - y_c)²) ≥ 0
  auto& x_c = target_wrt_field[0, 0];
  auto& y_c = target_wrt_field[1, 0];
  auto z_c = target_wrt_field[2, 0] - target_radius / std::tan(cone_angle);
  for (int k = 0; k < N; ++k) {
    auto& x = p_x[k];
    auto& y = p_y[k];
    auto& z = p_z[k];

    auto x2 = slp::pow(x - x_c, 2);
    auto y2 = slp::pow(y - y_c, 2);
    auto z2 = slp::pow(z - z_c, 2);
    auto tan = std::tan(cone_angle);
    auto cylinder = x2 + y2 - target_radius * target_radius;
    auto cone = z2 * tan * tan - x2 - y2;

    problem.subject_to(slp::max(cylinder, cone) >= 0);
  }

  // Dynamics constraints - RK4 integration
  auto h = dt;
  for (int k = 0; k < N - 1; ++k) {
    auto x_k = X.col(k);
    auto x_k1 = X.col(k + 1);

    auto k1 = f(x_k);
    auto k2 = f(x_k + h / 2 * k1);
    auto k3 = f(x_k + h / 2 * k2);
    auto k4 = f(x_k + h * k3);
    problem.subject_to(x_k1 == x_k + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4));
  }

  // Require final position is in center of target circle
  problem.subject_to(p.col(N - 1) == target_wrt_field.block(0, 0, 3, 1));

  // Require the final velocity is down
  problem.subject_to(v_z[N - 1] < 0.0);

  // Minimize time-to-target
  // problem.minimize(T);

  // Minimize initial velocity
  problem.minimize(v0_wrt_shooter.T() * v0_wrt_shooter);

  problem.solve({.diagnostics = true});

  // Initial velocity vector with respect to shooter
  Eigen::Vector3d v0 = v0_wrt_shooter.value();

  double velocity = v0.norm();
  fmt::println("Velocity = {:.03} ms", velocity);

  double pitch = std::atan2(v0[2], std::hypot(v0[0], v0[1]));
  fmt::println("Pitch = {:.03}°", pitch * 180.0 / std::numbers::pi);

  double yaw = std::atan2(v0[1], v0[0]);
  fmt::println("Yaw = {:.03}°", yaw * 180.0 / std::numbers::pi);

  fmt::println("Total time = {:.03} s", T.value());
  fmt::println("dt = {:.03} ms", dt.value() * 1e3);
}
#endif
