// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>
#include <print>

#include <Eigen/Core>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/optimization/problem.hpp>

// FRC 2024 shooter trajectory optimization.
//
// This program finds the initial velocity, pitch, and yaw for a game piece to
// hit the 2024 FRC game's target that minimizes either z sensitivity to initial
// velocity or initial velocity (see minimize() calls below).
//
// This optimization problem formulation uses single-shooting on the flight
// dynamics, including air resistance, to allow minimizing z sensitivity.

using Eigen::Vector3d;
using Vector6d = Eigen::Vector<double, 6>;

constexpr double field_width = 8.2296;    // 27 ft -> m
constexpr double field_length = 16.4592;  // 54 ft -> m
[[maybe_unused]]
constexpr double target_width = 1.05;       // m
constexpr double target_lower_edge = 1.98;  // m
constexpr double target_upper_edge = 2.11;  // m
constexpr double target_depth = 0.46;       // m
constexpr Vector6d target_wrt_field{
    {field_length - target_depth / 2.0},
    {field_width - 2.6575},
    {(target_upper_edge + target_lower_edge) / 2.0},
    {0.0},
    {0.0},
    {0.0}};
constexpr Eigen::Vector3d g{{0.0}, {0.0}, {9.806}};  // m/s²

slp::VariableMatrix<double> f(const slp::VariableMatrix<double>& x) {
  using namespace slp::slicing;

  // x' = x'
  // y' = y'
  // z' = z'
  // x" = −F_D(v)/m v̂_x
  // y" = −F_D(v)/m v̂_y
  // z" = −g − F_D(v)/m v̂_z
  //
  // Per https://en.wikipedia.org/wiki/Drag_(physics)#The_drag_equation:
  //   F_D(v) = ½ρv²C_D A
  //   ρ is the fluid density in kg/m³
  //   v is the velocity magnitude in m/s
  //   C_D is the drag coefficient (dimensionless)
  //   A is the cross-sectional area of a circle in m²
  //   m is the mass in kg
  //   v̂ is the velocity direction unit vector
  constexpr double ρ = 1.204;  // kg/m³
  auto v = x[slp::Slice{3, 6}, _];
  slp::Variable v2 = v.T() * v;
  constexpr double C_D = 0.5;
  constexpr double r = 0.15;                      // m
  constexpr double A = std::numbers::pi * r * r;  // m²
  auto F_D = 0.5 * ρ * v2 * C_D * A;

  constexpr double m = 0.283;  // kg
  auto v_hat = v / sqrt(v2);
  return slp::block<double>({{v}, {-g - F_D / m * v_hat}});
}

#ifndef RUNNING_TESTS
int main() {
  // Robot initial state
  constexpr Vector6d robot_wrt_field{{0.75 * field_length},
                                     {field_width / 3.0},
                                     {0.0},
                                     {1.524},
                                     {-1.524},
                                     {0.0}};

  constexpr double max_initial_velocity = 15.0;  // m/s

  Vector6d shooter_wrt_robot{{0.0}, {0.0}, {0.6096}, {0.0}, {0.0}, {0.0}};
  Vector6d shooter_wrt_field = robot_wrt_field + shooter_wrt_robot;

  slp::Problem<double> problem;

  // Set up duration decision variables
  constexpr int N = 10;
  auto T = problem.decision_variable();
  problem.subject_to(T >= 0);
  T.set_value(1);
  auto dt = T / N;

  // Disc state in field frame
  //
  //     [x position]
  //     [y position]
  //     [z position]
  // x = [x velocity]
  //     [y velocity]
  //     [z velocity]
  auto x = problem.decision_variable(6);

  // Position initial guess is start position
  x.segment(0, 3).set_value(shooter_wrt_field.segment(0, 3));

  // Velocity initial guess is max initial velocity toward target
  Vector3d uvec_shooter_to_target =
      (target_wrt_field.segment(0, 3) - shooter_wrt_field.segment(0, 3))
          .normalized();
  x.segment(3, 3).set_value(robot_wrt_field.segment(3, 3) +
                            max_initial_velocity * uvec_shooter_to_target);

  auto v0_wrt_shooter =
      x.segment(3, 3) - shooter_wrt_field.segment(3, 3).eval();

  // Shooter initial position
  problem.subject_to(x.segment(0, 3) == shooter_wrt_field.block(0, 0, 3, 1));

  // Require initial velocity is below max
  //
  //   √(v_x² + v_y² + v_z²) ≤ vₘₐₓ
  //   v_x² + v_y² + v_z² ≤ vₘₐₓ²
  problem.subject_to(slp::pow(x[3] - robot_wrt_field[3], 2) +
                         slp::pow(x[4] - robot_wrt_field[4], 2) +
                         slp::pow(x[5] - robot_wrt_field[5], 2) <=
                     max_initial_velocity * max_initial_velocity);

  // Dynamics constraints - RK4 integration
  auto h = dt;
  auto x_k = x;
  for (int k = 0; k < N - 1; ++k) {
    auto k1 = f(x_k);
    auto k2 = f(x_k + h / 2 * k1);
    auto k3 = f(x_k + h / 2 * k2);
    auto k4 = f(x_k + h * k3);
    x_k += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
  }

  // Require final position is in center of target circle
  problem.subject_to(x_k.segment(0, 3) == target_wrt_field.block(0, 0, 3, 1));

  // Require the final velocity is up
  problem.subject_to(x_k[5] > 0.0);

  // Minimize sensitivity of vertical position to velocity
  auto sensitivity = slp::Gradient(x_k[3], x.segment(3, 3)).get();
  problem.minimize((sensitivity.transpose() * sensitivity).eval().coeff(0, 0));

  // Minimize initial velocity
  // problem.minimize(v0_wrt_shooter.T() * v0_wrt_shooter);

  problem.solve({.diagnostics = true});

  // Initial velocity vector with respect to shooter
  Eigen::Vector3d v0 = v0_wrt_shooter.value();

  double velocity = v0.norm();
  std::println("Velocity = {:.03} ms", velocity);

  double pitch = std::atan2(v0[2], std::hypot(v0[0], v0[1]));
  std::println("Pitch = {:.03}°", pitch * 180.0 / std::numbers::pi);

  double yaw = std::atan2(v0[1], v0[0]);
  std::println("Yaw = {:.03}°", yaw * 180.0 / std::numbers::pi);

  std::println("Total time = {:.03} s", T.value());
  std::println("dt = {:.03} ms", dt.value() * 1e3);
}
#endif
