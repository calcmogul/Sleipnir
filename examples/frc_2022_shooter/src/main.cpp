// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>
#include <print>

#include <Eigen/Core>
#include <sleipnir/optimization/problem.hpp>

// FRC 2022 shooter trajectory optimization.
//
// This program finds the initial velocity, pitch, and yaw for a game piece to
// hit the 2022 FRC game's target that minimizes time-to-target.

using Eigen::Vector3d;
using Vector6d = Eigen::Vector<double, 6>;

constexpr double field_width = 8.2296;    // 27 ft -> m
constexpr double field_length = 16.4592;  // 54 ft -> m
constexpr Vector6d target_wrt_field{
    {field_length / 2.0}, {field_width / 2.0}, {2.64}, {0.0}, {0.0}, {0.0}};
[[maybe_unused]]
constexpr double target_radius = 0.61;  // m
constexpr double g = 9.806;             // m/s²

slp::VariableMatrix f(const slp::VariableMatrix& x) {
  // x' = x'
  // y' = y'
  // z' = z'
  // x" = −a_D(v_x)
  // y" = −a_D(v_y)
  // z" = −g − a_D(v_z)
  //
  // where a_D(v) = ½ρv² C_D A / m
  // (see https://en.wikipedia.org/wiki/Drag_(physics)#The_drag_equation)
  constexpr double rho = 1.204;  // kg/m³
  constexpr double C_D = 0.5;
  constexpr double A = std::numbers::pi * 0.3;
  constexpr double m = 2.0;  // kg
  auto a_D = [](auto v) { return 0.5 * rho * v * v * C_D * A / m; };

  auto v_x = x[3, 0];
  auto v_y = x[4, 0];
  auto v_z = x[5, 0];
  return slp::VariableMatrix{{v_x},       {v_y},       {v_z},
                             {-a_D(v_x)}, {-a_D(v_y)}, {-g - a_D(v_z)}};
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

  slp::Problem problem;

  // Set up duration decision variables
  constexpr int N = 10;
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
  //   √{v_x² + v_y² + v_z²) ≤ vₘₐₓ
  //   v_x² + v_y² + v_z² ≤ vₘₐₓ²
  problem.subject_to(slp::pow(v_x[0] - robot_wrt_field[3], 2) +
                         slp::pow(v_y[0] - robot_wrt_field[4], 2) +
                         slp::pow(v_z[0] - robot_wrt_field[5], 2) <=
                     max_initial_velocity * max_initial_velocity);

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
  problem.minimize(T);

  problem.solve({.diagnostics = true});

  // Initial velocity vector
  Eigen::Vector3d v0 =
      X.block(3, 0, 3, 1).value() - robot_wrt_field.segment(3, 3);

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
