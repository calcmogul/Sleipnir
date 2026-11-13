// Copyright (c) Sleipnir contributors

// Determines a robot pose from the corner pixel locations of several AprilTags.
//
// The robot pose is constrained to be on the floor (z = 0).

#include <print>
#include <ranges>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/problem.hpp>

int main() {
  slp::Problem<double> problem;

  // camera calibration
  constexpr double fx = 600;
  constexpr double fy = 600;
  constexpr double cx = 300;
  constexpr double cy = 150;

  // robot pose. The robot is on the floor, so its pose is 2D.
  auto field2robot = problem.decision_variable(slp::se2);

  // robot is ENU, cameras are SDE
  constexpr Eigen::Matrix3d robot2camera{
      {0, 0, 1},
      {-1, 0, 0},
      {0, -1, 0},
  };

  // list of points in field space to reproject. Each one is a 3x1 vector of
  // (x,y,z)
  std::vector field2points{Eigen::Vector3d{2, 0 - 0.08255, 0.4},
                           Eigen::Vector3d{2, 0 + 0.08255, 0.4}};

  // List of points we saw the target at. These are exactly what we expect for a
  // camera located at 0,0,0 (hand-calculated)
  std::vector point_observations{std::pair{325, 30}, std::pair{275, 30}};

  // initial guess at robot pose. We expect the robot to converge to 0,0,0
  field2robot.translation().set_value(Eigen::Vector2d{-0.1, 0.0});
  field2robot.rotation().set_value(0.2);

  auto robot2field = field2robot.inverse();

  // Cost
  slp::Variable J = 0.0;
  for (const auto& [field2point, observation] :
       std::views::zip(field2points, point_observations)) {
    // point's coordinates in robot frame. The robot is on the floor, so only
    // the point's x and y change.
    auto robot2point_xy =
        slp::Translation2<double>{field2point.head<2>()}.transform_by(
            robot2field);
    slp::VariableMatrix<double> robot2point{
        {robot2point_xy.x()}, {robot2point_xy.y()}, {field2point.z()}};

    // point's coordinates in camera frame
    auto camera2point = robot2camera.transpose() * robot2point;
    auto& x = camera2point[0];
    auto& y = camera2point[1];
    auto& z = camera2point[2];

    std::println("camera2point = {}, {}, {}", x.value(), y.value(), z.value());

    // coordinates observed at
    auto [u_observed, v_observed] = observation;

    auto X = x / z;
    auto Y = y / z;

    auto u = fx * X + cx;
    auto v = fy * Y + cy;

    std::println("Expected u {}, saw {}", u.value(), u_observed);
    std::println("Expected v {}, saw {}", v.value(), v_observed);

    auto u_err = u - u_observed;
    auto v_err = v - v_observed;

    // Cost function is square of reprojection error
    J += u_err * u_err + v_err * v_err;
  }

  problem.minimize(J);

  problem.solve({.diagnostics = true});

  std::println("x = {} m", field2robot.translation().x().value());
  std::println("y = {} m", field2robot.translation().y().value());
  std::println("θ = {} rad", field2robot.rotation().value());
}
