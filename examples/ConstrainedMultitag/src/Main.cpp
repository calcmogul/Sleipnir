// Copyright (c) Sleipnir contributors

#include <chrono>
#include <print>
#include <ranges>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

/**
 * Converts std::chrono::duration to a number of milliseconds rounded to three
 * decimals.
 */
template <typename Rep, typename Period = std::ratio<1>>
constexpr double ToMilliseconds(
    const std::chrono::duration<Rep, Period>& duration) {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  return duration_cast<microseconds>(duration).count() / 1e3;
}

int main() {
  auto setupStart = std::chrono::steady_clock::now();

  sleipnir::OptimizationProblem problem;

  // camera calibration
  constexpr double fx = 600;
  constexpr double fy = 600;
  constexpr double cx = 300;
  constexpr double cy = 150;

  // robot pose
  auto robot_x = problem.DecisionVariable();
  auto robot_y = problem.DecisionVariable();
  constexpr double robot_z = 0.0;
  auto robot_θ = problem.DecisionVariable();

  // cache autodiff variables
  auto sinθ = sleipnir::sin(robot_θ);
  auto cosθ = sleipnir::cos(robot_θ);

  sleipnir::VariableMatrix field2robot{
      {cosθ, -sinθ, 0, robot_x},
      {sinθ, cosθ, 0, robot_y},
      {0, 0, 1, robot_z},
      {0, 0, 0, 1},
  };

  // robot is ENU, cameras are SDE
  constexpr Eigen::Matrix4d robot2camera{
      {0, 0, 1, 0},
      {-1, 0, 0, 0},
      {0, -1, 0, 0},
      {0, 0, 0, 1},
  };

  auto field2camera = field2robot * robot2camera;

  // Cost
  sleipnir::Variable J = 0.0;

  // list of points in field space to reproject. Each one is a 4x1 vector of
  // (x,y,z,1)
  std::vector<sleipnir::VariableMatrix> field2points;
  field2points.push_back(
      sleipnir::VariableMatrix{{2, 0 - 0.08255, 0.4, 1}}.T());
  field2points.push_back(
      sleipnir::VariableMatrix{{2, 0 + 0.08255, 0.4, 1}}.T());

  // List of points we saw the target at. These are exactly what we expect for a
  // camera located at 0,0,0 (hand-calculated)
  std::vector point_observations{std::pair{325, 30}, std::pair{275, 30}};

  // initial guess at robot pose. We expect the robot to converge to 0,0,0
  robot_x.SetValue(-0.1);
  robot_y.SetValue(0.0);
  robot_θ.SetValue(0.2);

  // field2camera * field2camera⁻¹ = I
  auto camera2field =
      sleipnir::Solve(field2camera, Eigen::Matrix4d::Identity());

  for (const auto& [field2point, observation] :
       std::views::zip(field2points, point_observations)) {
    // camera2point = field2camera⁻¹ * field2point
    // field2camera * camera2point = field2point
    auto camera2point = camera2field * field2point;

    // point's coordinates in camera frame
    auto& x = camera2point(0);
    auto& y = camera2point(1);
    auto& z = camera2point(2);

    std::println("Camera2point = {}, {}, {}", x.Value(), y.Value(), z.Value());

    // coordinates observed at
    auto [u_observed, v_observed] = observation;

    auto X = x / z;
    auto Y = y / z;

    auto u = fx * X + cx;
    auto v = fy * Y + cy;

    std::println("Expected u {}, saw {}", u.Value(), u_observed);
    std::println("Expected v {}, saw {}", v.Value(), v_observed);

    auto u_err = u - u_observed;
    auto v_err = v - v_observed;

    // Cost function is square of reprojection error
    J += u_err * u_err + v_err * v_err;
  }

  problem.Minimize(J);

  auto setupEnd = std::chrono::steady_clock::now();

  auto solveStart = std::chrono::steady_clock::now();
  problem.Solve({.diagnostics = true});
  auto solveEnd = std::chrono::steady_clock::now();

  std::println("setup time = {} ms", ToMilliseconds(setupEnd - setupStart));
  std::println("solve time = {} ms", ToMilliseconds(solveEnd - solveStart));

  std::println("X={} m, Y={} m, theta={} rad", robot_x.Value(), robot_y.Value(),
               robot_θ.Value());
}
