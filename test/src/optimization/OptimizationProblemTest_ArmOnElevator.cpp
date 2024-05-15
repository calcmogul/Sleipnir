// Copyright (c) Sleipnir contributors

#include <chrono>
#include <numbers>

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CatchStringConverters.hpp"
#include "util/ScopeExit.hpp"

// This problem tests the case where regularization fails
TEST_CASE("OptimizationProblem - Arm on elevator", "[OptimizationProblem]") {
  using namespace std::chrono_literals;

  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::GlobalPoolResource().blocks_in_use() == 0u); }};

  constexpr int N = 800;

  constexpr double kElevatorStartHeight = 1.0;      // m
  constexpr double kElevatorEndHeight = 1.25;       // m
  constexpr double kElevatorMaxVelocity = 1.0;      // m/s
  constexpr double kElevatorMaxAcceleration = 2.0;  // m/s²

  constexpr double kArmLength = 1.0;                              // m
  constexpr double kArmStartAngle = 0.0;                          // rad
  constexpr double kArmEndAngle = std::numbers::pi;               // rad
  constexpr double kArmMaxVelocity = 2.0 * std::numbers::pi;      // rad/s
  constexpr double kArmMaxAcceleration = 4.0 * std::numbers::pi;  // rad/s²

  constexpr double kEndEffectorMaxHeight = 1.8;  // m

  constexpr std::chrono::duration<double> kTotalTime = 4s;
  constexpr auto dt = kTotalTime / N;

  sleipnir::OptimizationProblem problem;

  auto elevator = problem.DecisionVariable(2, N + 1);
  auto elevatorAccel = problem.DecisionVariable(1, N);

  auto arm = problem.DecisionVariable(2, N + 1);
  auto armAccel = problem.DecisionVariable(1, N);

  for (int k = 0; k < N; ++k) {
    // Elevator dynamics constraints
    problem.SubjectTo(elevator[0, k + 1] ==
                      elevator[0, k] + elevator[1, k] * dt.count() +
                          0.5 * elevatorAccel[0, k] * dt.count() * dt.count());
    problem.SubjectTo(elevator[1, k + 1] ==
                      elevator[1, k] + elevatorAccel[0, k] * dt.count());

    // Arm dynamics constraints
    problem.SubjectTo(arm[0, k + 1] ==
                      arm[0, k] + arm[1, k] * dt.count() +
                          0.5 * armAccel[0, k] * dt.count() * dt.count());
    problem.SubjectTo(arm[1, k + 1] == arm[1, k] + armAccel[0, k] * dt.count());
  }

  // Elevator start and end conditions
  problem.SubjectTo(elevator.Col(0) ==
                    Eigen::Vector2d({kElevatorStartHeight, 0.0}));
  problem.SubjectTo(elevator.Col(N) ==
                    Eigen::Vector2d({kElevatorEndHeight, 0.0}));

  // Arm start and end conditions
  problem.SubjectTo(arm.Col(0) == Eigen::Vector2d({kArmStartAngle, 0.0}));
  problem.SubjectTo(arm.Col(N) == Eigen::Vector2d({kArmEndAngle, 0.0}));

  // Elevator velocity limits
  problem.SubjectTo(-kElevatorMaxVelocity <= elevator.Row(1));
  problem.SubjectTo(elevator.Row(1) <= kElevatorMaxVelocity);

  // Elevator acceleration limits
  problem.SubjectTo(-kElevatorMaxAcceleration <= elevatorAccel);
  problem.SubjectTo(elevatorAccel <= kElevatorMaxAcceleration);

  // Arm velocity limits
  problem.SubjectTo(-kArmMaxVelocity <= arm.Row(1));
  problem.SubjectTo(arm.Row(1) <= kArmMaxVelocity);

  // Arm acceleration limits
  problem.SubjectTo(-kArmMaxAcceleration <= armAccel);
  problem.SubjectTo(armAccel <= kArmMaxAcceleration);

  // Height limit
  problem.SubjectTo(elevator.Row(0) +
                        kArmLength * arm.Row(0).CwiseTransform(sleipnir::sin) <=
                    kEndEffectorMaxHeight);

  // Cost function
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += sleipnir::pow(kElevatorEndHeight - elevator[0, k], 2) +
         sleipnir::pow(kArmEndAngle - arm[0, k], 2);
  }
  problem.Minimize(J);

  auto status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kLinear);
  CHECK(status.inequalityConstraintType ==
        sleipnir::ExpressionType::kNonlinear);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);
}
