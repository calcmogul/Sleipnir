// Copyright (c) Sleipnir contributors

#include <chrono>
#include <numbers>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

// This problem tests the case where regularization fails
TEMPLATE_TEST_CASE("Problem - Arm on elevator", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr int N = 800;

  constexpr T ELEVATOR_START_HEIGHT(1);      // m
  constexpr T ELEVATOR_END_HEIGHT(1.25);     // m
  constexpr T ELEVATOR_MAX_VELOCITY(1);      // m/s
  constexpr T ELEVATOR_MAX_ACCELERATION(2);  // m/s²

  [[maybe_unused]]
  constexpr T ARM_LENGTH(1);                                 // m
  constexpr T ARM_START_ANGLE(0);                            // rad
  constexpr T ARM_END_ANGLE(std::numbers::pi);               // rad
  constexpr T ARM_MAX_VELOCITY(2.0 * std::numbers::pi);      // rad/s
  constexpr T ARM_MAX_ACCELERATION(4.0 * std::numbers::pi);  // rad/s²

  [[maybe_unused]]
  constexpr T END_EFFECTOR_MAX_HEIGHT(1.8);  // m

  constexpr std::chrono::duration<T> TOTAL_TIME{T(4)};
  constexpr auto dt = TOTAL_TIME / T(N);

  slp::Problem<T> problem;

  auto elevator = problem.decision_variable(2, N + 1);
  auto elevator_accel = problem.decision_variable(1, N);

  auto arm = problem.decision_variable(2, N + 1);
  auto arm_accel = problem.decision_variable(1, N);

  for (int k = 0; k < N; ++k) {
    // Elevator dynamics constraints
    problem.subject_to(elevator(0, k + 1) == elevator(0, k) +
                                                 elevator(1, k) * dt.count() +
                                                 T(0.5) * elevator_accel(0, k) *
                                                     dt.count() * dt.count());
    problem.subject_to(elevator(1, k + 1) ==
                       elevator(1, k) + elevator_accel(0, k) * dt.count());

    // Arm dynamics constraints
    problem.subject_to(arm(0, k + 1) ==
                       arm(0, k) + arm(1, k) * dt.count() +
                           T(0.5) * arm_accel(0, k) * dt.count() * dt.count());
    problem.subject_to(arm(1, k + 1) ==
                       arm(1, k) + arm_accel(0, k) * dt.count());
  }

  // Elevator start and end conditions
  problem.subject_to(elevator.col(0) ==
                     Eigen::Vector<T, 2>({ELEVATOR_START_HEIGHT, T(0)}));
  problem.subject_to(elevator.col(N) ==
                     Eigen::Vector<T, 2>({ELEVATOR_END_HEIGHT, T(0)}));

  // Arm start and end conditions
  problem.subject_to(arm.col(0) ==
                     Eigen::Vector<T, 2>({ARM_START_ANGLE, T(0)}));
  problem.subject_to(arm.col(N) == Eigen::Vector<T, 2>({ARM_END_ANGLE, T(0)}));

  // Elevator velocity limits
  problem.subject_to(slp::bounds(-ELEVATOR_MAX_VELOCITY, elevator.row(1),
                                 ELEVATOR_MAX_VELOCITY));

  // Elevator acceleration limits
  problem.subject_to(slp::bounds(-ELEVATOR_MAX_ACCELERATION, elevator_accel,
                                 ELEVATOR_MAX_ACCELERATION));

  // Arm velocity limits
  problem.subject_to(
      slp::bounds(-ARM_MAX_VELOCITY, arm.row(1), ARM_MAX_VELOCITY));

  // Arm acceleration limits
  problem.subject_to(
      slp::bounds(-ARM_MAX_ACCELERATION, arm_accel, ARM_MAX_ACCELERATION));

  // Height limit
#if 0
  auto heights =
      elevator.row(0) + ARM_LENGTH * arm.row(0).cwise_transform(slp::sin<T>);
  problem.subject_to(heights <= END_EFFECTOR_MAX_HEIGHT);
#endif

  // Cost function
  slp::Variable J = T(0);
  for (int k = 0; k < N + 1; ++k) {
    J += pow(ELEVATOR_END_HEIGHT - elevator(0, k), T(2)) +
         pow(ARM_END_ANGLE - arm(0, k), T(2));
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  REQUIRE(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
}
