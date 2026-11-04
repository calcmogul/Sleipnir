// Copyright (c) Sleipnir contributors

#include <chrono>
#include <format>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/util/pool.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "cart_pole_util.hpp"
#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "lerp.hpp"
#include "rk4.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Problem - Cart-pole", "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<T> TOTAL_TIME{T(5)};

  // dt = 0.25 s is sensitive to small solver changes, so a failure here may
  // just mean the iterates found a different basin rather than a regression.
  const std::chrono::duration<T> dt{GENERATE(T(0.05), T(0.125), T(0.25))};
  const int N = static_cast<int>(TOTAL_TIME / dt);
  INFO(std::format("dt = {} s", dt.count()));

  constexpr T u_max(20);  // N
  constexpr T d_max(2);   // m

  constexpr Eigen::Vector4<T> x_initial{{T(0), T(0), T(0), T(0)}};
  constexpr Eigen::Vector4<T> x_final{{T(1), T(std::numbers::pi), T(0), T(0)}};

  slp::Problem<T> problem;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.decision_variable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    X[0, k].set_value(lerp(x_initial[0], x_final[0], T(k) / T(N)));
    X[1, k].set_value(lerp(x_initial[1], x_final[1], T(k) / T(N)));
  }

  // u = f_x
  auto U = problem.decision_variable(1, N);

  // Initial conditions
  problem.subject_to(X.col(0) == x_initial);

  // Final conditions
  problem.subject_to(X.col(N) == x_final);

  // Cart position constraints
  problem.subject_to(slp::bounds(T(0), X.row(0), d_max));

  // Input constraints
  problem.subject_to(slp::bounds(-u_max, U, u_max));

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.subject_to(
        X.col(k + 1) ==
        rk4<T, decltype(&CartPoleUtil<T>::dynamics_variable),
            slp::VariableMatrix<T>, slp::VariableMatrix<T>>(
            &CartPoleUtil<T>::dynamics_variable, X.col(k), U.col(k), dt));
  }

  // Minimize sum squared inputs
  slp::Variable J = T(0);
  for (int k = 0; k < N; ++k) {
    J += U.col(k).T() * U.col(k);
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  REQUIRE(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Verify initial state
  CHECK_THAT(X.value(0, 0), WithinAbs(x_initial[0], solution_tolerance<T>(1)));
  CHECK_THAT(X.value(1, 0),
             WithinAbs(x_initial[1],
                       solution_tolerance<T>(1, SolutionType::DEGENERATE)));
  CHECK_THAT(X.value(2, 0),
             WithinAbs(x_initial[2],
                       solution_tolerance<T>(1, SolutionType::DEGENERATE)));
  CHECK_THAT(X.value(3, 0), WithinAbs(x_initial[3], solution_tolerance<T>(1)));

  // Constraint residuals are part of the KKT error, so the solver only
  // guarantees bounds to within its tolerance. For example, the initial state
  // pins X(0, 0) to its lower bound, so its sign is rounding error.
  const T bound_tolerance = solution_tolerance<T>(1);

  // Verify solution
  for (int k = 0; k < N; ++k) {
    // Cart position constraints
    CHECK(X.value(0, k) >= -bound_tolerance);
    CHECK(X.value(0, k) <= d_max + bound_tolerance);

    // Input constraints
    CHECK(U.value(0, k) >= -u_max - bound_tolerance);
    CHECK(U.value(0, k) <= u_max + bound_tolerance);

    // Dynamics constraints
    Eigen::VectorX<T> expected_x_k1 =
        rk4<T>(&CartPoleUtil<T>::dynamics_scalar, X.col(k).value(),
               U.col(k).value(), dt);
    Eigen::VectorX<T> actual_x_k1 = X.col(k + 1).value();
    for (int row = 0; row < actual_x_k1.rows(); ++row) {
      INFO(std::format("  x({} @ k = {}", row, k));
      CHECK_THAT(actual_x_k1[row],
                 WithinAbs(expected_x_k1[row], solution_tolerance<T>(1)));
    }
  }

  // Verify final state
  CHECK_THAT(X.value(0, N),
             WithinAbs(x_final[0],
                       solution_tolerance<T>(1, SolutionType::DEGENERATE)));
  CHECK_THAT(X.value(1, N),
             WithinAbs(x_final[1],
                       solution_tolerance<T>(1, SolutionType::DEGENERATE)));
  CHECK_THAT(X.value(2, N),
             WithinAbs(x_final[2],
                       solution_tolerance<T>(1, SolutionType::DEGENERATE)));
  CHECK_THAT(X.value(3, N),
             WithinAbs(x_final[3],
                       solution_tolerance<T>(1, SolutionType::DEGENERATE)));

  // Log states for offline viewing
  std::ofstream states{"Problem - Cart-pole states.csv"};
  if (states.is_open()) {
    states << "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),"
              "Pole angular velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{},{},{},{}\n", T(k) * dt.count(),
                            X.value(0, k), X.value(1, k), X.value(2, k),
                            X.value(3, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Problem - Cart-pole inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Cart force (N)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{}\n", T(k) * dt.count(), U.value(0, k));
      } else {
        inputs << std::format("{},{}\n", T(k) * dt.count(), T(0));
      }
    }
  }
}
