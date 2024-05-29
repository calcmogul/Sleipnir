// Copyright (c) Sleipnir contributors

#include <chrono>
#include <fstream>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "differential_drive_util.hpp"
#include "lerp.hpp"
#include "rk4.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Problem - Differential drive", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<T> TOTAL_TIME{T(5)};
  constexpr std::chrono::duration<T> dt{T(0.05)};
  constexpr int N = static_cast<int>(TOTAL_TIME / dt);

  constexpr T u_max(12);  // V

  constexpr Eigen::Vector<T, 5> x_initial{{T(0), T(0), T(0), T(0), T(0)}};
  constexpr Eigen::Vector<T, 5> x_final{{T(1), T(1), T(0), T(0), T(0)}};

  slp::Problem<T> problem;

  // x = [x, y, heading, left velocity, right velocity]ᵀ
  auto X = problem.decision_variable(5, N + 1);

  // Initial guess
  for (int k = 0; k < N; ++k) {
    X[0, k].set_value(lerp(x_initial[0], x_final[0], T(k) / T(N)));
    X[1, k].set_value(lerp(x_initial[1], x_final[1], T(k) / T(N)));
  }

  // u = [left voltage, right voltage]ᵀ
  auto U = problem.decision_variable(2, N);

  // Initial conditions
  problem.subject_to(X.col(0) == x_initial);

  // Final conditions
  problem.subject_to(X.col(N) == x_final);

  // Input constraints
  problem.subject_to(slp::bounds(-u_max, U, u_max));

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.subject_to(
        X.col(k + 1) ==
        rk4<T, decltype(DifferentialDriveUtil<T>::dynamics_variable),
            slp::VariableMatrix<T>, slp::VariableMatrix<T>>(
            DifferentialDriveUtil<T>::dynamics_variable, X.col(k), U.col(k),
            dt));
  }

  // Minimize sum squared states and inputs
  slp::Variable J = T(0);
  for (int k = 0; k < N; ++k) {
    J += X.col(k).T() * X.col(k) + U.col(k).T() * U.col(k);
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  REQUIRE(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Verify initial state
  CHECK_THAT(X.value(0, 0), WithinAbs(x_initial[0], T(1e-8)));
  CHECK_THAT(X.value(1, 0), WithinAbs(x_initial[1], T(1e-8)));
  CHECK_THAT(X.value(2, 0), WithinAbs(x_initial[2], T(1e-8)));
  CHECK_THAT(X.value(3, 0), WithinAbs(x_initial[3], T(1e-8)));
  CHECK_THAT(X.value(4, 0), WithinAbs(x_initial[4], T(1e-8)));

  // Verify solution
  Eigen::Vector<T, 5> x{T(0), T(0), T(0), T(0), T(0)};
  Eigen::Vector<T, 2> u{T(0), T(0)};
  for (int k = 0; k < N; ++k) {
    u = U.col(k).value();

    // Input constraints
    CHECK(U[0, k].value() >= -u_max);
    CHECK(U[0, k].value() <= u_max);
    CHECK(U[1, k].value() >= -u_max);
    CHECK(U[1, k].value() <= u_max);

    // Verify state
    CHECK_THAT(X.value(0, k), WithinAbs(x[0], T(1e-8)));
    CHECK_THAT(X.value(1, k), WithinAbs(x[1], T(1e-8)));
    CHECK_THAT(X.value(2, k), WithinAbs(x[2], T(1e-8)));
    CHECK_THAT(X.value(3, k), WithinAbs(x[3], T(1e-8)));
    CHECK_THAT(X.value(4, k), WithinAbs(x[4], T(1e-8)));
    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = rk4<T>(DifferentialDriveUtil<T>::dynamics_scalar, x, u, dt);
  }

  // Verify final state
  CHECK_THAT(X.value(0, N), WithinAbs(x_final[0], T(1e-8)));
  CHECK_THAT(X.value(1, N), WithinAbs(x_final[1], T(1e-8)));
  CHECK_THAT(X.value(2, N), WithinAbs(x_final[2], T(1e-8)));
  CHECK_THAT(X.value(3, N), WithinAbs(x_final[3], T(1e-8)));
  CHECK_THAT(X.value(4, N), WithinAbs(x_final[4], T(1e-8)));

  // Log states for offline viewing
  std::ofstream states{"Problem - Differential drive states.csv"};
  if (states.is_open()) {
    states << "Time (s),X position (m),Y position (m),Heading (rad),Left "
              "velocity (m/s),Right velocity (m/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{},{},{},{},{}\n", T(k) * dt.count(),
                            X.value(0, k), X.value(1, k), X.value(2, k),
                            X.value(3, k), X.value(4, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Problem - Differential drive inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Left voltage (V),Right voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{},{}\n", T(k) * dt.count(), U.value(0, k),
                              U.value(1, k));
      } else {
        inputs << fmt::format("{},{},{}\n", T(k) * dt.count(), T(0), T(0));
      }
    }
  }
}
