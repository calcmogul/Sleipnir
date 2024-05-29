// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

namespace {
template <typename T>
bool near(T expected, T actual, T tolerance) {
  using std::abs;
  return abs(expected - actual) < tolerance;
}
}  // namespace

TEMPLATE_TEST_CASE("Problem - Flywheel", "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<T> TOTAL_TIME{T(5)};
  constexpr std::chrono::duration<T> dt{T(0.005)};
  constexpr int N = static_cast<int>(TOTAL_TIME / dt);

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  using std::exp;
  T A = exp(-dt.count());
  T B = T(1) - exp(-dt.count());

  slp::Problem<T> problem;
  auto X = problem.decision_variable(1, N + 1);
  auto U = problem.decision_variable(1, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.subject_to(X.col(k + 1) == A * X.col(k) + B * U.col(k));
  }

  // State and input constraints
  problem.subject_to(X.col(0) == T(0));
  problem.subject_to(slp::bounds(T(-12), U, T(12)));

  // Cost function - minimize error
  constexpr Eigen::Matrix<T, 1, 1> r{{T(10)}};
  slp::Variable J = T(0);
  for (int k = 0; k < N + 1; ++k) {
    J += (r - X.col(k)).T() * (r - X.col(k));
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  REQUIRE(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Voltage for steady-state velocity:
  //
  // rₖ₊₁ = Arₖ + Buₖ
  // uₖ = B⁺(rₖ₊₁ − Arₖ)
  // uₖ = B⁺(rₖ − Arₖ)
  // uₖ = B⁺(I − A)rₖ
  T u_ss = T(1) / B * (T(1) - A) * r[0];

  // Verify initial state
  CHECK_THAT(X.value(0, 0), WithinAbs(T(0), T(1e-8)));

  // Verify solution
  T x(0);
  T u(0);
  for (int k = 0; k < N; ++k) {
    // Verify state
    CHECK_THAT(X.value(0, k), WithinAbs(x, T(1e-2)));

    // Determine expected input for this timestep
    T error = r[0] - x;
    if (error > T(1e-2)) {
      // Max control input until the reference is reached
      u = T(12);
    } else {
      // Maintain speed
      u = u_ss;
    }

    // Verify input
    if (k > 0 && k < N - 1 && near(T(12), U.value(0, k - 1), T(1e-2)) &&
        near(u_ss, U.value(0, k + 1), T(1e-2))) {
      // If control input is transitioning between 12 and u_ss, ensure it's
      // within (u_ss, 12)
      CHECK(U.value(0, k) >= u_ss);
      CHECK(U.value(0, k) <= T(12));
    } else {
      CHECK_THAT(U.value(0, k), WithinAbs(u, T(1e-4)));
    }

    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = A * x + B * u;
  }

  // Verify final state
  CHECK_THAT(X.value(0, N), WithinAbs(r[0], T(1e-7)));

  // Log states for offline viewing
  std::ofstream states{"Problem - Flywheel states.csv"};
  if (states.is_open()) {
    states << "Time (s),Velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{}\n", T(k) * dt.count(), X.value(0, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Problem - Flywheel inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", T(k) * dt.count(), U.value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", T(k) * dt.count(), T(0));
      }
    }
  }
}
