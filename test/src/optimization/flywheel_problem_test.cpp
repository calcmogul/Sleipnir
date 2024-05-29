// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"
#include "util/scope_exit.hpp"

namespace {
bool near(double expected, double actual, double tolerance) {
  return std::abs(expected - actual) < tolerance;
}
}  // namespace

TEST_CASE("Problem - Flywheel", "[Problem]") {
  using namespace std::chrono_literals;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 5ms;
  constexpr int N = T / dt;

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  double A = std::exp(-dt.count());
  double B = 1.0 - std::exp(-dt.count());

  slp::Problem problem;
  auto X = problem.decision_variable(1, N + 1);
  auto U = problem.decision_variable(1, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.subject_to(X.col(k + 1) == A * X.col(k) + B * U.col(k));
  }

  // State and input constraints
  problem.subject_to(X.col(0) == 0.0);
  problem.subject_to(-12 <= U);
  problem.subject_to(U <= 12);

  // Cost function - minimize error
  constexpr Eigen::Matrix<double, 1, 1> r{{10.0}};
  slp::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += (r - X.col(k)).T() * (r - X.col(k));
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Voltage for steady-state velocity:
  //
  // rₖ₊₁ = Arₖ + Buₖ
  // uₖ = B⁺(rₖ₊₁ − Arₖ)
  // uₖ = B⁺(rₖ − Arₖ)
  // uₖ = B⁺(I − A)rₖ
  double u_ss = 1.0 / B * (1.0 - A) * r(0);

  // Verify initial state
  CHECK(X.value(0, 0) == Catch::Approx(0.0).margin(1e-8));

  // Verify solution
  double x = 0.0;
  double u = 0.0;
  for (int k = 0; k < N; ++k) {
    // Verify state
    CHECK(X.value(0, k) == Catch::Approx(x).margin(1e-2));

    // Determine expected input for this timestep
    double error = r(0) - x;
    if (error > 1e-2) {
      // Max control input until the reference is reached
      u = 12.0;
    } else {
      // Maintain speed
      u = u_ss;
    }

    // Verify input
    if (k > 0 && k < N - 1 && near(12.0, U.value(0, k - 1), 1e-2) &&
        near(u_ss, U.value(0, k + 1), 1e-2)) {
      // If control input is transitioning between 12 and u_ss, ensure it's
      // within (u_ss, 12)
      CHECK(U.value(0, k) >= u_ss);
      CHECK(U.value(0, k) <= 12.0);
    } else {
      CHECK(U.value(0, k) == Catch::Approx(u).margin(1e-4));
    }

    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = A * x + B * u;
  }

  // Verify final state
  CHECK(X.value(0, N) == Catch::Approx(r(0)).margin(1e-7));

  // Log states for offline viewing
  std::ofstream states{"Problem Flywheel states.csv"};
  if (states.is_open()) {
    states << "Time (s),Velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{}\n", k * dt.count(), X.value(0, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Problem Flywheel inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", k * dt.count(), U.value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", k * dt.count(), 0.0);
      }
    }
  }
}
