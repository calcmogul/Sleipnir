// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"
#include "util/scope_exit.hpp"

TEST_CASE("Problem - Double integrator", "[Problem]") {
  using namespace std::chrono_literals;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<double> T = 3.5s;
  constexpr std::chrono::duration<double> dt = 5ms;
  constexpr int N = T / dt;

  constexpr double r = 2.0;  // m

  slp::Problem problem;

  // 2x1 state vector with N + 1 timesteps (includes last state)
  auto X = problem.decision_variable(2, N + 1);

  // 1x1 input vector with N timesteps (input at last state doesn't matter)
  auto U = problem.decision_variable(1, N);

  // Kinematics constraint assuming constant acceleration between timesteps
  for (int k = 0; k < N; ++k) {
    constexpr double t = dt.count();
    auto p_k1 = X(0, k + 1);
    auto v_k1 = X(1, k + 1);
    auto p_k = X(0, k);
    auto v_k = X(1, k);
    auto a_k = U(0, k);

    // pₖ₊₁ = pₖ + vₖt + 1/2aₖt²
    problem.subject_to(p_k1 == p_k + v_k * t + 0.5 * a_k * t * t);

    // vₖ₊₁ = vₖ + aₖt
    problem.subject_to(v_k1 == v_k + a_k * t);
  }

  // Start and end at rest
  problem.subject_to(X.col(0) == Eigen::Matrix<double, 2, 1>{{0.0}, {0.0}});
  problem.subject_to(X.col(N) == Eigen::Matrix<double, 2, 1>{{r}, {0.0}});

  // Limit velocity
  problem.subject_to(-1 <= X.row(1));
  problem.subject_to(X.row(1) <= 1);

  // Limit acceleration
  problem.subject_to(-1 <= U);
  problem.subject_to(U <= 1);

  // Cost function - minimize position error
  slp::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += slp::pow(r - X(0, k), 2);
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  Eigen::Matrix<double, 2, 2> A{{1.0, dt.count()}, {0.0, 1.0}};
  Eigen::Matrix<double, 2, 1> B{0.5 * dt.count() * dt.count(), dt.count()};

  // Verify initial state
  CHECK(X.value(0, 0) == Catch::Approx(0.0).margin(1e-8));
  CHECK(X.value(1, 0) == Catch::Approx(0.0).margin(1e-8));

  // Verify solution
  Eigen::Matrix<double, 2, 1> x{0.0, 0.0};
  Eigen::Matrix<double, 1, 1> u{0.0};
  for (int k = 0; k < N; ++k) {
    // Verify state
    CHECK(X.value(0, k) == Catch::Approx(x(0)).margin(1e-2));
    CHECK(X.value(1, k) == Catch::Approx(x(1)).margin(1e-2));

    // Determine expected input for this timestep
    if (k * dt < 1s) {
      // Accelerate
      u(0) = 1.0;
    } else if (k * dt < 2.05s) {
      // Maintain speed
      u(0) = 0.0;
    } else if (k * dt < 3.275s) {
      // Decelerate
      u(0) = -1.0;
    } else {
      // Accelerate
      u(0) = 1.0;
    }

    // Verify input
    if (k > 0 && k < N - 1 &&
        std::abs(U.value(0, k - 1) - U.value(0, k + 1)) >= 1.0 - 1e-2) {
      // If control input is transitioning between -1, 0, or 1, ensure it's
      // within (-1, 1)
      CHECK(U.value(0, k) >= -1.0);
      CHECK(U.value(0, k) <= 1.0);
    } else {
      CHECK(U.value(0, k) == Catch::Approx(u(0)).margin(1e-4));
    }

    INFO(std::format("  k = {}", k));

    // Project state forward
    x = A * x + B * u;
  }

  // Verify final state
  CHECK(X.value(0, N) == Catch::Approx(r).margin(1e-8));
  CHECK(X.value(1, N) == Catch::Approx(0.0).margin(1e-8));

  // Log states for offline viewing
  std::ofstream states{"Problem Double Integrator states.csv"};
  if (states.is_open()) {
    states << "Time (s),Position (m),Velocity (m/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{},{}\n", k * dt.count(), X.value(0, k),
                            X.value(1, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Problem Double Integrator inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Acceleration (m/s²)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{}\n", k * dt.count(), U.value(0, k));
      } else {
        inputs << std::format("{},{}\n", k * dt.count(), 0.0);
      }
    }
  }
}
