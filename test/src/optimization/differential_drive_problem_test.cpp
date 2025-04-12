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
#include "differential_drive_util.hpp"
#include "rk4.hpp"
#include "util/scope_exit.hpp"

TEST_CASE("Problem - Differential drive", "[Problem]") {
  using namespace std::chrono_literals;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 50ms;
  constexpr int N = T / dt;

  constexpr double u_max = 12.0;  // V

  constexpr Eigen::Vector<double, 5> x_initial{{0.0, 0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 5> x_final{{1.0, 1.0, 0.0, 0.0, 0.0}};

  slp::Problem problem;

  // x = [x, y, heading, left velocity, right velocity]ᵀ
  auto X = problem.decision_variable(5, N + 1);

  // Initial guess
  for (int k = 0; k < N; ++k) {
    X(0, k).set_value(
        std::lerp(x_initial(0), x_final(0), static_cast<double>(k) / N));
    X(1, k).set_value(
        std::lerp(x_initial(1), x_final(1), static_cast<double>(k) / N));
  }

  // u = [left voltage, right voltage]ᵀ
  auto U = problem.decision_variable(2, N);

  // Initial conditions
  problem.subject_to(X.col(0) == x_initial);

  // Final conditions
  problem.subject_to(X.col(N) == x_final);

  // Input constraints
  problem.subject_to(U >= -u_max);
  problem.subject_to(U <= u_max);

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.subject_to(
        X.col(k + 1) ==
        rk4<decltype(differential_drive_dynamics), slp::VariableMatrix,
            slp::VariableMatrix>(differential_drive_dynamics, X.col(k),
                                 U.col(k), dt));
  }

  // Minimize sum squared states and inputs
  slp::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += X.col(k).T() * X.col(k) + U.col(k).T() * U.col(k);
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Verify initial state
  CHECK(X.value(0, 0) == Catch::Approx(x_initial(0)).margin(1e-8));
  CHECK(X.value(1, 0) == Catch::Approx(x_initial(1)).margin(1e-8));
  CHECK(X.value(2, 0) == Catch::Approx(x_initial(2)).margin(1e-8));
  CHECK(X.value(3, 0) == Catch::Approx(x_initial(3)).margin(1e-8));
  CHECK(X.value(4, 0) == Catch::Approx(x_initial(4)).margin(1e-8));

  // Verify solution
  Eigen::Vector<double, 5> x{0.0, 0.0, 0.0, 0.0, 0.0};
  Eigen::Vector<double, 2> u{0.0, 0.0};
  for (int k = 0; k < N; ++k) {
    u = U.col(k).value();

    // Input constraints
    CHECK(U(0, k).value() >= -u_max);
    CHECK(U(0, k).value() <= u_max);
    CHECK(U(1, k).value() >= -u_max);
    CHECK(U(1, k).value() <= u_max);

    // Verify state
    CHECK(X.value(0, k) == Catch::Approx(x[0]).margin(1e-8));
    CHECK(X.value(1, k) == Catch::Approx(x[1]).margin(1e-8));
    CHECK(X.value(2, k) == Catch::Approx(x[2]).margin(1e-8));
    CHECK(X.value(3, k) == Catch::Approx(x[3]).margin(1e-8));
    CHECK(X.value(4, k) == Catch::Approx(x[4]).margin(1e-8));
    INFO(std::format("  k = {}", k));

    // Project state forward
    x = rk4(differential_drive_dynamics_double, x, u, dt);
  }

  // Verify final state
  CHECK(X.value(0, N) == Catch::Approx(x_final[0]).margin(1e-8));
  CHECK(X.value(1, N) == Catch::Approx(x_final[1]).margin(1e-8));
  CHECK(X.value(2, N) == Catch::Approx(x_final[2]).margin(1e-8));
  CHECK(X.value(3, N) == Catch::Approx(x_final[3]).margin(1e-8));
  CHECK(X.value(4, N) == Catch::Approx(x_final[4]).margin(1e-8));

  // Log states for offline viewing
  std::ofstream states{"Differential drive states.csv"};
  if (states.is_open()) {
    states << "Time (s),X position (m),Y position (m),Heading (rad),Left "
              "velocity (m/s),Right velocity (m/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{},{},{},{},{}\n", k * dt.count(),
                            X.value(0, k), X.value(1, k), X.value(2, k),
                            X.value(3, k), X.value(4, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Differential drive inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Left voltage (V),Right voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{},{}\n", k * dt.count(), U.value(0, k),
                              U.value(1, k));
      } else {
        inputs << std::format("{},{},{}\n", k * dt.count(), 0.0, 0.0);
      }
    }
  }
}
