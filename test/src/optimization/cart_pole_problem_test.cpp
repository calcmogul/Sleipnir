// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "cart_pole_util.hpp"
#include "catch_string_converters.hpp"
#include "rk4.hpp"
#include "util/scope_exit.hpp"

TEST_CASE("Problem - Cart-pole", "[Problem]") {
  using namespace std::chrono_literals;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 50ms;
  constexpr int N = T / dt;

  constexpr double u_max = 20.0;  // N
  constexpr double d_max = 2.0;   // m

  constexpr Eigen::Vector<double, 4> x_initial{{0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 4> x_final{{1.0, std::numbers::pi, 0.0, 0.0}};

  slp::Problem problem;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.decision_variable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    X[0, k].set_value(
        std::lerp(x_initial(0), x_final(0), static_cast<double>(k) / N));
    X[1, k].set_value(
        std::lerp(x_initial(1), x_final(1), static_cast<double>(k) / N));
  }

  // u = f_x
  auto U = problem.decision_variable(1, N);

  // Initial conditions
  problem.subject_to(X.col(0) == x_initial);

  // Final conditions
  problem.subject_to(X.col(N) == x_final);

  // Cart position constraints
  problem.subject_to(X.row(0) >= 0.0);
  problem.subject_to(X.row(0) <= d_max);

  // Input constraints
  problem.subject_to(U >= -u_max);
  problem.subject_to(U <= u_max);

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.subject_to(
        X.col(k + 1) ==
        rk4<decltype(cart_pole_dynamics), slp::VariableMatrix,
            slp::VariableMatrix>(cart_pole_dynamics, X.col(k), U.col(k), dt));
  }

  // Minimize sum squared inputs
  slp::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U.col(k).T() * U.col(k);
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

  // Verify solution
  for (int k = 0; k < N; ++k) {
    // Cart position constraints
    CHECK(X[0, k] >= 0.0);
    CHECK(X[0, k] <= d_max);

    // Input constraints
    CHECK(U[0, k] >= -u_max);
    CHECK(U[0, k] <= u_max);

    // Dynamics constraints
    Eigen::VectorXd expected_x_k1 =
        rk4(cart_pole_dynamics_double, X.col(k).value(), U.col(k).value(), dt);
    Eigen::VectorXd actual_x_k1 = X.col(k + 1).value();
    for (int row = 0; row < actual_x_k1.rows(); ++row) {
      CHECK(actual_x_k1[row] == Catch::Approx(expected_x_k1[row]).margin(1e-8));
      INFO(std::format("  x({} @ k = {}", row, k));
    }
  }

  // Verify final state
  CHECK(X.value(0, N) == Catch::Approx(x_final(0)).margin(1e-8));
  CHECK(X.value(1, N) == Catch::Approx(x_final(1)).margin(1e-8));
  CHECK(X.value(2, N) == Catch::Approx(x_final(2)).margin(1e-8));
  CHECK(X.value(3, N) == Catch::Approx(x_final(3)).margin(1e-8));

  // Log states for offline viewing
  std::ofstream states{"Problem Cart-pole states.csv"};
  if (states.is_open()) {
    states << "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),"
              "Pole angular velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{},{},{},{}\n", k * dt.count(), X.value(0, k),
                            X.value(1, k), X.value(2, k), X.value(3, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Problem Cart-pole inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Cart force (N)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{}\n", k * dt.count(), U.value(0, k));
      } else {
        inputs << std::format("{},{}\n", k * dt.count(), 0.0);
      }
    }
  }
}
