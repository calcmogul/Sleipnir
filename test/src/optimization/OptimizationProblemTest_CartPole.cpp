// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CartPoleUtil.hpp"
#include "CatchStringConverters.hpp"
#include "RK4.hpp"
#include "util/ScopeExit.hpp"

TEST_CASE("OptimizationProblem - Cart-pole", "[OptimizationProblem]") {
  using namespace std::chrono_literals;

  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::GlobalPoolResource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 50ms;
  constexpr int N = T / dt;

  constexpr double u_max = 20.0;  // N
  constexpr double d_max = 2.0;   // m

  constexpr Eigen::Vector<double, 4> x_initial{{0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 4> x_final{{1.0, std::numbers::pi, 0.0, 0.0}};

  sleipnir::OptimizationProblem problem;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.DecisionVariable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    X[0, k].SetValue(
        std::lerp(x_initial(0), x_final(0), static_cast<double>(k) / N));
    X[1, k].SetValue(
        std::lerp(x_initial(1), x_final(1), static_cast<double>(k) / N));
  }

  // u = f_x
  auto U = problem.DecisionVariable(1, N);

  // Initial conditions
  problem.SubjectTo(X.Col(0) == x_initial);

  // Final conditions
  problem.SubjectTo(X.Col(N) == x_final);

  // Cart position constraints
  problem.SubjectTo(X.Row(0) >= 0.0);
  problem.SubjectTo(X.Row(0) <= d_max);

  // Input constraints
  problem.SubjectTo(U >= -u_max);
  problem.SubjectTo(U <= u_max);

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.SubjectTo(X.Col(k + 1) ==
                      RK4<decltype(CartPoleDynamics), sleipnir::VariableMatrix,
                          sleipnir::VariableMatrix>(CartPoleDynamics, X.Col(k),
                                                    U.Col(k), dt));
  }

  // Minimize sum squared inputs
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U.Col(k).T() * U.Col(k);
  }
  problem.Minimize(J);

  auto status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNonlinear);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kLinear);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

  // Verify initial state
  CHECK(X.Value(0, 0) == Catch::Approx(x_initial(0)).margin(1e-8));
  CHECK(X.Value(1, 0) == Catch::Approx(x_initial(1)).margin(1e-8));
  CHECK(X.Value(2, 0) == Catch::Approx(x_initial(2)).margin(1e-8));
  CHECK(X.Value(3, 0) == Catch::Approx(x_initial(3)).margin(1e-8));

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
        RK4(CartPoleDynamicsDouble, X.Col(k).Value(), U.Col(k).Value(), dt);
    Eigen::VectorXd actual_x_k1 = X.Col(k + 1).Value();
    for (int row = 0; row < actual_x_k1.rows(); ++row) {
      CHECK(actual_x_k1(row) == Catch::Approx(expected_x_k1(row)).margin(1e-8));
      INFO(std::format("  x({} @ k = {}", row, k));
    }
  }

  // Verify final state
  CHECK(X.Value(0, N) == Catch::Approx(x_final(0)).margin(1e-8));
  CHECK(X.Value(1, N) == Catch::Approx(x_final(1)).margin(1e-8));
  CHECK(X.Value(2, N) == Catch::Approx(x_final(2)).margin(1e-8));
  CHECK(X.Value(3, N) == Catch::Approx(x_final(3)).margin(1e-8));

  // Log states for offline viewing
  std::ofstream states{"OptimizationProblem Cart-pole states.csv"};
  if (states.is_open()) {
    states << "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),"
              "Pole angular velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{},{},{},{}\n", k * dt.count(), X.Value(0, k),
                            X.Value(1, k), X.Value(2, k), X.Value(3, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"OptimizationProblem Cart-pole inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Cart force (N)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{}\n", k * dt.count(), U.Value(0, k));
      } else {
        inputs << std::format("{},{}\n", k * dt.count(), 0.0);
      }
    }
  }
}
