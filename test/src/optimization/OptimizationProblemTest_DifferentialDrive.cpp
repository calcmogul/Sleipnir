// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CatchStringConverters.hpp"
#include "DifferentialDriveUtil.hpp"
#include "RK4.hpp"
#include "util/ScopeExit.hpp"

TEST_CASE("OptimizationProblem - Differential drive", "[OptimizationProblem]") {
  using namespace std::chrono_literals;

  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::GlobalPoolResource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 50ms;
  constexpr int N = T / dt;

  constexpr double u_max = 12.0;  // V

  constexpr Eigen::Vector<double, 5> x_initial{{0.0, 0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 5> x_final{{1.0, 1.0, 0.0, 0.0, 0.0}};

  sleipnir::OptimizationProblem problem;

  // x = [x, y, heading, left velocity, right velocity]ᵀ
  auto X = problem.DecisionVariable(5, N + 1);

  // Initial guess
  for (int k = 0; k < N; ++k) {
    X[0, k].SetValue(
        std::lerp(x_initial(0), x_final(0), static_cast<double>(k) / N));
    X[1, k].SetValue(
        std::lerp(x_initial(1), x_final(1), static_cast<double>(k) / N));
  }

  // u = [left voltage, right voltage]ᵀ
  auto U = problem.DecisionVariable(2, N);

  // Initial conditions
  problem.SubjectTo(X.Col(0) == x_initial);

  // Final conditions
  problem.SubjectTo(X.Col(N) == x_final);

  // Input constraints
  problem.SubjectTo(U >= -u_max);
  problem.SubjectTo(U <= u_max);

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.SubjectTo(X.Col(k + 1) ==
                      RK4<decltype(DifferentialDriveDynamics),
                          sleipnir::VariableMatrix, sleipnir::VariableMatrix>(
                          DifferentialDriveDynamics, X.Col(k), U.Col(k), dt));
  }

  // Minimize sum squared states and inputs
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += X.Col(k).T() * X.Col(k) + U.Col(k).T() * U.Col(k);
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
  CHECK(X.Value(4, 0) == Catch::Approx(x_initial(4)).margin(1e-8));

  // Verify solution
  Eigen::Vector<double, 5> x{0.0, 0.0, 0.0, 0.0, 0.0};
  Eigen::Vector<double, 2> u{0.0, 0.0};
  for (int k = 0; k < N; ++k) {
    u = U.Col(k).Value();

    // Input constraints
    CHECK(U[0, k].Value() >= -u_max);
    CHECK(U[0, k].Value() <= u_max);
    CHECK(U[1, k].Value() >= -u_max);
    CHECK(U[1, k].Value() <= u_max);

    // Verify state
    CHECK(X.Value(0, k) == Catch::Approx(x(0)).margin(1e-8));
    CHECK(X.Value(1, k) == Catch::Approx(x(1)).margin(1e-8));
    CHECK(X.Value(2, k) == Catch::Approx(x(2)).margin(1e-8));
    CHECK(X.Value(3, k) == Catch::Approx(x(3)).margin(1e-8));
    CHECK(X.Value(4, k) == Catch::Approx(x(4)).margin(1e-8));
    INFO(std::format("  k = {}", k));

    // Project state forward
    x = RK4(DifferentialDriveDynamicsDouble, x, u, dt);
  }

  // Verify final state
  CHECK(X.Value(0, N) == Catch::Approx(x_final(0)).margin(1e-8));
  CHECK(X.Value(1, N) == Catch::Approx(x_final(1)).margin(1e-8));
  CHECK(X.Value(2, N) == Catch::Approx(x_final(2)).margin(1e-8));
  CHECK(X.Value(3, N) == Catch::Approx(x_final(3)).margin(1e-8));
  CHECK(X.Value(4, N) == Catch::Approx(x_final(4)).margin(1e-8));

  // Log states for offline viewing
  std::ofstream states{"Differential drive states.csv"};
  if (states.is_open()) {
    states << "Time (s),X position (m),Y position (m),Heading (rad),Left "
              "velocity (m/s),Right velocity (m/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{},{},{},{},{}\n", k * dt.count(),
                            X.Value(0, k), X.Value(1, k), X.Value(2, k),
                            X.Value(3, k), X.Value(4, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Differential drive inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Left voltage (V),Right voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{},{}\n", k * dt.count(), U.Value(0, k),
                              U.Value(1, k));
      } else {
        inputs << std::format("{},{},{}\n", k * dt.count(), 0.0, 0.0);
      }
    }
  }
}
