// Copyright (c) Sleipnir contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CatchStringConverters.hpp"

TEST_CASE("QuadraticProblem - Unconstrained 1D", "[QuadraticProblem]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x.SetValue(2.0);

  problem.Minimize(x * x - 6.0 * x);

  auto status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

  CHECK(x.Value() == Catch::Approx(3.0).margin(1e-6));
}

TEST_CASE("QuadraticProblem - Unconstrained 2D", "[QuadraticProblem]") {
  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable();
    x.SetValue(1.0);
    auto y = problem.DecisionVariable();
    y.SetValue(2.0);

    problem.Minimize(x * x + y * y);

    auto status = problem.Solve({.diagnostics = true});

    CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
    CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

    CHECK(x.Value() == Catch::Approx(0.0).margin(1e-6));
    CHECK(y.Value() == Catch::Approx(0.0).margin(1e-6));
  }

  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable(2);
    x[0].SetValue(1.0);
    x[1].SetValue(2.0);

    problem.Minimize(x.T() * x);

    auto status = problem.Solve({.diagnostics = true});

    CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
    CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

    CHECK(x.Value(0) == Catch::Approx(0.0).margin(1e-6));
    CHECK(x.Value(1) == Catch::Approx(0.0).margin(1e-6));
  }
}

TEST_CASE("QuadraticProblem - Equality-constrained", "[QuadraticProblem]") {
  // Maximize xy subject to x + 3y = 36.
  //
  // Maximize f(x,y) = xy
  // subject to g(x,y) = x + 3y - 36 = 0
  //
  //         value func  constraint
  //              |          |
  //              v          v
  // L(x,y,λ) = f(x,y) - λg(x,y)
  // L(x,y,λ) = xy - λ(x + 3y - 36)
  // L(x,y,λ) = xy - xλ - 3yλ + 36λ
  //
  // ∇_x,y,λ L(x,y,λ) = 0
  //
  // ∂L/∂x = y - λ
  // ∂L/∂y = x - 3λ
  // ∂L/∂λ = -x - 3y + 36
  //
  //  0x + 1y - 1λ = 0
  //  1x + 0y - 3λ = 0
  // -1x - 3y + 0λ + 36 = 0
  //
  // [ 0  1 -1][x]   [  0]
  // [ 1  0 -3][y] = [  0]
  // [-1 -3  0][λ]   [-36]
  //
  // Solve with:
  // ```python
  //   np.linalg.solve(
  //     np.array([[0,1,-1],
  //               [1,0,-3],
  //               [-1,-3,0]]),
  //     np.array([[0], [0], [-36]]))
  // ```
  //
  // [x]   [18]
  // [y] = [ 6]
  // [λ]   [ 6]
  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable();
    auto y = problem.DecisionVariable();

    problem.Maximize(x * y);

    problem.SubjectTo(x + 3 * y == 36);

    auto status = problem.Solve({.diagnostics = true});

    CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
    CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kLinear);
    CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

    CHECK(x.Value() == Catch::Approx(18.0).margin(1e-5));
    CHECK(y.Value() == Catch::Approx(6.0).margin(1e-5));
  }

  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable(2);
    x[0].SetValue(1.0);
    x[1].SetValue(2.0);

    problem.Minimize(x.T() * x);

    problem.SubjectTo(x == Eigen::Matrix<double, 2, 1>{{3.0, 3.0}});

    auto status = problem.Solve({.diagnostics = true});

    CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
    CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kLinear);
    CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

    CHECK(x.Value(0) == Catch::Approx(3.0).margin(1e-5));
    CHECK(x.Value(1) == Catch::Approx(3.0).margin(1e-5));
  }
}
