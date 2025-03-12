// Copyright (c) Sleipnir contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"

TEST_CASE("Problem - Unconstrained 1D", "[Problem]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  x.set_value(2.0);

  problem.minimize(x * x - 6.0 * x);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK(x.value() == Catch::Approx(3.0).margin(1e-6));
}

TEST_CASE("Problem - Unconstrained 2D", "[Problem]") {
  {
    slp::Problem problem;

    auto x = problem.decision_variable();
    x.set_value(1.0);
    auto y = problem.decision_variable();
    y.set_value(2.0);

    problem.minimize(x * x + y * y);

    CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    CHECK(x.value() == Catch::Approx(0.0).margin(1e-6));
    CHECK(y.value() == Catch::Approx(0.0).margin(1e-6));
  }

  {
    slp::Problem problem;

    auto x = problem.decision_variable(2);
    x[0].set_value(1.0);
    x[1].set_value(2.0);

    problem.minimize(x.T() * x);

    CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    CHECK(x.value(0) == Catch::Approx(0.0).margin(1e-6));
    CHECK(x.value(1) == Catch::Approx(0.0).margin(1e-6));
  }
}

TEST_CASE("Problem - Equality-constrained", "[Problem]") {
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
    slp::Problem problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();

    problem.maximize(x * y);

    problem.subject_to(x + 3 * y == 36);

    CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    CHECK(x.value() == Catch::Approx(18.0).margin(1e-5));
    CHECK(y.value() == Catch::Approx(6.0).margin(1e-5));
  }

  {
    slp::Problem problem;

    auto x = problem.decision_variable(2);
    x[0].set_value(1.0);
    x[1].set_value(2.0);

    problem.minimize(x.T() * x);

    problem.subject_to(x == Eigen::Matrix<double, 2, 1>{{3.0, 3.0}});

    CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    CHECK(x.value(0) == Catch::Approx(3.0).margin(1e-5));
    CHECK(x.value(1) == Catch::Approx(3.0).margin(1e-5));
  }
}

TEST_CASE("Problem - Inequality-constrained 2D", "[Problem]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  x.set_value(5.0);
  auto y = problem.decision_variable();
  y.set_value(5.0);

  problem.minimize(x * x + y * 2 * y);
  problem.subject_to(y >= -x + 5);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK(x.value() == Catch::Approx(3.0 + 1.0 / 3.0).margin(1e-6));
  CHECK(y.value() == Catch::Approx(1.0 + 2.0 / 3.0).margin(1e-6));
}
