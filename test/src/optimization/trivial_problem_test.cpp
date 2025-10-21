// Copyright (c) Sleipnir contributors

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Problem - Empty", "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
}

TEMPLATE_TEST_CASE("Problem - No cost, unconstrained", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  {
    slp::Problem<T> problem;

    auto X = problem.decision_variable(2, 3);

    CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    for (int row = 0; row < X.rows(); ++row) {
      for (int col = 0; col < X.cols(); ++col) {
        CHECK(X.value(row, col) == T(0.0));
      }
    }
  }

  {
    slp::Problem<T> problem;

    auto X = problem.decision_variable(2, 3);
    X.set_value(Eigen::Matrix<T, 2, 3>::Ones());

    CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    for (int row = 0; row < X.rows(); ++row) {
      for (int col = 0; col < X.cols(); ++col) {
        CHECK(X.value(row, col) == T(1.0));
      }
    }
  }
}
