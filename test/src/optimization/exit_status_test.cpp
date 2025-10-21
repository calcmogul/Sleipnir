// Copyright (c) Sleipnir contributors

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

// These tests ensure coverage of the off-nominal exit statuses

TEMPLATE_TEST_CASE("ExitStatus - Callback requested stop", "[ExitStatus]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  problem.add_callback([](const slp::IterationInfo<T>&) {});
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  problem.add_callback([](const slp::IterationInfo<T>&) { return false; });
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  problem.add_callback([](const slp::IterationInfo<T>&) { return true; });
  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::CALLBACK_REQUESTED_STOP);

  problem.clear_callbacks();
  problem.add_callback([](const slp::IterationInfo<T>&) { return false; });
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Ensure persistent callbacks aren't removed by clear_callbacks()
  problem.add_persistent_callback(
      [](const slp::IterationInfo<T>&) { return true; });
  problem.clear_callbacks();
  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::CALLBACK_REQUESTED_STOP);
}

TEMPLATE_TEST_CASE("ExitStatus - Too few DOFs", "[ExitStatus]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();
  auto z = problem.decision_variable();

  problem.subject_to(x == T(1));
  problem.subject_to(x == T(2));
  problem.subject_to(y == T(1));
  problem.subject_to(z == T(1));

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::TOO_FEW_DOFS);
}

TEMPLATE_TEST_CASE("ExitStatus - Locally infeasible", "[ExitStatus]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  // Equality constraints
  {
    slp::Problem<T> problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();
    auto z = problem.decision_variable();

    problem.subject_to(x == y + T(1));
    problem.subject_to(y == z + T(1));
    problem.subject_to(z == x + T(1));

    CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) ==
          slp::ExitStatus::LOCALLY_INFEASIBLE);
  }

  // Inequality constraints
  {
    slp::Problem<T> problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();
    auto z = problem.decision_variable();

    problem.subject_to(x >= y + T(1));
    problem.subject_to(y >= z + T(1));
    problem.subject_to(z >= x + T(1));

    CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

    CHECK(problem.solve({.diagnostics = true}) ==
          slp::ExitStatus::LOCALLY_INFEASIBLE);
  }
}

TEMPLATE_TEST_CASE("ExitStatus - Nonfinite initial cost or constraints",
                   "[ExitStatus]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  x.set_value(T(-1.0));
  problem.minimize(slp::sqrt(x));

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::NONFINITE_INITIAL_COST_OR_CONSTRAINTS);
}

TEMPLATE_TEST_CASE("ExitStatus - Diverging iterates", "[ExitStatus]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  problem.minimize(x);

  CHECK(problem.cost_function_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::DIVERGING_ITERATES);
}

TEMPLATE_TEST_CASE("ExitStatus - Max iterations exceeded", "[ExitStatus]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.max_iterations = 0, .diagnostics = true}) ==
        slp::ExitStatus::MAX_ITERATIONS_EXCEEDED);
}

TEMPLATE_TEST_CASE("ExitStatus - Timeout", "[ExitStatus]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  using namespace std::chrono_literals;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.timeout = 0s, .diagnostics = true}) ==
        slp::ExitStatus::TIMEOUT);
}
