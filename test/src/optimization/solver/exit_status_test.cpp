// Copyright (c) Sleipnir contributors

#include <format>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

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
  x.set_value(T(1));
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  problem.add_callback([](const slp::IterationInfo<T>&) { return false; });
  x.set_value(T(1));
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  problem.add_callback([](const slp::IterationInfo<T>&) { return true; });
  x.set_value(T(1));
  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::CALLBACK_REQUESTED_STOP);

  problem.clear_callbacks();
  problem.add_callback([](const slp::IterationInfo<T>&) { return false; });
  x.set_value(T(1));
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Ensure persistent callbacks aren't removed by clear_callbacks()
  problem.add_persistent_callback(
      [](const slp::IterationInfo<T>&) { return true; });
  problem.clear_callbacks();
  x.set_value(T(1));
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

TEMPLATE_TEST_CASE("ExitStatus - Nonfinite initial guess", "[ExitStatus]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  // Nonfinite cost
  {
    slp::Problem<T> problem;
    auto x = problem.decision_variable();
    problem.minimize(T(1) / x);

    CHECK(problem.solve() == slp::ExitStatus::NONFINITE_INITIAL_GUESS);
  }

  // Nonfinite gradient
  {
    slp::Problem<T> problem;
    auto x = problem.decision_variable();
    problem.minimize(sqrt(x));

    CHECK(problem.solve() == slp::ExitStatus::NONFINITE_INITIAL_GUESS);
  }

  // Nonfinite equality constraint
  {
    slp::Problem<T> problem;
    auto x = problem.decision_variable();
    problem.subject_to(T(1) / x == T(1));

    CHECK(problem.solve() == slp::ExitStatus::NONFINITE_INITIAL_GUESS);
  }

  // Nonfinite equality constraint Jacobian
  {
    slp::Problem<T> problem;
    auto x = problem.decision_variable();
    problem.subject_to(sqrt(x) == T(1));

    CHECK(problem.solve() == slp::ExitStatus::NONFINITE_INITIAL_GUESS);
  }

  // Nonfinite inequality constraint
  {
    slp::Problem<T> problem;
    auto x = problem.decision_variable();
    problem.subject_to(T(1) / x > T(1));

    CHECK(problem.solve() == slp::ExitStatus::NONFINITE_INITIAL_GUESS);
  }

  // Nonfinite inequality constraint Jacobian
  {
    slp::Problem<T> problem;
    auto x = problem.decision_variable();
    problem.subject_to(sqrt(x) > T(1));

    CHECK(problem.solve() == slp::ExitStatus::NONFINITE_INITIAL_GUESS);
  }
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
  x.set_value(T(1));

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
  x.set_value(T(1));

  problem.minimize(x * x);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.timeout = 0s, .diagnostics = true}) ==
        slp::ExitStatus::TIMEOUT);
}

TEST_CASE("ExitStatus - Formatter", "[Formatter]") {
  CHECK(std::format("{}", slp::ExitStatus::SUCCESS) == "success");
  CHECK(std::format("{}", slp::ExitStatus::CALLBACK_REQUESTED_STOP) ==
        "callback requested stop");
  CHECK(std::format("{}", slp::ExitStatus::TOO_FEW_DOFS) ==
        "too few degrees of freedom");
  CHECK(std::format("{}", slp::ExitStatus::LOCALLY_INFEASIBLE) ==
        "locally infeasible");
  CHECK(std::format("{}", slp::ExitStatus::GLOBALLY_INFEASIBLE) ==
        "globally infeasible");
  CHECK(std::format("{}", slp::ExitStatus::LINE_SEARCH_FAILED) ==
        "line search failed");
  CHECK(std::format("{}", slp::ExitStatus::FEASIBILITY_RESTORATION_FAILED) ==
        "feasibility restoration failed");
  CHECK(std::format("{}", slp::ExitStatus::NONFINITE_INITIAL_GUESS) ==
        "nonfinite initial guess");
  CHECK(std::format("{}", slp::ExitStatus::DIVERGING_ITERATES) ==
        "diverging iterates");
  CHECK(std::format("{}", slp::ExitStatus::MAX_ITERATIONS_EXCEEDED) ==
        "max iterations exceeded");
  CHECK(std::format("{}", slp::ExitStatus::TIMEOUT) == "timeout");
}
