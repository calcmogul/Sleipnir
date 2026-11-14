// Copyright (c) Sleipnir contributors

#include <chrono>  // IWYU pragma: keep
#include <format>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

// These tests ensure coverage of the off-nominal exit statuses

TEMPLATE_TEST_CASE("ExitStatus - Callback requested stop", "[ExitStatus]",
                   SCALAR_TYPES) {
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

TEMPLATE_TEST_CASE("ExitStatus - Too few DOFs", "[ExitStatus]", SCALAR_TYPES) {
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
                   SCALAR_TYPES) {
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

TEMPLATE_TEST_CASE("ExitStatus - Small constraint scaling is feasible",
                   "[ExitStatus]", SCALAR_TYPES) {
  using T = TestType;

  // The solver bounds the unscaled constraint violation by the tolerance, so a
  // constraint scaled by 1e-5 bounds x's error by the tolerance / 1e-5
  constexpr T scale(1e-5);
  const T x_tolerance = solution_tolerance<T>(1) / scale;

  // Equality constraints
  {
    slp::Problem<T> problem;

    auto x = problem.decision_variable();
    x.set_value(T(5005));

    problem.subject_to(scale * (x - T(5)) == T(0));

    CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
    CHECK_THAT(x.value(), WithinAbs(T(5), x_tolerance));
  }

  // Inequality constraints
  {
    slp::Problem<T> problem;

    auto x = problem.decision_variable();
    x.set_value(T(5005));

    problem.subject_to(scale * (T(5) - x) >= T(0));

    CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
    CHECK(x.value() <= T(5) + x_tolerance);
  }
}

TEMPLATE_TEST_CASE("ExitStatus - Nonfinite initial guess", "[ExitStatus]",
                   SCALAR_TYPES) {
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
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  x.set_value(T(1));

  problem.minimize(-x * x);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::DIVERGING_ITERATES);
}

TEMPLATE_TEST_CASE("ExitStatus - Large iterates aren't diverging",
                   "[ExitStatus]", SCALAR_TYPES) {
  using T = TestType;

  // Variables and slacks in the 1e5–1e6 range are common in unscaled problems
  // with physical units, so they shouldn't trigger the divergence check
  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  x.set_value(T(0));

  problem.minimize((x - T(1e6)) * (x - T(1e6)));
  problem.subject_to(x >= T(0));

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
}

TEMPLATE_TEST_CASE("ExitStatus - Max iterations exceeded", "[ExitStatus]",
                   SCALAR_TYPES) {
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

TEMPLATE_TEST_CASE("ExitStatus - Timeout", "[ExitStatus]", SCALAR_TYPES) {
  using T = TestType;

  using namespace std::chrono_literals;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  x.set_value(T(1));

  problem.minimize(x * x);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  // NOLINTNEXTLINE(misc-include-cleaner)
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
  CHECK(std::format("{}", slp::ExitStatus::FACTORIZATION_FAILED) ==
        "factorization failed");
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
