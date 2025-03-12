// Copyright (c) Sleipnir contributors

#include <format>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "range.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Problem - Quartic", "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  x.set_value(T(20));

  problem.minimize(pow(x, T(4)));

  problem.subject_to(x >= T(1));

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(x.value(), WithinAbs(T(1), T(1e-6)));
}

TEMPLATE_TEST_CASE("Problem - Rosenbrock with cubic and line constraint",
                   "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization

  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();

  problem.minimize(100 * pow(y - pow(x, 2), 2) + pow(1 - x, 2));

  problem.subject_to(y >= pow(x - 1, 3) + 1);
  problem.subject_to(y <= -x + 2);

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONLINEAR);

  for (auto x0 : range(T(-1.5), T(1.5), T(0.1))) {
    for (auto y0 : range(T(-0.5), T(2.5), T(0.1))) {
      x.set_value(x0);
      y.set_value(y0);

      CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

      auto near = [](T expected, T actual, T tolerance) {
        using std::abs;
        return abs(expected - actual) < tolerance;
      };

      // Local minimum at (0, 0)
      // Global minimum at (1, 1)
      CHECK((near(T(0), x.value(), T(1e-2)) || near(T(1), x.value(), T(1e-2))));
      INFO(std::format("  (x₀, y₀) = ({}, {})", x0, y0));
      INFO(std::format("  (x, y) = ({}, {})", x.value(), y.value()));
      CHECK((near(T(0), y.value(), T(1e-2)) || near(T(1), y.value(), T(1e-2))));
      INFO(std::format("  (x₀, y₀) = ({}, {})", x0, y0));
      INFO(std::format("  (x, y) = ({}, {})", x.value(), y.value()));
    }
  }
}

TEMPLATE_TEST_CASE("Problem - Rosenbrock with disk constraint", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization

  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();

  problem.minimize(pow(T(1) - x, T(2)) + T(100) * pow(y - pow(x, T(2)), T(2)));

  problem.subject_to(pow(x, T(2)) + pow(y, T(2)) <= T(2));

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::QUADRATIC);

  for (auto x0 : range(T(-1.5), T(1.5), T(0.1))) {
    for (auto y0 : range(T(-1.5), T(1.5), T(0.1))) {
      x.set_value(x0);
      y.set_value(y0);

      CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

      CHECK_THAT(x.value(), WithinAbs(T(1), T(1e-3)));
      INFO(std::format("  (x₀, y₀) = ({}, {})", x0, y0));
      INFO(std::format("  (x, y) = ({}, {})", x.value(), y.value()));
      CHECK_THAT(y.value(), WithinAbs(T(1), T(1e-3)));
      INFO(std::format("  (x₀, y₀) = ({}, {})", x0, y0));
      INFO(std::format("  (x, y) = ({}, {})", x.value(), y.value()));
    }
  }
}

TEMPLATE_TEST_CASE("Problem - Minimum 2D distance with linear constraint",
                   "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();
  x.set_value(T(20));
  y.set_value(T(50));

  problem.minimize(sqrt(x * x + y * y));

  problem.subject_to(y == -x + T(5));

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

#if defined(__linux__) && defined(__aarch64__)
  // FIXME: Fails on Linux aarch64 with "line search failed"
  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::LINE_SEARCH_FAILED);
  SKIP("Fails with \"line search failed\"");
#else
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
#endif

  CHECK_THAT(x.value(), WithinAbs(T(2.5), T(1e-2)));
  CHECK_THAT(y.value(), WithinAbs(T(2.5), T(1e-2)));
}

TEMPLATE_TEST_CASE("Problem - Conflicting bounds", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();

  problem.minimize(hypot(x, y));

  problem.subject_to(hypot(x, y) <= T(1));
  problem.subject_to(slp::bounds(T(0.5), x, T(-0.5)));

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONLINEAR);

  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::GLOBALLY_INFEASIBLE);
}

TEMPLATE_TEST_CASE("Problem - Wachter and Biegler line search failure",
                   "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  // See example 19.2 of [1]
  //
  // [1] Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19.
  //     Springer, 2006.

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto s1 = problem.decision_variable();
  auto s2 = problem.decision_variable();
  x.set_value(T(-2));
  s1.set_value(T(3));
  s2.set_value(T(1));

  problem.minimize(x);

  problem.subject_to(pow(x, T(2)) - s1 - T(1) == T(0));
  problem.subject_to(x - s2 - T(0.5) == T(0));
  problem.subject_to(s1 >= T(0));
  problem.subject_to(s2 >= T(0));

  CHECK(problem.cost_function_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(x.value(), WithinAbs(T(1), T(1e-6)));
  CHECK_THAT(s1.value(), WithinAbs(T(0), T(1e-6)));
  CHECK_THAT(s2.value(), WithinAbs(T(0.5), T(1e-6)));
}
