// Copyright (c) Sleipnir contributors

#include <format>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_string_converters.hpp"
#include "range.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Problem - Quartic", "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  x.set_value(T(20.0));

  problem.minimize(slp::pow(x, T(4)));

  problem.subject_to(x >= T(1));

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK(x.value() == Catch::Approx(T(1.0)).margin(T(1e-6)));
}

TEMPLATE_TEST_CASE("Problem - Rosenbrock with cubic and line constraint",
                   "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
  for (auto x0 : range(T(-1.5), T(1.5), T(0.1))) {
    for (auto y0 : range(T(-0.5), T(2.5), T(0.1))) {
      slp::Problem<T> problem;

      auto x = problem.decision_variable();
      x.set_value(x0);
      auto y = problem.decision_variable();
      y.set_value(y0);

      problem.minimize(100 * slp::pow(y - slp::pow(x, 2), 2) +
                       slp::pow(1 - x, 2));

      problem.subject_to(y >= slp::pow(x - 1, 3) + 1);
      problem.subject_to(y <= -x + 2);

      CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
      CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
      CHECK(problem.inequality_constraint_type() ==
            slp::ExpressionType::NONLINEAR);

      CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

      auto near = [](T expected, T actual, T tolerance) {
        using std::abs;
        return abs(expected - actual) < tolerance;  // NOLINT
      };

      // Local minimum at (0.0, 0.0)
      // Global minimum at (1.0, 1.0)
      CHECK((near(T(0.0), x.value(), T(1e-2)) ||
             near(T(1.0), x.value(), T(1e-2))));
      INFO(std::format("  (x₀, y₀) = ({}, {})", x0, y0));
      INFO(std::format("  (x, y) = ({}, {})", x.value(), y.value()));
      CHECK((near(T(0.0), y.value(), T(1e-2)) ||
             near(T(1.0), y.value(), T(1e-2))));
      INFO(std::format("  (x₀, y₀) = ({}, {})", x0, y0));
      INFO(std::format("  (x, y) = ({}, {})", x.value(), y.value()));
    }
  }
}

TEMPLATE_TEST_CASE("Problem - Rosenbrock with disk constraint", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
  for (auto x0 : range(T(-1.5), T(1.5), T(0.1))) {
    for (auto y0 : range(T(-1.5), T(1.5), T(0.1))) {
      slp::Problem<T> problem;

      auto x = problem.decision_variable();
      x.set_value(x0);
      auto y = problem.decision_variable();
      y.set_value(y0);

      problem.minimize(slp::pow(T(1) - x, T(2)) +
                       T(100) * slp::pow(y - slp::pow(x, T(2)), T(2)));

      problem.subject_to(slp::pow(x, T(2)) + slp::pow(y, T(2)) <= T(2));

      CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
      CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
      CHECK(problem.inequality_constraint_type() ==
            slp::ExpressionType::QUADRATIC);

      CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

      CHECK(x.value() == Catch::Approx(T(1.0)).margin(T(1e-3)));
      INFO(std::format("  (x₀, y₀) = ({}, {})", x0, y0));
      INFO(std::format("  (x, y) = ({}, {})", x.value(), y.value()));
      CHECK(y.value() == Catch::Approx(T(1.0)).margin(T(1e-3)));
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
  x.set_value(T(20.0));

  auto y = problem.decision_variable();
  y.set_value(T(50.0));

  problem.minimize(slp::sqrt(x * x + y * y));

  problem.subject_to(y == -x + T(5.0));

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

  CHECK(x.value() == Catch::Approx(T(2.5)).margin(T(1e-2)));
  CHECK(y.value() == Catch::Approx(T(2.5)).margin(T(1e-2)));
}

TEMPLATE_TEST_CASE("Problem - Conflicting bounds", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();

  problem.minimize(slp::hypot(x, y));

  problem.subject_to(slp::hypot(x, y) <= T(1));
  problem.subject_to(x >= T(0.5));
  problem.subject_to(x <= T(-0.5));

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

  problem.subject_to(slp::pow(x, T(2)) - s1 - T(1) == T(0));
  problem.subject_to(x - s2 - T(0.5) == T(0));
  problem.subject_to(s1 >= T(0));
  problem.subject_to(s2 >= T(0));

  CHECK(problem.cost_function_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  // FIXME: Fails with "line search failed"
  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::LINE_SEARCH_FAILED);
  SKIP("Fails with \"line search failed\"");

  CHECK(x.value() == T(1.0));
  CHECK(s1.value() == T(0.0));
  CHECK(s2.value() == T(0.5));
}
