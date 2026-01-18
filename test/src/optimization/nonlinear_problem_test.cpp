// Copyright (c) Sleipnir contributors

#include <format>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "range.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("Problem - Quartic", "[Problem]", SCALAR_TYPES) {
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
                   "[Problem]", SCALAR_TYPES) {
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

      // Local minimum at (0, 0)
      // Global minimum at (1, 1)
      INFO(std::format("  (x₀, y₀) = ({}, {})", x0, y0));
      Eigen::Vector<T, 2> sol{x.value(), y.value()};
      CHECK_THAT(sol, WithinAbs(Eigen::Vector<T, 2>{T(0), T(0)}, T(1e-2)) ||
                          WithinAbs(Eigen::Vector<T, 2>{T(1), T(1)}, T(1e-2)));
    }
  }
}

TEMPLATE_TEST_CASE("Problem - Rosenbrock with disk constraint", "[Problem]",
                   SCALAR_TYPES) {
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

      INFO(std::format("  (x₀, y₀) = ({}, {})", x0, y0));
      Eigen::Vector<T, 2> sol{x.value(), y.value()};
      CHECK_THAT(sol, WithinAbs(Eigen::Vector<T, 2>{T(1), T(1)}, T(1e-3)));
    }
  }
}

TEMPLATE_TEST_CASE("Problem - Gomez and Levy", "[Problem]", SCALAR_TYPES) {
  // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization

  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();

  problem.minimize(T(4) * pow(x, T(2)) - T(2.1) * pow(x, T(4)) +
                   pow(x, T(6)) / T(3) + x * y - T(4) * pow(y, T(2)) +
                   T(4) * pow(y, T(4)));

  auto c = -sin(T(4) * T(std::numbers::pi) * x) +
           T(2) * pow(sin(T(2) * T(std::numbers::pi) * y), T(2));
  problem.subject_to(c <= T(1.5));
  problem.subject_to(-T(1) <= x);
  problem.subject_to(x <= T(0.75));
  problem.subject_to(-T(1) <= y);
  problem.subject_to(y <= T(1));

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONLINEAR);

  // The feasible region is a set of disconnected islands, so each solve
  // converges to one of many local minima. Solving from a grid of initial
  // guesses with one Problem also exercises warm starts from a previous
  // solution far from the new initial guess.
  for (auto x0 : range(T(-0.95), T(0.75), T(0.1))) {
    for (auto y0 : range(T(-0.95), T(1), T(0.1))) {
      x.set_value(x0);
      y.set_value(y0);

      CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

      INFO(std::format("  (x₀, y₀) = ({}, {})", x0, y0));
      CHECK(c.value() <= T(1.5) + T(1e-8));
      CHECK(x.value() >= -T(1) - T(1e-8));
      CHECK(x.value() <= T(0.75) + T(1e-8));
      CHECK(y.value() >= -T(1) - T(1e-8));
      CHECK(y.value() <= T(1) + T(1e-8));
    }
  }
}

TEMPLATE_TEST_CASE("Problem - Minimum 2D distance with linear constraint",
                   "[Problem]", SCALAR_TYPES) {
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

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(x.value(), WithinAbs(T(2.5), T(1e-2)));
  CHECK_THAT(y.value(), WithinAbs(T(2.5), T(1e-2)));
}

TEMPLATE_TEST_CASE("Problem - Conflicting bounds", "[Problem]", SCALAR_TYPES) {
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
                   "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  // Example 19.2 of [1] exercises a deficiency of line search IPMs that must be
  // overcome by feasibility restoration. Depending on the specific trajectory
  // taken, either success or local infeasibility are expected.
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

  auto status = problem.solve({.diagnostics = true});
  CHECK_THAT(status, Equals(slp::ExitStatus::SUCCESS) ||
                         Equals(slp::ExitStatus::LOCALLY_INFEASIBLE));

  if (status == slp::ExitStatus::SUCCESS) {
    CHECK_THAT(x.value(), WithinAbs(T(1), T(1e-6)));
    CHECK_THAT(s1.value(), WithinAbs(T(0), T(1e-6)));
    CHECK_THAT(s2.value(), WithinAbs(T(0.5), T(1e-6)));
  } else {
    CHECK_THAT(x.value(), WithinAbs(T(-1), T(1e-6)));
    CHECK_THAT(s1.value(), WithinAbs(T(0), T(1e-6)));
    CHECK_THAT(s2.value(), WithinAbs(T(0), T(3e-5)));
  }
}

TEMPLATE_TEST_CASE("Problem - Hock-Schittkowski 13", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  // Problem 13 of [1] (see
  // https://vanderbei.princeton.edu/ampl/nlmodels/cute/hs013.mod) has
  // constraint gradients that are linearly dependent at the solution.
  //
  // [1] Hock, W. and Schittkowski, K. "Test Examples for Nonlinear Programming
  //     Codes", Springer, 1981.

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();
  x.set_value(T(-2));
  y.set_value(T(-2));

  problem.minimize(pow(x - T(2), 2) + pow(y, 2));

  problem.subject_to(pow(T(1) - x, 3) >= y);
  problem.subject_to(x >= T(0));
  problem.subject_to(y >= T(0));

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONLINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // The solution has no finite Lagrange multipliers, so the iterates only
  // approach it to within the square root of the tolerance
  CHECK_THAT(x.value(), WithinAbs(T(1), T(1e-4)));
  CHECK_THAT(y.value(), WithinAbs(T(0), T(1e-4)));
}
