// Copyright (c) Sleipnir contributors

#include <format>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <sleipnir/optimization/lint.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types.hpp"

using Catch::Matchers::ContainsSubstring;

// Returns the squared distance between two 2D points
template <typename A, typename B>
auto squared_distance(const A& a, const B& b) {
  auto dx = a[0] - b[0];
  auto dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

// Returns the distance between two 2D points
template <typename A, typename B>
auto distance(const A& a, const B& b) {
  return slp::sqrt(squared_distance(a, b));
}

TEMPLATE_TEST_CASE("Problem - Lint clean problem", "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  x.set_value(T(0.5));
  problem.minimize((x - 1) * (x - 1) + (y - 2) * (y - 2));
  problem.subject_to(x + y <= 3);
  problem.subject_to(x >= 0);
  problem.subject_to(x * x + y * y == 4 * x);

  CHECK(problem.lint().empty());

  // Linting doesn't change the problem
  CHECK(x.value() == T(0.5));
  CHECK(y.value() == T(0));
}

TEMPLATE_TEST_CASE("Problem - Lint unused decision variables", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable(5);
  problem.minimize(x[0] * x[0]);
  problem.subject_to(x[2] >= 1);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 1);
  CHECK(findings[0].kind == slp::LintKind::UNUSED_DECISION_VARIABLE);
  CHECK(findings[0].decision_variables == std::vector{1, 3, 4});
  CHECK(findings[0].equality_constraints.empty());
  CHECK(findings[0].inequality_constraints.empty());
  CHECK(findings[0].message ==
        "Decision variables x[1], x[3], and x[4] don't appear in the cost "
        "function or any constraint. Remove them, or use them in the "
        "problem.");
  CHECK(std::format("{}", findings[0]) == findings[0].message);
}

TEMPLATE_TEST_CASE("Problem - Lint constant constraints", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto p = problem.parameter();
  p.set_value(T(1));
  problem.minimize(x * x);
  problem.subject_to(p == 2);
  problem.subject_to(p >= 0);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 2);

  CHECK(findings[0].kind == slp::LintKind::CONSTANT_CONSTRAINT);
  CHECK(findings[0].equality_constraints == std::vector{0});
  CHECK(findings[0].inequality_constraints.empty());
  CHECK_THAT(findings[0].message, ContainsSubstring("can't be satisfied"));
  CHECK_THAT(findings[0].message,
             ContainsSubstring("With the parameters' current values"));

  CHECK(findings[1].kind == slp::LintKind::CONSTANT_CONSTRAINT);
  CHECK(findings[1].equality_constraints.empty());
  CHECK(findings[1].inequality_constraints == std::vector{0});
  CHECK_THAT(findings[1].message, ContainsSubstring("always satisfied"));
}

TEMPLATE_TEST_CASE("Problem - Lint linear equality constraints: redundant",
                   "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(x * x + y * y);
  problem.subject_to(x + 2 * y == 1);
  problem.subject_to(2 * x + 4 * y == 2);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 1);
  CHECK(findings[0].kind == slp::LintKind::REDUNDANT_CONSTRAINTS);
  CHECK(findings[0].decision_variables == std::vector{0, 1});
  CHECK(findings[0].equality_constraints == std::vector{0, 1});
  CHECK(findings[0].message ==
        "Equality constraints 0 and 1 set x[0] + 2·x[1] = 1. Keep one "
        "of them; duplicate equality constraints make the constraint "
        "Jacobian rank-deficient.");
}

TEMPLATE_TEST_CASE("Problem - Lint linear equality constraints: conflicting",
                   "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(x * x + y * y);
  problem.subject_to(x - y == 1);
  problem.subject_to(y - x == 1);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 1);
  CHECK(findings[0].kind == slp::LintKind::CONFLICTING_CONSTRAINTS);
  CHECK(findings[0].equality_constraints == std::vector{0, 1});
  CHECK(findings[0].message ==
        "Equality constraints 0 and 1 set x[0] − x[1] to different values "
        "(1 and -1), so the problem is infeasible.");
}

TEMPLATE_TEST_CASE(
    "Problem - Lint linear equality constraints: inequality constraints "
    "implied or violated",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(x * x + y * y);
  problem.subject_to(x + y == 1);
  problem.subject_to(x + y >= 0);
  problem.subject_to(x + y <= 0);
  problem.subject_to(x + y <= 1);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 2);

  CHECK(findings[0].kind == slp::LintKind::REDUNDANT_CONSTRAINTS);
  CHECK(findings[0].equality_constraints == std::vector{0});
  CHECK(findings[0].inequality_constraints == std::vector{0, 2});
  CHECK(findings[0].message ==
        "Equality constraint 0 sets x[0] + x[1] = 1, which satisfies "
        "inequality constraints 0 and 2. Remove them.");

  CHECK(findings[1].kind == slp::LintKind::CONFLICTING_CONSTRAINTS);
  CHECK(findings[1].equality_constraints == std::vector{0});
  CHECK(findings[1].inequality_constraints == std::vector{1});
}

TEMPLATE_TEST_CASE(
    "Problem - Lint linear inequality constraints: redundant bounds",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  problem.minimize(x * x);
  problem.subject_to(x >= 0);
  problem.subject_to(x >= 1);
  problem.subject_to(x <= 5);
  problem.subject_to(-x >= -3);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 2);

  CHECK(findings[0].kind == slp::LintKind::REDUNDANT_CONSTRAINTS);
  CHECK(findings[0].inequality_constraints == std::vector{0, 1});
  CHECK(findings[0].message ==
        "Inequality constraint 1 requires x[0] ≥ 1, which implies "
        "inequality constraint 0. Remove it.");

  CHECK(findings[1].kind == slp::LintKind::REDUNDANT_CONSTRAINTS);
  CHECK(findings[1].inequality_constraints == std::vector{2, 3});
  CHECK(findings[1].message ==
        "Inequality constraint 3 requires x[0] ≤ 3, which implies "
        "inequality constraint 2. Remove it.");
}

TEMPLATE_TEST_CASE(
    "Problem - Lint linear inequality constraints: equality as inequalities",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(x * x + y * y);
  problem.subject_to(2 * x - 2 * y >= 2);
  problem.subject_to(x - y <= 1);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 1);
  CHECK(findings[0].kind == slp::LintKind::EQUALITY_AS_INEQUALITIES);
  CHECK(findings[0].inequality_constraints == std::vector{0, 1});
  CHECK(findings[0].message ==
        "Inequality constraints 0 and 1 require x[0] − x[1] ≥ 1 and "
        "x[0] − x[1] ≤ 1. Replace them with the equality constraint "
        "x[0] − x[1] = 1, which the solver handles directly instead of "
        "through two inequality constraints with no strict interior.");
}

TEMPLATE_TEST_CASE(
    "Problem - Lint linear inequality constraints: conflicting bounds",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  problem.minimize(x * x);
  problem.subject_to(x >= 2);
  problem.subject_to(x <= 1);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 1);
  CHECK(findings[0].kind == slp::LintKind::CONFLICTING_CONSTRAINTS);
  CHECK(findings[0].inequality_constraints == std::vector{0, 1});
}

TEMPLATE_TEST_CASE(
    "Problem - Lint linear inequality constraints: different linear "
    "combinations",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(x * x + y * y);
  problem.subject_to(x + y >= 0);
  problem.subject_to(x + 2 * y >= 1);
  problem.subject_to(x - y <= 1);

  CHECK(problem.lint().empty());
}

TEMPLATE_TEST_CASE("Problem - Lint nonsmooth functions: abs(a) <= b",
                   "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(x * x + y * y);
  problem.subject_to(slp::abs(y - 1) <= 2);
  problem.subject_to(slp::abs(x) <= y);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 1);
  CHECK(findings[0].kind == slp::LintKind::NONSMOOTH_FUNCTION);
  CHECK_FALSE(findings[0].cost);
  CHECK(findings[0].equality_constraints.empty());
  CHECK(findings[0].inequality_constraints == std::vector{0, 1});
  CHECK(findings[0].message ==
        "Inequality constraints 0 and 1 have the form abs(a) ≤ b, where a "
        "and b are expressions. abs(a) isn't differentiable where a = 0. "
        "Write them as −b ≤ a ≤ b instead.");
}

TEMPLATE_TEST_CASE(
    "Problem - Lint nonsmooth functions: max(a, b) <= c and min(a, b) >= c",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(x * x + y * y);
  problem.subject_to(slp::max(x, y) <= 1);
  problem.subject_to(-1 <= slp::min(x, y));

  auto findings = problem.lint();
  REQUIRE(findings.size() == 2);
  CHECK(findings[0].inequality_constraints == std::vector{0});
  CHECK(findings[0].message ==
        "Inequality constraint 0 has the form max(a, b) ≤ c, where a, b, "
        "and c are expressions. max(a, b) isn't differentiable where a = b. "
        "Write it as a ≤ c and b ≤ c instead.");
  CHECK(findings[1].inequality_constraints == std::vector{1});
  CHECK_THAT(findings[1].message,
             ContainsSubstring("Write it as a ≥ c and b ≥ c instead."));
}

TEMPLATE_TEST_CASE("Problem - Lint nonsmooth functions: norms", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto t = problem.real_decision_variable(2);
  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  auto r = problem.parameter();
  r.set_value(T(2));
  Eigen::Vector2<T> target{T(1), T(2)};
  problem.minimize(slp::sqrt(x * x + y * y));
  problem.subject_to(distance(t, target) <= 1);
  problem.subject_to(slp::hypot(x, y) >= r);
  problem.subject_to(distance(t, target) == 1);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 2);
  CHECK(findings[0].cost);
  CHECK(findings[0].equality_constraints.empty());
  CHECK(findings[0].inequality_constraints.empty());
  CHECK(findings[0].message ==
        "The cost function is a norm, like hypot(), which isn't "
        "differentiable where it's 0. Minimize the squared norm instead, "
        "which has the same minimizer.");
  CHECK_FALSE(findings[1].cost);
  CHECK(findings[1].equality_constraints == std::vector{0});
  CHECK(findings[1].inequality_constraints == std::vector{0, 1});
  CHECK(findings[1].message ==
        "Equality constraint 0 and inequality constraints 0 and 1 compare a "
        "norm, like hypot(), to a value that doesn't depend on decision "
        "variables. The norm isn't differentiable where it's 0. Compare the "
        "squared norm to the value's square instead.");
}

TEMPLATE_TEST_CASE(
    "Problem - Lint nonsmooth functions: forms without a differentiable "
    "replacement",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto a = problem.real_decision_variable(2);
  auto b = problem.real_decision_variable(2);
  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  Eigen::Vector2<T> start{T(0), T(0)};
  Eigen::Vector2<T> end{T(3), T(4)};

  // Path length and L1 cost
  problem.minimize(distance(a, start) + distance(b, a) + distance(end, b) +
                   slp::abs(x) + slp::max(x, y));

  // Bounds of the other direction, equalities, and norms compared to
  // decision variables
  problem.subject_to(slp::abs(x) >= 1);
  problem.subject_to(slp::abs(y) == 1);
  problem.subject_to(slp::max(x, y) >= 1);
  problem.subject_to(slp::min(x, y) <= 1);
  problem.subject_to(slp::sign(x) == 1);
  problem.subject_to(distance(a, b) <= x);

  CHECK(problem.lint().empty());
}

TEMPLATE_TEST_CASE("Problem - Lint nonsmooth functions: smooth uses",
                   "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto t = problem.real_decision_variable(2);
  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  Eigen::Vector2<T> target{T(1), T(2)};
  problem.minimize(slp::sqrt(x * x + y * y + 1) + slp::sqrt(x + 2));
  problem.subject_to(squared_distance(t, target) <= 1);
  problem.subject_to(slp::hypot(x, T(1)) <= 2);
  auto p = problem.parameter();
  p.set_value(T(-3));
  problem.subject_to(slp::abs(p) * y >= 0);

  CHECK(problem.lint().empty());
}

TEMPLATE_TEST_CASE(
    "Problem - Lint badly scaled coefficients: within a linear constraint",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(x * x);
  problem.subject_to(x + T(1e9) * y >= 1);
  problem.subject_to(x - T(2e9) * y <= 3);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 1);
  CHECK(findings[0].kind == slp::LintKind::BADLY_SCALED);
  CHECK_FALSE(findings[0].cost);
  CHECK(findings[0].decision_variables == std::vector{1});
  CHECK(findings[0].inequality_constraints == std::vector{0, 1});
  CHECK(findings[0].message ==
        "Coefficients in inequality constraints 0 and 1 span a ratio of up "
        "to 2.0e+09, which costs the solver about half its precision. "
        "Substituting x[1] = 1e-09·y[1] for a new decision variable y[1], "
        "like by changing units, reduces it to 2.0e+00.");
}

TEMPLATE_TEST_CASE(
    "Problem - Lint badly scaled coefficients: within a quadratic cost "
    "function",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(T(1e-5) * x * x + T(1e5) * y * y);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 1);
  CHECK(findings[0].kind == slp::LintKind::BADLY_SCALED);
  CHECK(findings[0].cost);
  CHECK(findings[0].decision_variables == std::vector{1});
  CHECK_THAT(findings[0].message,
             ContainsSubstring("Coefficients in the cost function"));
  CHECK_THAT(findings[0].message, ContainsSubstring("x[1] = 1e-05·y[1]"));
}

TEMPLATE_TEST_CASE(
    "Problem - Lint badly scaled coefficients: not fixable by rescaling "
    "decision variables",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(x * x + y * y);
  problem.subject_to(x + T(1e9) * y >= 1);
  problem.subject_to(T(1e9) * x + y >= 1);

  CHECK(problem.lint().empty());
}

TEMPLATE_TEST_CASE(
    "Problem - Lint badly scaled coefficients: small constraint gradient",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(x * x + y * y);
  problem.subject_to(T(1e-9) * x >= -1);
  problem.subject_to(y >= 1);

  auto findings = problem.lint();
  REQUIRE(findings.size() == 1);
  CHECK(findings[0].kind == slp::LintKind::BADLY_SCALED);
  CHECK(findings[0].inequality_constraints == std::vector{0});
  CHECK(findings[0].message ==
        "Inequality constraint 0 has a gradient 1.0e+09 times smaller than "
        "inequality constraint 1's, even after the solver scales down "
        "gradients larger than 100. Multiply it by 1e+09 so its gradient is "
        "closer to 1.");
}

TEMPLATE_TEST_CASE(
    "Problem - Lint badly scaled coefficients: large gradients the solver "
    "scales down",
    "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.real_decision_variable();
  auto y = problem.real_decision_variable();
  problem.minimize(x * x + y * y);
  problem.subject_to(T(1e12) * x >= -1);
  problem.subject_to(y >= T(1e-3));
  problem.subject_to(T(1e-3) * x + y >= 0);

  CHECK(problem.lint().empty());
}
