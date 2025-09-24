// Copyright (c) Sleipnir contributors

#include <cmath>
#include <limits>
#include <numbers>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression.hpp>
#include <sleipnir/autodiff/expression_functions.hpp>
#include <sleipnir/autodiff/expression_type.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"

using slp::detail::constant_ptr;
using slp::detail::DecisionVariableExpression;
using slp::detail::make_expression_ptr;

TEMPLATE_TEST_CASE("Expression - is_constant", "[Expression]", SCALAR_TYPES) {
  using T = TestType;

  // Constant zero
  auto zero = constant_ptr(T(0));
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  CHECK(zero->is_constant(T(0)));
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  CHECK_FALSE(zero->is_constant(T(1)));

  // Constant one
  auto one = constant_ptr(T(1));
  CHECK_FALSE(one->is_constant(T(0)));
  CHECK(one->is_constant(T(1)));

  // Linear variable
  auto x = make_expression_ptr<DecisionVariableExpression<T>>(T(1));
  CHECK_FALSE(x->is_constant(T(0)));
  CHECK_FALSE(x->is_constant(T(1)));
}

TEMPLATE_TEST_CASE("Expression - constant_ptr", "[Expression]", SCALAR_TYPES) {
  using T = TestType;

  CHECK(constant_ptr(T(0))->is_constant(T(0)));
}

TEMPLATE_TEST_CASE("Expression - Prune multiply", "[Expression]",
                   SCALAR_TYPES) {
  using T = TestType;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));
  auto two = constant_ptr(T(2));

  CHECK((zero * one)->is_constant(T(0)));
  CHECK((zero * two)->is_constant(T(0)));
  CHECK((one * zero)->is_constant(T(0)));
  CHECK(one * one == one);
  CHECK(one * two == two);
  CHECK(two * one == two);
}

TEMPLATE_TEST_CASE("Expression - Prune divide", "[Expression]", SCALAR_TYPES) {
  using T = TestType;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));
  auto two = constant_ptr(T(2));

  CHECK((zero / one)->is_constant(T(0)));
  CHECK(one / one == one);
  CHECK(two / one == two);
}

TEMPLATE_TEST_CASE("Expression - Prune binary plus", "[Expression]",
                   SCALAR_TYPES) {
  using T = TestType;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));
  auto two = constant_ptr(T(2));

  CHECK(zero + zero == zero);
  CHECK(zero + one == one);
  CHECK(zero + two == two);
  CHECK(one + zero == one);
  CHECK(two + zero == two);
}

TEMPLATE_TEST_CASE("Expression - Prune binary minus", "[Expression]",
                   SCALAR_TYPES) {
  using T = TestType;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));
  auto two = constant_ptr(T(2));

  CHECK((zero - zero)->is_constant(T(0)));
  CHECK(one - zero == one);
  CHECK(two - zero == two);
}

TEMPLATE_TEST_CASE("Expression - Prune unary plus", "[Expression]",
                   SCALAR_TYPES) {
  using T = TestType;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));
  auto two = constant_ptr(T(2));

  CHECK((+zero)->is_constant(T(0)));
  CHECK(+one == one);
  CHECK(+two == two);
}

TEMPLATE_TEST_CASE("Expression - Prune unary minus", "[Expression]",
                   SCALAR_TYPES) {
  using T = TestType;

  auto zero = constant_ptr(T(0));

  CHECK((-zero)->is_constant(T(0)));
}

TEMPLATE_TEST_CASE("Expression - Prune abs()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::abs(zero)->is_constant(T(0)));
  CHECK(slp::detail::abs(one)->is_constant(T(1)));
}

TEMPLATE_TEST_CASE("Expression - Prune acos()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::acos;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::acos(zero)->is_constant(T(std::numbers::pi / 2)));
  CHECK(slp::detail::acos(one)->is_constant(acos(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune asin()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::asin;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::asin(zero)->is_constant(T(0)));
  CHECK(slp::detail::asin(one)->is_constant(asin(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune atan()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::atan;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::atan(zero)->is_constant(T(0)));
  CHECK(slp::detail::atan(one)->is_constant(atan(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune atan2()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::atan2;

  auto negative_one = constant_ptr(T(-1));
  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::atan2(zero, one)->is_constant(T(0)));
  CHECK(slp::detail::atan2(one, zero)->is_constant(T(std::numbers::pi / 2)));
  CHECK(
      slp::detail::atan2(zero, negative_one)->is_constant(atan2(T(0), T(-1))));
  CHECK(
      slp::detail::atan2(negative_one, zero)->is_constant(atan2(T(-1), T(0))));
  CHECK(slp::detail::atan2(one, one)->is_constant(atan2(T(1), T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune cbrt()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::cbrt;

  auto negative_one = constant_ptr(T(-1));
  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));
  auto two = constant_ptr(T(2));

  CHECK(slp::detail::cbrt(negative_one) == negative_one);
  CHECK(slp::detail::cbrt(zero)->is_constant(T(0)));
  CHECK(slp::detail::cbrt(one) == one);

  auto c = slp::detail::cbrt(two);
  CHECK(c->type() == slp::ExpressionType::CONSTANT);
  CHECK_THAT(c->val, WithinAbs(T(cbrt(T(2))), T(1e-15)));
}

TEMPLATE_TEST_CASE("Expression - Prune cos()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::cos;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::cos(zero)->is_constant(T(1)));
  CHECK(slp::detail::cos(one)->is_constant(cos(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune cosh()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::cosh;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::cosh(zero)->is_constant(T(1)));
  CHECK(slp::detail::cosh(one)->is_constant(cosh(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune erf()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::erf;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::erf(zero)->is_constant(T(0)));
  CHECK(slp::detail::erf(one)->is_constant(erf(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune exp()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::exp;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::exp(zero)->is_constant(T(1)));
  CHECK(slp::detail::exp(one)->is_constant(exp(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune hypot()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::hypot;

  auto negative_one = constant_ptr(T(-1));
  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  // hypot(x, y)
  CHECK(slp::detail::hypot(zero, zero) == zero);
  CHECK(slp::detail::hypot(zero, one)->is_constant(T(1)));
  CHECK(slp::detail::hypot(one, zero)->is_constant(T(1)));
  CHECK(
      slp::detail::hypot(zero, negative_one)->is_constant(hypot(T(0), T(-1))));
  CHECK(
      slp::detail::hypot(negative_one, zero)->is_constant(hypot(T(-1), T(0))));
  CHECK(slp::detail::hypot(one, one)->is_constant(T(std::numbers::sqrt2)));

  // hypot(x, y, z)
  CHECK(slp::detail::hypot(zero, zero, zero) == zero);
  CHECK(slp::detail::hypot(zero, zero, one)->is_constant(T(1)));
  CHECK(slp::detail::hypot(zero, one, zero)->is_constant(T(1)));
  CHECK(slp::detail::hypot(one, zero, zero)->is_constant(T(1)));
  CHECK(
      slp::detail::hypot(zero, one, one)->is_constant(T(std::numbers::sqrt2)));
  CHECK(
      slp::detail::hypot(one, zero, one)->is_constant(T(std::numbers::sqrt2)));
  CHECK(
      slp::detail::hypot(one, one, zero)->is_constant(T(std::numbers::sqrt2)));
  CHECK(
      slp::detail::hypot(one, one, one)->is_constant(hypot(T(1), T(1), T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune if_else()", "[Expression]",
                   SCALAR_TYPES) {
  using T = TestType;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));
  auto x = make_expression_ptr<DecisionVariableExpression<T>>(T(1));
  auto y = make_expression_ptr<DecisionVariableExpression<T>>(T(2));
  auto lt = +[](T a, T b) { return a < b; };

  // Identical branches
  CHECK(slp::detail::if_else(lt, x, y, x, x) == x);

  // Equal constant branches
  auto other_one = constant_ptr(T(1));
  CHECK(slp::detail::if_else(lt, x, y, one, other_one) == one);

  // Constant condition
  CHECK(slp::detail::if_else(lt, zero, one, x, y) == x);
  CHECK(slp::detail::if_else(lt, one, zero, x, y) == y);

  // Non-constant condition
  CHECK(slp::detail::if_else(lt, x, one, x, y)->name() == "if-else");
  CHECK(slp::detail::if_else(lt, zero, y, x, y)->name() == "if-else");
}

TEMPLATE_TEST_CASE("Expression - Prune log()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::log;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(
      slp::detail::log(zero)->is_constant(-std::numeric_limits<T>::infinity()));
  CHECK(slp::detail::log(one)->is_constant(log(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune log10()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::log10;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::log10(zero)->is_constant(
      -std::numeric_limits<T>::infinity()));
  CHECK(slp::detail::log10(one)->is_constant(log10(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune max()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;

  auto negative_one = constant_ptr(T(-1));
  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::max(negative_one, zero)->is_constant(T(0)));
  CHECK(slp::detail::max(zero, negative_one)->is_constant(T(0)));
  CHECK(slp::detail::max(zero, zero)->is_constant(T(0)));
  CHECK(slp::detail::max(zero, one)->is_constant(T(1)));
  CHECK(slp::detail::max(one, zero)->is_constant(T(1)));
}

TEMPLATE_TEST_CASE("Expression - Prune min()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;

  auto negative_one = constant_ptr(T(-1));
  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::min(negative_one, zero)->is_constant(T(-1)));
  CHECK(slp::detail::min(zero, negative_one)->is_constant(T(-1)));
  CHECK(slp::detail::min(zero, zero)->is_constant(T(0)));
  CHECK(slp::detail::min(zero, one)->is_constant(T(0)));
  CHECK(slp::detail::min(one, zero)->is_constant(T(0)));
}

TEMPLATE_TEST_CASE("Expression - Prune pow()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));
  auto two = constant_ptr(T(2));

  CHECK(slp::detail::pow(zero, zero)->is_constant(T(0)));
  CHECK(slp::detail::pow(zero, one)->is_constant(T(0)));
  CHECK(slp::detail::pow(zero, two)->is_constant(T(0)));
  CHECK(slp::detail::pow(one, zero) == one);
  CHECK(slp::detail::pow(one, one) == one);
  CHECK(slp::detail::pow(one, two) == one);
  CHECK(slp::detail::pow(two, zero)->is_constant(T(1)));
  CHECK(slp::detail::pow(two, one) == two);
  CHECK(slp::detail::pow(two, two)->is_constant(T(4)));

  auto x = make_expression_ptr<DecisionVariableExpression<T>>(T(2));
  CHECK(slp::detail::pow(x, one) == x);
  CHECK(slp::detail::pow(x, two)->name() == "multiplication");
  CHECK(slp::detail::pow(x, constant_ptr(T(0.5)))->name() == "sqrt");
}

TEMPLATE_TEST_CASE("Expression - Prune sign()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;

  auto zero = constant_ptr(T(0));

  CHECK(slp::detail::sign(constant_ptr(T(-2)))->is_constant(T(-1)));
  CHECK(slp::detail::sign(zero) == zero);
  CHECK(slp::detail::sign(constant_ptr(T(2)))->is_constant(T(1)));
}

TEMPLATE_TEST_CASE("Expression - Prune sin()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::sin;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::sin(zero)->is_constant(T(0)));
  CHECK(slp::detail::sin(one)->is_constant(sin(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune sinh()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::sinh;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::sinh(zero)->is_constant(T(0)));
  CHECK(slp::detail::sinh(one)->is_constant(sinh(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune sqrt()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));
  auto two = constant_ptr(T(2));

  CHECK(slp::detail::sqrt(zero)->is_constant(T(0)));
  CHECK(slp::detail::sqrt(one) == one);
  CHECK(slp::detail::sqrt(two)->is_constant(T(std::numbers::sqrt2)));
}

TEMPLATE_TEST_CASE("Expression - Prune tan()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::tan;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::tan(zero)->is_constant(T(0)));
  CHECK(slp::detail::tan(one)->is_constant(tan(T(1))));
}

TEMPLATE_TEST_CASE("Expression - Prune tanh()", "[Expression]", SCALAR_TYPES) {
  using T = TestType;
  using std::tanh;

  auto zero = constant_ptr(T(0));
  auto one = constant_ptr(T(1));

  CHECK(slp::detail::tanh(zero)->is_constant(T(0)));
  CHECK(slp::detail::tanh(one)->is_constant(tanh(T(1))));
}
