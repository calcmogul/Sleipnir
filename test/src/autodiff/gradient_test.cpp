// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_matchers.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Gradient - Trivial case", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> a;
  a.set_value(T(10));
  slp::Variable<T> b;
  b.set_value(T(20));
  slp::Variable c = a;

  CHECK(slp::Gradient(a, a).value().coeff(0) == T(1));
  CHECK(slp::Gradient(a, b).value().coeff(0) == T(0));
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1));
  CHECK(slp::Gradient(c, b).value().coeff(0) == T(0));
}

TEMPLATE_TEST_CASE("Gradient - Unary plus", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> a;
  a.set_value(T(10));
  slp::Variable c = +a;

  CHECK(c.value() == a.value());
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1));
}

TEMPLATE_TEST_CASE("Gradient - Unary minus", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> a;
  a.set_value(T(10));
  slp::Variable c = -a;

  CHECK(c.value() == -a.value());
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(-1));
}

TEMPLATE_TEST_CASE("Gradient - Identical variables", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> a;
  a.set_value(T(10));
  slp::Variable x = a;
  slp::Variable c = a * a + x;

  CHECK(c.value() == a.value() * a.value() + x.value());
  CHECK(slp::Gradient(c, a).value().coeff(0) ==
        T(2) * a.value() + slp::Gradient(x, a).value().coeff(0));
  CHECK(slp::Gradient(c, x).value().coeff(0) ==
        T(2) * a.value() * slp::Gradient(a, x).value().coeff(0) + T(1));
}

TEMPLATE_TEST_CASE("Gradient - Elementary", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> a;
  a.set_value(T(1));
  slp::Variable<T> b;
  b.set_value(T(2));

  auto c = -2 * a;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(-2));

  c = a / 3.0;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1.0 / 3.0));

  a.set_value(T(100));
  b.set_value(T(200));

  c = a + b;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1));
  CHECK(slp::Gradient(c, b).value().coeff(0) == T(1));

  c = a - b;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1));
  CHECK(slp::Gradient(c, b).value().coeff(0) == T(-1));

  c = -a + b;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(-1));
  CHECK(slp::Gradient(c, b).value().coeff(0) == T(1));

  c = a + 1;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1));
}

TEMPLATE_TEST_CASE("Gradient - Comparison", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  x.set_value(T(10));
  slp::Variable<T> a;
  a.set_value(T(10));
  slp::Variable<T> b;
  b.set_value(T(200));

  CHECK(a.value() == a.value());
  CHECK(a.value() == x.value());
  CHECK(a.value() == T(10));
  CHECK(T(10) == a.value());

  CHECK(a.value() != b.value());
  CHECK(a.value() != T(20));
  CHECK(T(20) != a.value());

  CHECK(a.value() < b.value());
  CHECK(a.value() < T(20));

  CHECK(b.value() > a.value());
  CHECK(T(20) > a.value());

  CHECK(a.value() <= a.value());
  CHECK(a.value() <= x.value());
  CHECK(a.value() <= b.value());
  CHECK(a.value() <= T(10));
  CHECK(a.value() <= T(20));

  CHECK(a.value() >= a.value());
  CHECK(x.value() >= a.value());
  CHECK(b.value() >= a.value());
  CHECK(T(10) >= a.value());
  CHECK(T(20) >= a.value());

  // Comparison between variables and expressions
  CHECK(a.value() == a.value() / a.value() * a.value());
  CHECK(a.value() / a.value() * a.value() == a.value());

  CHECK(a.value() != (a - a).value());
  CHECK((a - a).value() != a.value());

  CHECK((a - a).value() < a.value());
  CHECK(a.value() < (a + a).value());

  CHECK((a + a).value() > a.value());
  CHECK(a.value() > (a - a).value());

  CHECK(a.value() <= (a - a + a).value());
  CHECK((a - a + a).value() <= a.value());

  CHECK(a.value() <= (a + a).value());
  CHECK((a - a).value() <= a.value());

  CHECK(a.value() >= (a - a + a).value());
  CHECK((a - a + a).value() >= a.value());

  CHECK((a + a).value() >= a.value());
  CHECK(a.value() >= (a - a).value());
}

TEMPLATE_TEST_CASE("Gradient - Trigonometry", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::acos;
  using std::asin;
  using std::atan;
  using std::cos;
  using std::sin;
  using std::sqrt;
  using std::tan;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  x.set_value(T(0.5));

  // sin(x)
  CHECK(slp::sin(x).value() == sin(x.value()));

  auto g = slp::Gradient(slp::sin(x), x);
  CHECK(slp::value(g.get()).coeff(0) == cos(x.value()));
  CHECK(g.value().coeff(0) == cos(x.value()));

  // cos(x)
  CHECK(slp::cos(x).value() == cos(x.value()));

  g = slp::Gradient(slp::cos(x), x);
  CHECK(slp::value(g.get()).coeff(0) == -sin(x.value()));
  CHECK(g.value().coeff(0) == -sin(x.value()));

  // tan(x)
  CHECK(slp::tan(x).value() == tan(x.value()));

  g = slp::Gradient(slp::tan(x), x);
  CHECK(slp::value(g.get()).coeff(0) ==
        T(1) / (cos(x.value()) * cos(x.value())));
  CHECK(g.value().coeff(0) == T(1) / (cos(x.value()) * cos(x.value())));

  // asin(x)
  CHECK(slp::asin(x).value() == asin(x.value()));

  g = slp::Gradient(slp::asin(x), x);
  CHECK(slp::value(g.get()).coeff(0) ==
        T(1) / sqrt(T(1) - x.value() * x.value()));
  CHECK(g.value().coeff(0) == T(1) / sqrt(T(1) - x.value() * x.value()));

  // acos(x)
  CHECK(slp::acos(x).value() == acos(x.value()));

  g = slp::Gradient(slp::acos(x), x);
  CHECK(slp::value(g.get()).coeff(0) ==
        T(-1) / sqrt(T(1) - x.value() * x.value()));
  CHECK(g.value().coeff(0) == T(-1) / sqrt(T(1) - x.value() * x.value()));

  // atan(x)
  CHECK(slp::atan(x).value() == atan(x.value()));

  g = slp::Gradient(slp::atan(x), x);
  CHECK(slp::value(g.get()).coeff(0) == T(1) / (T(1) + x.value() * x.value()));
  CHECK(g.value().coeff(0) == T(1) / (T(1) + x.value() * x.value()));
}

TEMPLATE_TEST_CASE("Gradient - Hyperbolic", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::cosh;
  using std::sinh;
  using std::tanh;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  x.set_value(T(1));

  // sinh(x)
  CHECK(slp::sinh(x).value() == sinh(x.value()));

  auto g = slp::Gradient(slp::sinh(x), x);
  CHECK(slp::value(g.get()).coeff(0) == cosh(x.value()));
  CHECK(g.value().coeff(0) == cosh(x.value()));

  // cosh(x)
  CHECK(slp::cosh(x).value() == cosh(x.value()));

  g = slp::Gradient(slp::cosh(x), x);
  CHECK(slp::value(g.get()).coeff(0) == sinh(x.value()));
  CHECK(g.value().coeff(0) == sinh(x.value()));

  // tanh(x)
  CHECK(slp::tanh(x).value() == tanh(x.value()));

  g = slp::Gradient(slp::tanh(x), x);
  CHECK(slp::value(g.get()).coeff(0) ==
        T(1) / (cosh(x.value()) * cosh(x.value())));
  CHECK(g.value().coeff(0) == T(1) / (cosh(x.value()) * cosh(x.value())));
}

TEMPLATE_TEST_CASE("Gradient - Exponential", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::exp;
  using std::log;
  using std::log10;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  x.set_value(T(1));

  // log(x)
  CHECK(slp::log(x).value() == log(x.value()));

  auto g = slp::Gradient(slp::log(x), x);
  CHECK(slp::value(g.get()).coeff(0) == T(1) / x.value());
  CHECK(g.value().coeff(0) == T(1) / x.value());

  // log10(x)
  CHECK(slp::log10(x).value() == log10(x.value()));

  g = slp::Gradient(slp::log10(x), x);
  CHECK(slp::value(g.get()).coeff(0) == T(1) / (log(T(10)) * x.value()));
  CHECK(g.value().coeff(0) == T(1) / (log(T(10)) * x.value()));

  // exp(x)
  CHECK(slp::exp(x).value() == exp(x.value()));

  g = slp::Gradient(slp::exp(x), x);
  CHECK(slp::value(g.get()).coeff(0) == exp(x.value()));
  CHECK(g.value().coeff(0) == exp(x.value()));
}

TEMPLATE_TEST_CASE("Gradient - Power", "[Gradient]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::cbrt;
  using std::log;
  using std::pow;
  using std::sqrt;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  x.set_value(T(1));
  slp::Variable<T> a;
  a.set_value(T(2));
  slp::Variable y = 2 * a;

  // sqrt(x)
  CHECK(slp::sqrt(x).value() == sqrt(x.value()));

  auto g = slp::Gradient(slp::sqrt(x), x);
  CHECK(slp::value(g.get()).coeff(0) == T(0.5) / sqrt(x.value()));
  CHECK(g.value().coeff(0) == T(0.5) / sqrt(x.value()));

  // sqrt(a)
  CHECK(slp::sqrt(a).value() == sqrt(a.value()));

  g = slp::Gradient(slp::sqrt(a), a);
  CHECK(slp::value(g.get()).coeff(0) == T(0.5) / sqrt(a.value()));
  CHECK(g.value().coeff(0) == T(0.5) / sqrt(a.value()));

  // cbrt(x)
  CHECK(slp::cbrt(x).value() == cbrt(x.value()));

  g = slp::Gradient(slp::cbrt(x), x);
  CHECK(slp::value(g.get()).coeff(0) ==
        T(1) / (T(3) * cbrt(x.value()) * cbrt(x.value())));
  CHECK(g.value().coeff(0) ==
        T(1) / (T(3) * cbrt(x.value()) * cbrt(x.value())));

  // cbrt(a)
  CHECK(slp::cbrt(a).value() == cbrt(a.value()));

  g = slp::Gradient(slp::cbrt(a), a);
  CHECK(slp::value(g.get()).coeff(0) ==
        T(1) / (T(3) * cbrt(a.value()) * cbrt(a.value())));
  CHECK(g.value().coeff(0) ==
        T(1) / (T(3) * cbrt(a.value()) * cbrt(a.value())));

  // x²
  CHECK(slp::pow(x, T(2)).value() == pow(x.value(), T(2)));

  g = slp::Gradient(slp::pow(x, T(2)), x);
  CHECK(slp::value(g.get()).coeff(0) == T(2) * x.value());
  CHECK(g.value().coeff(0) == T(2) * x.value());

  // 2ˣ
  CHECK(slp::pow(T(2), x).value() == pow(T(2), x.value()));

  g = slp::Gradient(slp::pow(T(2), x), x);
  CHECK(slp::value(g.get()).coeff(0) == log(T(2)) * pow(T(2), x.value()));
  CHECK(g.value().coeff(0) == log(T(2)) * pow(T(2), x.value()));

  // xˣ
  CHECK(slp::pow(x, x).value() == pow(x.value(), x.value()));

  g = slp::Gradient(slp::pow(x, x), x);
  CHECK(slp::value(g.get()).coeff(0) ==
        (log(x.value()) + T(1)) * pow(x.value(), x.value()));
  CHECK(g.value().coeff(0) ==
        (log(x.value()) + T(1)) * pow(x.value(), x.value()));

  // y(a)
  CHECK(y.value() == T(2) * a.value());

  g = slp::Gradient(y, a);
  CHECK(slp::value(g.get()).coeff(0) == T(2));
  CHECK(g.value().coeff(0) == T(2));

  // xʸ(x)
  CHECK(slp::pow(x, y).value() == pow(x.value(), y.value()));

  g = slp::Gradient(slp::pow(x, y), x);
  CHECK(slp::value(g.get()).coeff(0) ==
        y.value() / x.value() * pow(x.value(), y.value()));
  CHECK(g.value().coeff(0) ==
        y.value() / x.value() * pow(x.value(), y.value()));

  // xʸ(a)
  CHECK(slp::pow(x, y).value() == pow(x.value(), y.value()));

  g = slp::Gradient(slp::pow(x, y), a);
  CHECK(slp::value(g.get()).coeff(0) ==
        pow(x.value(), y.value()) *
            (y.value() / x.value() * slp::Gradient(x, a).value().coeff(0) +
             log(x.value()) * slp::Gradient(y, a).value().coeff(0)));
  CHECK(g.value().coeff(0) ==
        pow(x.value(), y.value()) *
            (y.value() / x.value() * slp::Gradient(x, a).value().coeff(0) +
             log(x.value()) * slp::Gradient(y, a).value().coeff(0)));

  // xʸ(y)
  CHECK(slp::pow(x, y).value() == pow(x.value(), y.value()));

  g = slp::Gradient(slp::pow(x, y), y);
  CHECK(slp::value(g.get()).coeff(0) ==
        log(x.value()) * pow(x.value(), y.value()));
  CHECK(g.value().coeff(0) == log(x.value()) * pow(x.value(), y.value()));
}

TEMPLATE_TEST_CASE("Gradient - abs()", "[Gradient]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::abs;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  auto g = slp::Gradient(slp::abs(x), x);

  x.set_value(T(1));
  CHECK(slp::abs(x).value() == abs(x.value()));
  CHECK(slp::value(g.get()).coeff(0) == T(1));
  CHECK(g.value().coeff(0) == T(1));

  x.set_value(T(-1));
  CHECK(slp::abs(x).value() == abs(x.value()));
  CHECK(slp::value(g.get()).coeff(0) == T(-1));
  CHECK(g.value().coeff(0) == T(-1));

  x.set_value(T(0));
  CHECK(slp::abs(x).value() == abs(x.value()));
  CHECK(slp::value(g.get()).coeff(0) == T(0));
  CHECK(g.value().coeff(0) == T(0));
}

TEMPLATE_TEST_CASE("Gradient - atan2()", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::atan2;
  using std::cos;
  using std::sin;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  slp::Variable<T> y;

  // Testing atan2 function on (T, var)
  x.set_value(T(1));
  y.set_value(T(0.9));
  CHECK(slp::atan2(2.0, x).value() == atan2(T(2), x.value()));

  auto g = slp::Gradient(slp::atan2(T(2), x), x);
  CHECK_THAT(
      slp::value(g.get()).coeff(0),
      WithinAbs(T(-2) / (T(2) * T(2) + x.value() * x.value()), T(1e-15)));
  CHECK_THAT(
      g.value().coeff(0),
      WithinAbs(T(-2) / (T(2) * T(2) + x.value() * x.value()), T(1e-15)));

  // Testing atan2 function on (var, T)
  x.set_value(T(1));
  y.set_value(T(0.9));
  CHECK(slp::atan2(x, T(2)).value() == atan2(x.value(), T(2)));

  g = slp::Gradient(slp::atan2(x, T(2)), x);
  CHECK_THAT(slp::value(g.get()).coeff(0),
             WithinAbs(T(2) / (T(2) * T(2) + x.value() * x.value()), T(1e-15)));
  CHECK_THAT(g.value().coeff(0),
             WithinAbs(T(2) / (T(2) * T(2) + x.value() * x.value()), T(1e-15)));

  // Testing atan2 function on (var, var)
  x.set_value(T(1.1));
  y.set_value(T(0.9));
  CHECK(slp::atan2(y, x).value() == atan2(y.value(), x.value()));

  g = slp::Gradient(slp::atan2(y, x), y);
  CHECK_THAT(
      slp::value(g.get()).coeff(0),
      WithinAbs(x.value() / (x.value() * x.value() + y.value() * y.value()),
                T(1e-15)));
  CHECK_THAT(g.value().coeff(0), WithinAbs(x.value() / (x.value() * x.value() +
                                                        y.value() * y.value()),
                                           T(1e-15)));

  g = slp::Gradient(slp::atan2(y, x), x);
  CHECK_THAT(
      slp::value(g.get()).coeff(0),
      WithinAbs(-y.value() / (x.value() * x.value() + y.value() * y.value()),
                T(1e-15)));
  CHECK_THAT(g.value().coeff(0), WithinAbs(-y.value() / (x.value() * x.value() +
                                                         y.value() * y.value()),
                                           T(1e-15)));

  // Testing atan2 function on (expr, expr)
  CHECK(T(3) * slp::atan2(slp::sin(y), T(2) * x + T(1)).value() ==
        T(3) * atan2(sin(y.value()), T(2) * x.value() + T(1)));

  g = slp::Gradient(T(3) * slp::atan2(slp::sin(y), T(2) * x + T(1)), y);
  CHECK_THAT(
      slp::value(g.get()).coeff(0),
      WithinAbs(T(3) * (T(2) * x.value() + T(1)) * cos(y.value()) /
                    ((T(2) * x.value() + T(1)) * (T(2) * x.value() + T(1)) +
                     sin(y.value()) * sin(y.value())),
                T(1e-15)));
  CHECK_THAT(
      g.value().coeff(0),
      WithinAbs(T(3) * (T(2) * x.value() + T(1)) * cos(y.value()) /
                    ((T(2) * x.value() + T(1)) * (T(2) * x.value() + T(1)) +
                     sin(y.value()) * sin(y.value())),
                T(1e-15)));

  g = slp::Gradient(T(3) * slp::atan2(slp::sin(y), T(2) * x + T(1)), x);
  CHECK_THAT(
      slp::value(g.get()).coeff(0),
      WithinAbs(T(3) * T(-2) * sin(y.value()) /
                    ((T(2) * x.value() + T(1)) * (T(2) * x.value() + T(1)) +
                     sin(y.value()) * sin(y.value())),
                T(1e-15)));
  CHECK_THAT(
      g.value().coeff(0),
      WithinAbs(T(3) * T(-2) * sin(y.value()) /
                    ((T(2) * x.value() + T(1)) * (T(2) * x.value() + T(1)) +
                     sin(y.value()) * sin(y.value())),
                T(1e-15)));
}

TEMPLATE_TEST_CASE("Gradient - hypot()", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::hypot;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  slp::Variable<T> y;

  // Testing hypot function on (var, T)
  x.set_value(T(1.8));
  y.set_value(T(1.5));
  CHECK(slp::hypot(x, T(2)).value() == hypot(x.value(), T(2)));

  auto g = slp::Gradient(slp::hypot(x, T(2)), x);
  CHECK(slp::value(g.get()).coeff(0) == x.value() / hypot(x.value(), T(2)));
  CHECK(g.value().coeff(0) == x.value() / hypot(x.value(), T(2)));

  // Testing hypot function on (T, var)
  CHECK(slp::hypot(T(2), y).value() == hypot(T(2), y.value()));

  g = slp::Gradient(slp::hypot(T(2), y), y);
  CHECK(slp::value(g.get()).coeff(0) == y.value() / hypot(T(2), y.value()));
  CHECK(g.value().coeff(0) == y.value() / hypot(T(2), y.value()));

  // Testing hypot function on (var, var)
  x.set_value(T(1.3));
  y.set_value(T(2.3));
  CHECK(slp::hypot(x, y).value() == hypot(x.value(), y.value()));

  g = slp::Gradient(slp::hypot(x, y), x);
  CHECK(slp::value(g.get()).coeff(0) ==
        x.value() / hypot(x.value(), y.value()));
  CHECK(g.value().coeff(0) == x.value() / hypot(x.value(), y.value()));

  g = slp::Gradient(slp::hypot(x, y), y);
  CHECK(slp::value(g.get()).coeff(0) ==
        y.value() / hypot(x.value(), y.value()));
  CHECK(g.value().coeff(0) == y.value() / hypot(x.value(), y.value()));

  // Testing hypot function on (expr, expr)
  x.set_value(T(1.3));
  y.set_value(T(2.3));
  CHECK(slp::hypot(T(2) * x, T(3) * y).value() ==
        hypot(T(2) * x.value(), T(3) * y.value()));

  g = slp::Gradient(slp::hypot(T(2) * x, T(3) * y), x);
  CHECK(slp::value(g.get()).coeff(0) ==
        T(4) * x.value() / hypot(T(2) * x.value(), T(3) * y.value()));
  CHECK(g.value().coeff(0) ==
        T(4) * x.value() / hypot(T(2) * x.value(), T(3) * y.value()));

  g = slp::Gradient(slp::hypot(T(2) * x, T(3) * y), y);
  CHECK(slp::value(g.get()).coeff(0) ==
        T(9) * y.value() / hypot(T(2) * x.value(), T(3) * y.value()));
  CHECK(g.value().coeff(0) ==
        T(9) * y.value() / hypot(T(2) * x.value(), T(3) * y.value()));

  // Testing hypot function on (var, var, var)
  slp::Variable<T> z;
  x.set_value(T(1.3));
  y.set_value(T(2.3));
  z.set_value(T(3.3));
  CHECK(slp::hypot(x, y, z).value() == hypot(x.value(), y.value(), z.value()));

  g = slp::Gradient(slp::hypot(x, y, z), x);
  CHECK(slp::value(g.get()).coeff(0) ==
        (x.value() / hypot(x.value(), y.value(), z.value())));
  CHECK(g.value().coeff(0) ==
        x.value() / hypot(x.value(), y.value(), z.value()));

  g = slp::Gradient(slp::hypot(x, y, z), y);
  CHECK(slp::value(g.get()).coeff(0) ==
        y.value() / hypot(x.value(), y.value(), z.value()));
  CHECK(g.value().coeff(0) ==
        y.value() / hypot(x.value(), y.value(), z.value()));

  g = slp::Gradient(slp::hypot(x, y, z), z);
  CHECK(slp::value(g.get()).coeff(0) ==
        z.value() / hypot(x.value(), y.value(), z.value()));
  CHECK(g.value().coeff(0) ==
        z.value() / hypot(x.value(), y.value(), z.value()));
}

TEMPLATE_TEST_CASE("Gradient - max()", "[Gradient]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  x.set_value(T(2));

  slp::Variable<T> x2 = x * x;
  slp::Variable<T> x3 = x * x * x;

  // Testing lhs < rhs
  auto g = slp::Gradient(slp::max(x2, x3), x);
  CHECK(slp::max(x2, x3).value() == x3.value());
  CHECK(g.get().value().coeff(0) == slp::Gradient(x3, x).value().coeff(0));
  CHECK(g.value().coeff(0) == slp::Gradient(x3, x).value().coeff(0));

  // Testing lhs > rhs
  g = slp::Gradient(slp::max(x3, x2), x);
  CHECK(slp::max(x3, x2).value() == x3.value());
  CHECK(g.get().value().coeff(0) == slp::Gradient(x3, x).value().coeff(0));
  CHECK(g.value().coeff(0) == slp::Gradient(x3, x).value().coeff(0));

  // Testing lhs == rhs
  g = slp::Gradient(slp::max(x, x), x);
  CHECK(slp::max(x, x).value() == x.value());
  CHECK(g.get().value().coeff(0) == T(1));
  CHECK(g.value().coeff(0) == T(1));
}

TEMPLATE_TEST_CASE("Gradient - min()", "[Gradient]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  x.set_value(T(2));

  slp::Variable<T> x2 = x * x;
  slp::Variable<T> x3 = x * x * x;

  // Testing lhs < rhs
  auto g = slp::Gradient(slp::min(x2, x3), x);
  CHECK(slp::min(x2, x3).value() == x2.value());
  CHECK(g.get().value().coeff(0) == slp::Gradient(x2, x).value().coeff(0));
  CHECK(g.value().coeff(0) == slp::Gradient(x2, x).value().coeff(0));

  // Testing lhs > rhs
  g = slp::Gradient(slp::min(x3, x2), x);
  CHECK(slp::min(x3, x2).value() == x2.value());
  CHECK(g.get().value().coeff(0) == slp::Gradient(x2, x).value().coeff(0));
  CHECK(g.value().coeff(0) == slp::Gradient(x2, x).value().coeff(0));

  // Testing lhs == rhs
  g = slp::Gradient(slp::min(x, x), x);
  CHECK(slp::min(x, x).value() == x.value());
  CHECK(g.get().value().coeff(0) == T(1));
  CHECK(g.value().coeff(0) == T(1));
}

TEMPLATE_TEST_CASE("Gradient - Miscellaneous", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::abs;
  using std::exp;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;

  // dx/dx
  x.set_value(T(3));
  CHECK(slp::abs(x).value() == abs(x.value()));

  auto g = slp::Gradient(x, x);
  CHECK(slp::value(g.get()).coeff(0) == T(1));
  CHECK(g.value().coeff(0) == T(1));

  // erf(x)
  x.set_value(T(0.5));
  CHECK(slp::erf(x).value() == erf(x.value()));

  g = slp::Gradient(slp::erf(x), x);
  CHECK(slp::value(g.get()).coeff(0) ==
        T(2) * T(std::numbers::inv_sqrtpi) * exp(-x.value() * x.value()));
  CHECK(g.value().coeff(0) ==
        T(2) * T(std::numbers::inv_sqrtpi) * exp(-x.value() * x.value()));
}

TEMPLATE_TEST_CASE("Gradient - Variable reuse", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> a;
  a.set_value(T(10));

  slp::Variable<T> b;
  b.set_value(T(20));

  slp::Variable x = a * b;

  auto g = slp::Gradient(x, a);

  CHECK(slp::value(g.get()).coeff(0) == T(20));
  CHECK(g.value().coeff(0) == T(20));

  b.set_value(T(10));
  CHECK(slp::value(g.get()).coeff(0) == T(10));
  CHECK(g.value().coeff(0) == T(10));
}

TEMPLATE_TEST_CASE("Gradient - sign()", "[Gradient]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  auto sign = [](T x) {
    if (x < T(0)) {
      return T(-1);
    } else if (x == T(0)) {
      return T(0);
    } else {
      return T(1);
    }
  };

  slp::Variable<T> x;

  // sign(1)
  x.set_value(T(1));
  CHECK(slp::sign(x).value() == sign(x.value()));

  auto g = slp::Gradient(slp::sign(x), x);
  CHECK(slp::value(g.get()).coeff(0) == T(0));
  CHECK(g.value().coeff(0) == T(0));

  // sign(-1)
  x.set_value(T(-1));
  CHECK(slp::sign(x).value() == sign(x.value()));

  g = slp::Gradient(slp::sign(x), x);
  CHECK(slp::value(g.get()).coeff(0) == T(0));
  CHECK(g.value().coeff(0) == T(0));

  // sign(0)
  x.set_value(T(0));
  CHECK(slp::sign(x).value() == sign(x.value()));

  g = slp::Gradient(slp::sign(x), x);
  CHECK(slp::value(g.get()).coeff(0) == T(0));
  CHECK(g.value().coeff(0) == T(0));
}

TEMPLATE_TEST_CASE("Gradient - Non-scalar", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix<T> x{3};
  x[0].set_value(T(1));
  x[1].set_value(T(2));
  x[2].set_value(T(3));

  // y = [x₁ + 3x₂ − 5x₃]
  //
  // dy/dx = [1  3  −5]
  auto y = x[0] + T(3) * x[1] - T(5) * x[2];
  auto g = slp::Gradient(y, x);

  Eigen::Matrix<T, 3, 1> expected_g{{T(1)}, {T(3)}, {T(-5)}};

  auto g_get_value = slp::value(g.get());
  CHECK(g_get_value.rows() == 3);
  CHECK(g_get_value.cols() == 1);
  CHECK(g_get_value.toDense() == expected_g);

  auto g_value = g.value();
  CHECK(g_value.rows() == 3);
  CHECK(g_value.cols() == 1);
  CHECK(g_value.toDense() == expected_g);
}
