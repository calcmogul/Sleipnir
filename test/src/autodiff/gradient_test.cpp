// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Gradient - Trivial case", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Variable<T> a;
  a.set_value(T(10));
  slp::Variable<T> b;
  b.set_value(T(20));
  slp::Variable c = a;

  CHECK(slp::Gradient(a, a).value().coeff(0) == T(1.0));
  CHECK(slp::Gradient(a, b).value().coeff(0) == T(0.0));
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1.0));
  CHECK(slp::Gradient(c, b).value().coeff(0) == T(0.0));
}

TEMPLATE_TEST_CASE("Gradient - Unary plus", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Variable<T> a;
  a.set_value(T(10));
  slp::Variable c = +a;

  CHECK(c.value() == a.value());
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1.0));
}

TEMPLATE_TEST_CASE("Gradient - Unary minus", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Variable<T> a;
  a.set_value(T(10));
  slp::Variable c = -a;

  CHECK(c.value() == -a.value());
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(-1.0));
}

TEMPLATE_TEST_CASE("Gradient - Identical variables", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Variable<T> a;
  a.set_value(T(10));
  slp::Variable x = a;
  slp::Variable c = a * a + x;

  CHECK(c.value() == a.value() * a.value() + x.value());
  CHECK(slp::Gradient(c, a).value().coeff(0) ==
        2 * a.value() + slp::Gradient(x, a).value().coeff(0));
  CHECK(slp::Gradient(c, x).value().coeff(0) ==
        2 * a.value() * slp::Gradient(a, x).value().coeff(0) + 1);
}

TEMPLATE_TEST_CASE("Gradient - Elementary", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Variable<T> a;
  a.set_value(T(1.0));
  slp::Variable<T> b;
  b.set_value(T(2.0));

  auto c = -2 * a;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(-2.0));

  c = a / 3.0;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1.0 / 3.0));

  a.set_value(T(100.0));
  b.set_value(T(200.0));

  c = a + b;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1.0));
  CHECK(slp::Gradient(c, b).value().coeff(0) == T(1.0));

  c = a - b;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1.0));
  CHECK(slp::Gradient(c, b).value().coeff(0) == T(-1.0));

  c = -a + b;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(-1.0));
  CHECK(slp::Gradient(c, b).value().coeff(0) == T(1.0));

  c = a + 1;
  CHECK(slp::Gradient(c, a).value().coeff(0) == T(1.0));
}

TEMPLATE_TEST_CASE("Gradient - Comparison", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Variable<T> x;
  x.set_value(T(10.0));
  slp::Variable<T> a;
  a.set_value(T(10.0));
  slp::Variable<T> b;
  b.set_value(T(200.0));

  CHECK(a.value() == a.value());
  CHECK(a.value() == x.value());
  CHECK(a.value() == T(10.0));
  CHECK(T(10.0) == a.value());

  CHECK(a.value() != b.value());
  CHECK(a.value() != T(20.0));
  CHECK(T(20.0) != a.value());

  CHECK(a.value() < b.value());
  CHECK(a.value() < T(20.0));

  CHECK(b.value() > a.value());
  CHECK(T(20.0) > a.value());

  CHECK(a.value() <= a.value());
  CHECK(a.value() <= x.value());
  CHECK(a.value() <= b.value());
  CHECK(a.value() <= T(10.0));
  CHECK(a.value() <= T(20.0));

  CHECK(a.value() >= a.value());
  CHECK(x.value() >= a.value());
  CHECK(b.value() >= a.value());
  CHECK(T(10.0) >= a.value());
  CHECK(T(20.0) >= a.value());

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

  slp::Variable<T> x;
  x.set_value(T(0.5));

  // std::sin(x)
  CHECK(slp::sin(x).value() == sin(x.value()));  // NOLINT

  auto g = slp::Gradient(slp::sin(x), x);
  CHECK(g.get().value().coeff(0) == cos(x.value()));  // NOLINT
  CHECK(g.value().coeff(0) == cos(x.value()));        // NOLINT

  // std::cos(x)
  CHECK(slp::cos(x).value() == cos(x.value()));  // NOLINT

  g = slp::Gradient(slp::cos(x), x);
  CHECK(g.get().value().coeff(0) == -sin(x.value()));  // NOLINT
  CHECK(g.value().coeff(0) == -sin(x.value()));        // NOLINT

  // std::tan(x)
  CHECK(slp::tan(x).value() == tan(x.value()));  // NOLINT

  g = slp::Gradient(slp::tan(x), x);
  CHECK(g.get().value().coeff(0) ==
        T(1.0) / (cos(x.value()) * cos(x.value())));  // NOLINT
  CHECK(g.value().coeff(0) ==
        T(1.0) / (cos(x.value()) * cos(x.value())));  // NOLINT

  // std::asin(x)
  CHECK(slp::asin(x).value() == asin(x.value()));  // NOLINT

  g = slp::Gradient(slp::asin(x), x);
  CHECK(g.get().value().coeff(0) ==
        T(1.0) / sqrt(T(1) - x.value() * x.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        T(1.0) / sqrt(T(1) - x.value() * x.value()));  // NOLINT

  // std::acos(x)
  CHECK(slp::acos(x).value() == acos(x.value()));  // NOLINT

  g = slp::Gradient(slp::acos(x), x);
  CHECK(g.get().value().coeff(0) ==
        T(-1.0) / sqrt(T(1) - x.value() * x.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        T(-1.0) / sqrt(T(1) - x.value() * x.value()));  // NOLINT

  // std::atan(x)
  CHECK(slp::atan(x).value() == atan(x.value()));  // NOLINT

  g = slp::Gradient(slp::atan(x), x);
  CHECK(g.get().value().coeff(0) == T(1.0) / (T(1) + x.value() * x.value()));
  CHECK(g.value().coeff(0) == T(1.0) / (T(1) + x.value() * x.value()));
}

TEMPLATE_TEST_CASE("Gradient - Hyperbolic", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::cosh;
  using std::sinh;
  using std::tanh;

  slp::Variable<T> x;
  x.set_value(T(1.0));

  // sinh(x)
  CHECK(slp::sinh(x).value() == sinh(x.value()));  // NOLINT

  auto g = slp::Gradient(slp::sinh(x), x);
  CHECK(g.get().value().coeff(0) == cosh(x.value()));  // NOLINT
  CHECK(g.value().coeff(0) == cosh(x.value()));        // NOLINT

  // std::cosh(x)
  CHECK(slp::cosh(x).value() == cosh(x.value()));  // NOLINT

  g = slp::Gradient(slp::cosh(x), x);
  CHECK(g.get().value().coeff(0) == sinh(x.value()));  // NOLINT
  CHECK(g.value().coeff(0) == sinh(x.value()));        // NOLINT

  // tanh(x)
  CHECK(slp::tanh(x).value() == tanh(x.value()));  // NOLINT

  g = slp::Gradient(slp::tanh(x), x);
  CHECK(g.get().value().coeff(0) ==
        T(1.0) / (cosh(x.value()) * cosh(x.value())));  // NOLINT
  CHECK(g.value().coeff(0) ==
        T(1.0) / (cosh(x.value()) * cosh(x.value())));  // NOLINT
}

TEMPLATE_TEST_CASE("Gradient - Exponential", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::exp;
  using std::log;
  using std::log10;

  slp::Variable<T> x;
  x.set_value(T(1.0));

  // std::log(x)
  CHECK(slp::log(x).value() == log(x.value()));  // NOLINT

  auto g = slp::Gradient(slp::log(x), x);
  CHECK(g.get().value().coeff(0) == T(1.0) / x.value());
  CHECK(g.value().coeff(0) == T(1.0) / x.value());

  // std::log10(x)
  CHECK(slp::log10(x).value() == log10(x.value()));  // NOLINT

  g = slp::Gradient(slp::log10(x), x);
  CHECK(g.get().value().coeff(0) ==
        T(1.0) / (log(T(10.0)) * x.value()));                        // NOLINT
  CHECK(g.value().coeff(0) == T(1.0) / (log(T(10.0)) * x.value()));  // NOLINT

  // std::exp(x)
  CHECK(slp::exp(x).value() == exp(x.value()));  // NOLINT

  g = slp::Gradient(slp::exp(x), x);
  CHECK(g.get().value().coeff(0) == exp(x.value()));  // NOLINT
  CHECK(g.value().coeff(0) == exp(x.value()));        // NOLINT
}

TEMPLATE_TEST_CASE("Gradient - Power", "[Gradient]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::cbrt;
  using std::log;
  using std::pow;
  using std::sqrt;

  slp::Variable<T> x;
  x.set_value(T(1.0));
  slp::Variable<T> a;
  a.set_value(T(2.0));
  slp::Variable y = 2 * a;

  // std::sqrt(x)
  CHECK(slp::sqrt(x).value() == sqrt(x.value()));  // NOLINT

  auto g = slp::Gradient(slp::sqrt(x), x);
  CHECK(g.get().value().coeff(0) == T(0.5) / sqrt(x.value()));  // NOLINT
  CHECK(g.value().coeff(0) == T(0.5) / sqrt(x.value()));        // NOLINT

  // std::sqrt(a)
  CHECK(slp::sqrt(a).value() == sqrt(a.value()));  // NOLINT

  g = slp::Gradient(slp::sqrt(a), a);
  CHECK(g.get().value().coeff(0) == T(0.5) / sqrt(a.value()));  // NOLINT
  CHECK(g.value().coeff(0) == T(0.5) / sqrt(a.value()));        // NOLINT

  // std::cbrt(x)
  CHECK(slp::cbrt(x).value() == cbrt(x.value()));  // NOLINT

  g = slp::Gradient(slp::cbrt(x), x);
  CHECK(g.get().value().coeff(0) ==
        T(1.0) / (T(3.0) * cbrt(x.value()) * cbrt(x.value())));  // NOLINT
  CHECK(g.value().coeff(0) ==
        T(1.0) / (T(3.0) * cbrt(x.value()) * cbrt(x.value())));  // NOLINT

  // std::cbrt(a)
  CHECK(slp::cbrt(a).value() == cbrt(a.value()));  // NOLINT

  g = slp::Gradient(slp::cbrt(a), a);
  CHECK(g.get().value().coeff(0) ==
        T(1.0) / (T(3.0) * cbrt(a.value()) * cbrt(a.value())));  // NOLINT
  CHECK(g.value().coeff(0) ==
        T(1.0) / (T(3.0) * cbrt(a.value()) * cbrt(a.value())));  // NOLINT

  // x²
  CHECK(slp::pow(x, T(2.0)).value() == pow(x.value(), T(2.0)));  // NOLINT

  g = slp::Gradient(slp::pow(x, T(2.0)), x);
  CHECK(g.get().value().coeff(0) == T(2.0) * x.value());
  CHECK(g.value().coeff(0) == T(2.0) * x.value());

  // 2ˣ
  CHECK(slp::pow(T(2.0), x).value() == pow(T(2.0), x.value()));  // NOLINT

  g = slp::Gradient(slp::pow(T(2.0), x), x);
  CHECK(g.get().value().coeff(0) ==
        log(T(2.0)) * pow(T(2.0), x.value()));                        // NOLINT
  CHECK(g.value().coeff(0) == log(T(2.0)) * pow(T(2.0), x.value()));  // NOLINT

  // xˣ
  CHECK(slp::pow(x, x).value() == pow(x.value(), x.value()));  // NOLINT

  g = slp::Gradient(slp::pow(x, x), x);
  CHECK(g.get().value().coeff(0) ==
        (log(x.value()) + T(1)) * pow(x.value(), x.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        (log(x.value()) + T(1)) * pow(x.value(), x.value()));  // NOLINT

  // y(a)
  CHECK(y.value() == 2 * a.value());

  g = slp::Gradient(y, a);
  CHECK(g.get().value().coeff(0) == T(2.0));
  CHECK(g.value().coeff(0) == T(2.0));

  // xʸ(x)
  CHECK(slp::pow(x, y).value() == pow(x.value(), y.value()));  // NOLINT

  g = slp::Gradient(slp::pow(x, y), x);
  CHECK(g.get().value().coeff(0) ==
        y.value() / x.value() * pow(x.value(), y.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        y.value() / x.value() * pow(x.value(), y.value()));  // NOLINT

  // xʸ(a)
  CHECK(slp::pow(x, y).value() == pow(x.value(), y.value()));  // NOLINT

  g = slp::Gradient(slp::pow(x, y), a);
  CHECK(g.get().value().coeff(0) ==
        pow(x.value(), y.value()) *  // NOLINT
            (y.value() / x.value() * slp::Gradient(x, a).value().coeff(0) +
             log(x.value()) * slp::Gradient(y, a).value().coeff(0)));  // NOLINT
  CHECK(g.value().coeff(0) ==
        pow(x.value(), y.value()) *  // NOLINT
            (y.value() / x.value() * slp::Gradient(x, a).value().coeff(0) +
             log(x.value()) * slp::Gradient(y, a).value().coeff(0)));  // NOLINT

  // xʸ(y)
  CHECK(slp::pow(x, y).value() == pow(x.value(), y.value()));  // NOLINT

  g = slp::Gradient(slp::pow(x, y), y);
  CHECK(g.get().value().coeff(0) ==
        log(x.value()) * pow(x.value(), y.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        log(x.value()) * pow(x.value(), y.value()));  // NOLINT
}

TEMPLATE_TEST_CASE("Gradient - std::abs()", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::abs;

  slp::Variable<T> x;
  auto g = slp::Gradient(slp::abs(x), x);

  x.set_value(T(1.0));
  CHECK(slp::abs(x).value() == abs(x.value()));  // NOLINT
  CHECK(g.get().value().coeff(0) == T(1.0));
  CHECK(g.value().coeff(0) == T(1.0));

  x.set_value(T(-1.0));
  CHECK(slp::abs(x).value() == abs(x.value()));  // NOLINT
  CHECK(g.get().value().coeff(0) == T(-1.0));
  CHECK(g.value().coeff(0) == T(-1.0));

  x.set_value(T(0.0));
  CHECK(slp::abs(x).value() == abs(x.value()));  // NOLINT
  CHECK(g.get().value().coeff(0) == T(0.0));
  CHECK(g.value().coeff(0) == T(0.0));
}

TEMPLATE_TEST_CASE("Gradient - std::atan2()", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::atan2;
  using std::cos;
  using std::sin;

  slp::Variable<T> x;
  slp::Variable<T> y;

  // Testing atan2 function on (T, var)
  x.set_value(T(1.0));
  y.set_value(T(0.9));
  CHECK(slp::atan2(2.0, x).value() == atan2(T(2.0), x.value()));  // NOLINT

  auto g = slp::Gradient(slp::atan2(T(2.0), x), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(T(-2.0) / (T(2) * T(2) + x.value() * x.value()))
            .margin(T(1e-15)));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(T(-2.0) / (T(2) * T(2) + x.value() * x.value()))
            .margin(T(1e-15)));

  // Testing atan2 function on (var, T)
  x.set_value(T(1.0));
  y.set_value(T(0.9));
  CHECK(slp::atan2(x, T(2.0)).value() == atan2(x.value(), T(2.0)));  // NOLINT

  g = slp::Gradient(slp::atan2(x, T(2.0)), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(T(2.0) / (T(2) * T(2) + x.value() * x.value()))
            .margin(T(1e-15)));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(T(2.0) / (T(2) * T(2) + x.value() * x.value()))
            .margin(T(1e-15)));

  // Testing atan2 function on (var, var)
  x.set_value(T(1.1));
  y.set_value(T(0.9));
  CHECK(slp::atan2(y, x).value() == atan2(y.value(), x.value()));  // NOLINT

  g = slp::Gradient(slp::atan2(y, x), y);
  CHECK(
      g.get().value().coeff(0) ==
      Catch::Approx(x.value() / (x.value() * x.value() + y.value() * y.value()))
          .margin(T(1e-15)));
  CHECK(g.value().coeff(0) == Catch::Approx(x.value() / (x.value() * x.value() +
                                                         y.value() * y.value()))
                                  .margin(T(1e-15)));

  g = slp::Gradient(slp::atan2(y, x), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(-y.value() /
                      (x.value() * x.value() + y.value() * y.value()))
            .margin(T(1e-15)));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(-y.value() /
                      (x.value() * x.value() + y.value() * y.value()))
            .margin(T(1e-15)));

  // Testing atan2 function on (expr, expr)
  CHECK(3 * slp::atan2(slp::sin(y), T(2) * x + T(1)).value() ==
        3 * atan2(sin(y.value()), T(2) * x.value() + T(1)));  // NOLINT

  g = slp::Gradient(T(3) * slp::atan2(slp::sin(y), T(2) * x + T(1)), y);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(T(3) * (T(2) * x.value() + T(1)) *
                      cos(y.value()) /  // NOLINT
                      ((T(2) * x.value() + T(1)) * (T(2) * x.value() + T(1)) +
                       sin(y.value()) * sin(y.value())))  // NOLINT
            .margin(T(1e-15)));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(T(3) * (T(2) * x.value() + T(1)) *
                      cos(y.value()) /  // NOLINT
                      ((T(2) * x.value() + T(1)) * (T(2) * x.value() + T(1)) +
                       sin(y.value()) * sin(y.value())))  // NOLINT
            .margin(T(1e-15)));

  g = slp::Gradient(T(3) * slp::atan2(slp::sin(y), T(2) * x + T(1)), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(T(3) * -T(2) * sin(y.value()) /  // NOLINT
                      ((T(2) * x.value() + T(1)) * (T(2) * x.value() + T(1)) +
                       sin(y.value()) * sin(y.value())))  // NOLINT
            .margin(T(1e-15)));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(T(3) * T(-2) * sin(y.value()) /  // NOLINT
                      ((T(2) * x.value() + T(1)) * (T(2) * x.value() + T(1)) +
                       sin(y.value()) * sin(y.value())))  // NOLINT
            .margin(T(1e-15)));
}

TEMPLATE_TEST_CASE("Gradient - std::hypot()", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::hypot;

  slp::Variable<T> x;
  slp::Variable<T> y;

  // Testing hypot function on (var, T)
  x.set_value(T(1.8));
  y.set_value(T(1.5));
  CHECK(slp::hypot(x, 2.0).value() == hypot(x.value(), T(2.0)));  // NOLINT

  auto g = slp::Gradient(slp::hypot(x, T(2.0)), x);
  CHECK(g.get().value().coeff(0) ==
        x.value() / hypot(x.value(), T(2.0)));                        // NOLINT
  CHECK(g.value().coeff(0) == x.value() / hypot(x.value(), T(2.0)));  // NOLINT

  // Testing hypot function on (T, var)
  CHECK(slp::hypot(2.0, y).value() == hypot(T(2.0), y.value()));  // NOLINT

  g = slp::Gradient(slp::hypot(T(2.0), y), y);
  CHECK(g.get().value().coeff(0) ==
        y.value() / hypot(T(2.0), y.value()));                        // NOLINT
  CHECK(g.value().coeff(0) == y.value() / hypot(T(2.0), y.value()));  // NOLINT

  // Testing hypot function on (var, var)
  x.set_value(T(1.3));
  y.set_value(T(2.3));
  CHECK(slp::hypot(x, y).value() == hypot(x.value(), y.value()));  // NOLINT

  g = slp::Gradient(slp::hypot(x, y), x);
  CHECK(g.get().value().coeff(0) ==
        x.value() / hypot(x.value(), y.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        x.value() / hypot(x.value(), y.value()));  // NOLINT

  g = slp::Gradient(slp::hypot(x, y), y);
  CHECK(g.get().value().coeff(0) ==
        y.value() / hypot(x.value(), y.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        y.value() / hypot(x.value(), y.value()));  // NOLINT

  // Testing hypot function on (expr, expr)
  x.set_value(T(1.3));
  y.set_value(T(2.3));
  CHECK(slp::hypot(T(2.0) * x, T(3.0) * y).value() ==
        hypot(T(2.0) * x.value(), T(3.0) * y.value()));  // NOLINT

  g = slp::Gradient(slp::hypot(T(2.0) * x, T(3.0) * y), x);
  CHECK(g.get().value().coeff(0) ==
        T(4.0) * x.value() /
            hypot(T(2.0) * x.value(), T(3.0) * y.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        T(4.0) * x.value() /
            hypot(T(2.0) * x.value(), T(3.0) * y.value()));  // NOLINT

  g = slp::Gradient(slp::hypot(T(2.0) * x, T(3.0) * y), y);
  CHECK(g.get().value().coeff(0) ==
        T(9.0) * y.value() /
            hypot(T(2.0) * x.value(), T(3.0) * y.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        T(9.0) * y.value() /
            hypot(T(2.0) * x.value(), T(3.0) * y.value()));  // NOLINT

  // Testing hypot function on (var, var, var)
  slp::Variable<T> z;
  x.set_value(T(1.3));
  y.set_value(T(2.3));
  z.set_value(T(3.3));
  CHECK(slp::hypot(x, y, z).value() ==
        hypot(x.value(), y.value(), z.value()));  // NOLINT

  g = slp::Gradient(slp::hypot(x, y, z), x);
  CHECK(g.get().value().coeff(0) ==
        x.value() / hypot(x.value(), y.value(), z.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        x.value() / hypot(x.value(), y.value(), z.value()));  // NOLINT

  g = slp::Gradient(slp::hypot(x, y, z), y);
  CHECK(g.get().value().coeff(0) ==
        y.value() / hypot(x.value(), y.value(), z.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        y.value() / hypot(x.value(), y.value(), z.value()));  // NOLINT

  g = slp::Gradient(slp::hypot(x, y, z), z);
  CHECK(g.get().value().coeff(0) ==
        z.value() / hypot(x.value(), y.value(), z.value()));  // NOLINT
  CHECK(g.value().coeff(0) ==
        z.value() / hypot(x.value(), y.value(), z.value()));  // NOLINT
}

TEMPLATE_TEST_CASE("Gradient - Miscellaneous", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::abs;
  using std::exp;

  slp::Variable<T> x;

  // dx/dx
  x.set_value(T(3.0));
  CHECK(slp::abs(x).value() == abs(x.value()));  // NOLINT

  auto g = slp::Gradient(x, x);
  CHECK(g.get().value().coeff(0) == T(1.0));
  CHECK(g.value().coeff(0) == T(1.0));

  // std::erf(x)
  x.set_value(T(0.5));
  CHECK(slp::erf(x).value() == erf(x.value()));  // NOLINT

  g = slp::Gradient(slp::erf(x), x);
  CHECK(g.get().value().coeff(0) == T(2.0) * T(std::numbers::inv_sqrtpi) *
                                        exp(-x.value() * x.value()));  // NOLINT
  CHECK(g.value().coeff(0) == T(2.0) * T(std::numbers::inv_sqrtpi) *
                                  exp(-x.value() * x.value()));  // NOLINT
}

TEMPLATE_TEST_CASE("Gradient - Variable reuse", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Variable<T> a;
  a.set_value(T(10));

  slp::Variable<T> b;
  b.set_value(T(20));

  slp::Variable x = a * b;

  auto g = slp::Gradient(x, a);

  CHECK(g.get().value().coeff(0) == T(20.0));
  CHECK(g.value().coeff(0) == T(20.0));

  b.set_value(T(10));
  CHECK(g.get().value().coeff(0) == T(10.0));
  CHECK(g.value().coeff(0) == T(10.0));
}

TEMPLATE_TEST_CASE("Gradient - sign()", "[Gradient]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  auto sign = [](T x) {
    if (x < T(0.0)) {
      return T(-1.0);
    } else if (x == T(0.0)) {
      return T(0.0);
    } else {
      return T(1.0);
    }
  };

  slp::Variable<T> x;

  // sign(1.0)
  x.set_value(T(1.0));
  CHECK(slp::sign(x).value() == sign(x.value()));

  auto g = slp::Gradient(slp::sign(x), x);
  CHECK(g.get().value().coeff(0) == T(0.0));
  CHECK(g.value().coeff(0) == T(0.0));

  // sign(-1.0)
  x.set_value(T(-1.0));
  CHECK(slp::sign(x).value() == sign(x.value()));

  g = slp::Gradient(slp::sign(x), x);
  CHECK(g.get().value().coeff(0) == T(0.0));
  CHECK(g.value().coeff(0) == T(0.0));

  // sign(0.0)
  x.set_value(T(0.0));
  CHECK(slp::sign(x).value() == sign(x.value()));

  g = slp::Gradient(slp::sign(x), x);
  CHECK(g.get().value().coeff(0) == T(0.0));
  CHECK(g.value().coeff(0) == T(0.0));
}

TEMPLATE_TEST_CASE("Gradient - Non-scalar", "[Gradient]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::VariableMatrix<T> x{3};
  x[0].set_value(T(1));
  x[1].set_value(T(2));
  x[2].set_value(T(3));

  // y = [x₁ + 3x₂ − 5x₃]
  //
  // dy/dx = [1  3  −5]
  auto y = x[0] + T(3) * x[1] - T(5) * x[2];
  auto g = slp::Gradient(y, x);

  Eigen::Matrix<T, 3, 1> expected_g{{T(1.0)}, {T(3.0)}, {T(-5.0)}};

  auto g_get_value = g.get().value();
  CHECK(g_get_value.rows() == 3);
  CHECK(g_get_value.cols() == 1);
  CHECK(g_get_value == expected_g);

  auto g_value = g.value();
  CHECK(g_value.rows() == 3);
  CHECK(g_value.cols() == 1);
  CHECK(g_value.toDense() == expected_g);
}
