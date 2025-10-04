// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

TEST_CASE("Gradient - Trivial case", "[Gradient]") {
  slp::Variable a;
  a.set_value(10);
  slp::Variable b;
  b.set_value(20);
  slp::Variable c = a;

  CHECK(slp::Gradient(a, a).value().coeff(0) == 1.0);
  CHECK(slp::Gradient(a, b).value().coeff(0) == 0.0);
  CHECK(slp::Gradient(c, a).value().coeff(0) == 1.0);
  CHECK(slp::Gradient(c, b).value().coeff(0) == 0.0);
}

TEST_CASE("Gradient - Unary plus", "[Gradient]") {
  slp::Variable a;
  a.set_value(10);
  slp::Variable c = +a;

  CHECK(c.value() == a.value());
  CHECK(slp::Gradient(c, a).value().coeff(0) == 1.0);
}

TEST_CASE("Gradient - Unary minus", "[Gradient]") {
  slp::Variable a;
  a.set_value(10);
  slp::Variable c = -a;

  CHECK(c.value() == -a.value());
  CHECK(slp::Gradient(c, a).value().coeff(0) == -1.0);
}

TEST_CASE("Gradient - Identical variables", "[Gradient]") {
  slp::Variable a;
  a.set_value(10);
  slp::Variable x = a;
  slp::Variable c = a * a + x;

  CHECK(c.value() == a.value() * a.value() + x.value());
  CHECK(slp::Gradient(c, a).value().coeff(0) ==
        2 * a.value() + slp::Gradient(x, a).value().coeff(0));
  CHECK(slp::Gradient(c, x).value().coeff(0) ==
        2 * a.value() * slp::Gradient(a, x).value().coeff(0) + 1);
}

TEST_CASE("Gradient - Elementary", "[Gradient]") {
  slp::Variable a;
  a.set_value(1.0);
  slp::Variable b;
  b.set_value(2.0);

  auto c = -2 * a;
  CHECK(slp::Gradient(c, a).value().coeff(0) == -2.0);

  c = a / 3.0;
  CHECK(slp::Gradient(c, a).value().coeff(0) == 1.0 / 3.0);

  a.set_value(100.0);
  b.set_value(200.0);

  c = a + b;
  CHECK(slp::Gradient(c, a).value().coeff(0) == 1.0);
  CHECK(slp::Gradient(c, b).value().coeff(0) == 1.0);

  c = a - b;
  CHECK(slp::Gradient(c, a).value().coeff(0) == 1.0);
  CHECK(slp::Gradient(c, b).value().coeff(0) == -1.0);

  c = -a + b;
  CHECK(slp::Gradient(c, a).value().coeff(0) == -1.0);
  CHECK(slp::Gradient(c, b).value().coeff(0) == 1.0);

  c = a + 1;
  CHECK(slp::Gradient(c, a).value().coeff(0) == 1.0);
}

TEST_CASE("Gradient - Comparison", "[Gradient]") {
  slp::Variable x;
  x.set_value(10.0);
  slp::Variable a;
  a.set_value(10.0);
  slp::Variable b;
  b.set_value(200.0);

  CHECK(a.value() == a.value());
  CHECK(a.value() == x.value());
  CHECK(a.value() == 10.0);
  CHECK(10.0 == a.value());

  CHECK(a.value() != b.value());
  CHECK(a.value() != 20.0);
  CHECK(20.0 != a.value());

  CHECK(a.value() < b.value());
  CHECK(a.value() < 20.0);

  CHECK(b.value() > a.value());
  CHECK(20.0 > a.value());

  CHECK(a.value() <= a.value());
  CHECK(a.value() <= x.value());
  CHECK(a.value() <= b.value());
  CHECK(a.value() <= 10.0);
  CHECK(a.value() <= 20.0);

  CHECK(a.value() >= a.value());
  CHECK(x.value() >= a.value());
  CHECK(b.value() >= a.value());
  CHECK(10.0 >= a.value());
  CHECK(20.0 >= a.value());

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

TEST_CASE("Gradient - Trigonometry", "[Gradient]") {
  slp::Variable x;
  x.set_value(0.5);

  // std::sin(x)
  CHECK(slp::sin(x).value() == std::sin(x.value()));

  auto g = slp::Gradient(slp::sin(x), x);
  CHECK(g.get().value().coeff(0) == std::cos(x.value()));
  CHECK(g.value().coeff(0) == std::cos(x.value()));

  // std::cos(x)
  CHECK(slp::cos(x).value() == std::cos(x.value()));

  g = slp::Gradient(slp::cos(x), x);
  CHECK(g.get().value().coeff(0) == -std::sin(x.value()));
  CHECK(g.value().coeff(0) == -std::sin(x.value()));

  // std::tan(x)
  CHECK(slp::tan(x).value() == std::tan(x.value()));

  g = slp::Gradient(slp::tan(x), x);
  CHECK(g.get().value().coeff(0) ==
        1.0 / (std::cos(x.value()) * std::cos(x.value())));
  CHECK(g.value().coeff(0) ==
        1.0 / (std::cos(x.value()) * std::cos(x.value())));

  // std::asin(x)
  CHECK(slp::asin(x).value() == std::asin(x.value()));

  g = slp::Gradient(slp::asin(x), x);
  CHECK(g.get().value().coeff(0) == 1.0 / std::sqrt(1 - x.value() * x.value()));
  CHECK(g.value().coeff(0) == 1.0 / std::sqrt(1 - x.value() * x.value()));

  // std::acos(x)
  CHECK(slp::acos(x).value() == std::acos(x.value()));

  g = slp::Gradient(slp::acos(x), x);
  CHECK(g.get().value().coeff(0) ==
        -1.0 / std::sqrt(1 - x.value() * x.value()));
  CHECK(g.value().coeff(0) == -1.0 / std::sqrt(1 - x.value() * x.value()));

  // std::atan(x)
  CHECK(slp::atan(x).value() == std::atan(x.value()));

  g = slp::Gradient(slp::atan(x), x);
  CHECK(g.get().value().coeff(0) == 1.0 / (1 + x.value() * x.value()));
  CHECK(g.value().coeff(0) == 1.0 / (1 + x.value() * x.value()));
}

TEST_CASE("Gradient - Hyperbolic", "[Gradient]") {
  slp::Variable x;
  x.set_value(1.0);

  // sinh(x)
  CHECK(slp::sinh(x).value() == std::sinh(x.value()));

  auto g = slp::Gradient(slp::sinh(x), x);
  CHECK(g.get().value().coeff(0) == std::cosh(x.value()));
  CHECK(g.value().coeff(0) == std::cosh(x.value()));

  // std::cosh(x)
  CHECK(slp::cosh(x).value() == std::cosh(x.value()));

  g = slp::Gradient(slp::cosh(x), x);
  CHECK(g.get().value().coeff(0) == std::sinh(x.value()));
  CHECK(g.value().coeff(0) == std::sinh(x.value()));

  // tanh(x)
  CHECK(slp::tanh(x).value() == std::tanh(x.value()));

  g = slp::Gradient(slp::tanh(x), x);
  CHECK(g.get().value().coeff(0) ==
        1.0 / (std::cosh(x.value()) * std::cosh(x.value())));
  CHECK(g.value().coeff(0) ==
        1.0 / (std::cosh(x.value()) * std::cosh(x.value())));
}

TEST_CASE("Gradient - Exponential", "[Gradient]") {
  slp::Variable x;
  x.set_value(1.0);

  // std::log(x)
  CHECK(slp::log(x).value() == std::log(x.value()));

  auto g = slp::Gradient(slp::log(x), x);
  CHECK(g.get().value().coeff(0) == 1.0 / x.value());
  CHECK(g.value().coeff(0) == 1.0 / x.value());

  // std::log10(x)
  CHECK(slp::log10(x).value() == std::log10(x.value()));

  g = slp::Gradient(slp::log10(x), x);
  CHECK(g.get().value().coeff(0) == 1.0 / (std::log(10.0) * x.value()));
  CHECK(g.value().coeff(0) == 1.0 / (std::log(10.0) * x.value()));

  // std::exp(x)
  CHECK(slp::exp(x).value() == std::exp(x.value()));

  g = slp::Gradient(slp::exp(x), x);
  CHECK(g.get().value().coeff(0) == std::exp(x.value()));
  CHECK(g.value().coeff(0) == std::exp(x.value()));
}

TEST_CASE("Gradient - Power", "[Gradient]") {
  slp::Variable x;
  x.set_value(1.0);
  slp::Variable a;
  a.set_value(2.0);
  slp::Variable y = 2 * a;

  // std::sqrt(x)
  CHECK(slp::sqrt(x).value() == std::sqrt(x.value()));

  auto g = slp::Gradient(slp::sqrt(x), x);
  CHECK(g.get().value().coeff(0) == 0.5 / std::sqrt(x.value()));
  CHECK(g.value().coeff(0) == 0.5 / std::sqrt(x.value()));

  // std::sqrt(a)
  CHECK(slp::sqrt(a).value() == std::sqrt(a.value()));

  g = slp::Gradient(slp::sqrt(a), a);
  CHECK(g.get().value().coeff(0) == 0.5 / std::sqrt(a.value()));
  CHECK(g.value().coeff(0) == 0.5 / std::sqrt(a.value()));

  // std::cbrt(x)
  CHECK(slp::cbrt(x).value() == std::cbrt(x.value()));

  g = slp::Gradient(slp::cbrt(x), x);
  CHECK(g.get().value().coeff(0) ==
        1.0 / (3.0 * std::cbrt(x.value()) * std::cbrt(x.value())));
  CHECK(g.value().coeff(0) ==
        1.0 / (3.0 * std::cbrt(x.value()) * std::cbrt(x.value())));

  // std::cbrt(a)
  CHECK(slp::cbrt(a).value() == std::cbrt(a.value()));

  g = slp::Gradient(slp::cbrt(a), a);
  CHECK(g.get().value().coeff(0) ==
        1.0 / (3.0 * std::cbrt(a.value()) * std::cbrt(a.value())));
  CHECK(g.value().coeff(0) ==
        1.0 / (3.0 * std::cbrt(a.value()) * std::cbrt(a.value())));

  // x²
  CHECK(slp::pow(x, 2.0).value() == std::pow(x.value(), 2.0));

  g = slp::Gradient(slp::pow(x, 2.0), x);
  CHECK(g.get().value().coeff(0) == 2.0 * x.value());
  CHECK(g.value().coeff(0) == 2.0 * x.value());

  // 2ˣ
  CHECK(slp::pow(2.0, x).value() == std::pow(2.0, x.value()));

  g = slp::Gradient(slp::pow(2.0, x), x);
  CHECK(g.get().value().coeff(0) == std::log(2.0) * std::pow(2.0, x.value()));
  CHECK(g.value().coeff(0) == std::log(2.0) * std::pow(2.0, x.value()));

  // xˣ
  CHECK(slp::pow(x, x).value() == std::pow(x.value(), x.value()));

  g = slp::Gradient(slp::pow(x, x), x);
  CHECK(g.get().value().coeff(0) ==
        (std::log(x.value()) + 1) * std::pow(x.value(), x.value()));
  CHECK(g.value().coeff(0) ==
        (std::log(x.value()) + 1) * std::pow(x.value(), x.value()));

  // y(a)
  CHECK(y.value() == 2 * a.value());

  g = slp::Gradient(y, a);
  CHECK(g.get().value().coeff(0) == 2.0);
  CHECK(g.value().coeff(0) == 2.0);

  // xʸ(x)
  CHECK(slp::pow(x, y).value() == std::pow(x.value(), y.value()));

  g = slp::Gradient(slp::pow(x, y), x);
  CHECK(g.get().value().coeff(0) ==
        y.value() / x.value() * std::pow(x.value(), y.value()));
  CHECK(g.value().coeff(0) ==
        y.value() / x.value() * std::pow(x.value(), y.value()));

  // xʸ(a)
  CHECK(slp::pow(x, y).value() == std::pow(x.value(), y.value()));

  g = slp::Gradient(slp::pow(x, y), a);
  CHECK(g.get().value().coeff(0) ==
        std::pow(x.value(), y.value()) *
            (y.value() / x.value() * slp::Gradient(x, a).value().coeff(0) +
             std::log(x.value()) * slp::Gradient(y, a).value().coeff(0)));
  CHECK(g.value().coeff(0) ==
        std::pow(x.value(), y.value()) *
            (y.value() / x.value() * slp::Gradient(x, a).value().coeff(0) +
             std::log(x.value()) * slp::Gradient(y, a).value().coeff(0)));

  // xʸ(y)
  CHECK(slp::pow(x, y).value() == std::pow(x.value(), y.value()));

  g = slp::Gradient(slp::pow(x, y), y);
  CHECK(g.get().value().coeff(0) ==
        std::log(x.value()) * std::pow(x.value(), y.value()));
  CHECK(g.value().coeff(0) ==
        std::log(x.value()) * std::pow(x.value(), y.value()));
}

TEST_CASE("Gradient - std::abs()", "[Gradient]") {
  slp::Variable x;
  auto g = slp::Gradient(slp::abs(x), x);

  x.set_value(1.0);
  CHECK(slp::abs(x).value() == std::abs(x.value()));
  CHECK(g.get().value().coeff(0) == 1.0);
  CHECK(g.value().coeff(0) == 1.0);

  x.set_value(-1.0);
  CHECK(slp::abs(x).value() == std::abs(x.value()));
  CHECK(g.get().value().coeff(0) == -1.0);
  CHECK(g.value().coeff(0) == -1.0);

  x.set_value(0.0);
  CHECK(slp::abs(x).value() == std::abs(x.value()));
  CHECK(g.get().value().coeff(0) == 0.0);
  CHECK(g.value().coeff(0) == 0.0);
}

TEST_CASE("Gradient - std::atan2()", "[Gradient]") {
  slp::Variable x;
  slp::Variable y;

  // Testing atan2 function on (double, var)
  x.set_value(1.0);
  y.set_value(0.9);
  CHECK(slp::atan2(2.0, x).value() == std::atan2(2.0, x.value()));

  auto g = slp::Gradient(slp::atan2(2.0, x), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(-2.0 / (2 * 2 + x.value() * x.value())).margin(1e-15));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(-2.0 / (2 * 2 + x.value() * x.value())).margin(1e-15));

  // Testing atan2 function on (var, double)
  x.set_value(1.0);
  y.set_value(0.9);
  CHECK(slp::atan2(x, 2.0).value() == std::atan2(x.value(), 2.0));

  g = slp::Gradient(slp::atan2(x, 2.0), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(2.0 / (2 * 2 + x.value() * x.value())).margin(1e-15));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(2.0 / (2 * 2 + x.value() * x.value())).margin(1e-15));

  // Testing atan2 function on (var, var)
  x.set_value(1.1);
  y.set_value(0.9);
  CHECK(slp::atan2(y, x).value() == std::atan2(y.value(), x.value()));

  g = slp::Gradient(slp::atan2(y, x), y);
  CHECK(
      g.get().value().coeff(0) ==
      Catch::Approx(x.value() / (x.value() * x.value() + y.value() * y.value()))
          .margin(1e-15));
  CHECK(g.value().coeff(0) == Catch::Approx(x.value() / (x.value() * x.value() +
                                                         y.value() * y.value()))
                                  .margin(1e-15));

  g = slp::Gradient(slp::atan2(y, x), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(-y.value() /
                      (x.value() * x.value() + y.value() * y.value()))
            .margin(1e-15));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(-y.value() /
                      (x.value() * x.value() + y.value() * y.value()))
            .margin(1e-15));

  // Testing atan2 function on (expr, expr)
  CHECK(3 * slp::atan2(slp::sin(y), 2 * x + 1).value() ==
        3 * std::atan2(std::sin(y.value()), 2 * x.value() + 1));

  g = slp::Gradient(3 * slp::atan2(slp::sin(y), 2 * x + 1), y);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(3 * (2 * x.value() + 1) * std::cos(y.value()) /
                      ((2 * x.value() + 1) * (2 * x.value() + 1) +
                       std::sin(y.value()) * std::sin(y.value())))
            .margin(1e-15));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(3 * (2 * x.value() + 1) * std::cos(y.value()) /
                      ((2 * x.value() + 1) * (2 * x.value() + 1) +
                       std::sin(y.value()) * std::sin(y.value())))
            .margin(1e-15));

  g = slp::Gradient(3 * slp::atan2(slp::sin(y), 2 * x + 1), x);
  CHECK(g.get().value().coeff(0) ==
        Catch::Approx(3 * -2 * std::sin(y.value()) /
                      ((2 * x.value() + 1) * (2 * x.value() + 1) +
                       std::sin(y.value()) * std::sin(y.value())))
            .margin(1e-15));
  CHECK(g.value().coeff(0) ==
        Catch::Approx(3 * -2 * std::sin(y.value()) /
                      ((2 * x.value() + 1) * (2 * x.value() + 1) +
                       std::sin(y.value()) * std::sin(y.value())))
            .margin(1e-15));
}

TEST_CASE("Gradient - std::hypot()", "[Gradient]") {
  slp::Variable x;
  slp::Variable y;

  // Testing hypot function on (var, double)
  x.set_value(1.8);
  y.set_value(1.5);
  CHECK(slp::hypot(x, 2.0).value() == std::hypot(x.value(), 2.0));

  auto g = slp::Gradient(slp::hypot(x, 2.0), x);
  CHECK(g.get().value().coeff(0) == x.value() / std::hypot(x.value(), 2.0));
  CHECK(g.value().coeff(0) == x.value() / std::hypot(x.value(), 2.0));

  // Testing hypot function on (double, var)
  CHECK(slp::hypot(2.0, y).value() == std::hypot(2.0, y.value()));

  g = slp::Gradient(slp::hypot(2.0, y), y);
  CHECK(g.get().value().coeff(0) == y.value() / std::hypot(2.0, y.value()));
  CHECK(g.value().coeff(0) == y.value() / std::hypot(2.0, y.value()));

  // Testing hypot function on (var, var)
  x.set_value(1.3);
  y.set_value(2.3);
  CHECK(slp::hypot(x, y).value() == std::hypot(x.value(), y.value()));

  g = slp::Gradient(slp::hypot(x, y), x);
  CHECK(g.get().value().coeff(0) ==
        x.value() / std::hypot(x.value(), y.value()));
  CHECK(g.value().coeff(0) == x.value() / std::hypot(x.value(), y.value()));

  g = slp::Gradient(slp::hypot(x, y), y);
  CHECK(g.get().value().coeff(0) ==
        y.value() / std::hypot(x.value(), y.value()));
  CHECK(g.value().coeff(0) == y.value() / std::hypot(x.value(), y.value()));

  // Testing hypot function on (expr, expr)
  x.set_value(1.3);
  y.set_value(2.3);
  CHECK(slp::hypot(2.0 * x, 3.0 * y).value() ==
        std::hypot(2.0 * x.value(), 3.0 * y.value()));

  g = slp::Gradient(slp::hypot(2.0 * x, 3.0 * y), x);
  CHECK(g.get().value().coeff(0) ==
        4.0 * x.value() / std::hypot(2.0 * x.value(), 3.0 * y.value()));
  CHECK(g.value().coeff(0) ==
        4.0 * x.value() / std::hypot(2.0 * x.value(), 3.0 * y.value()));

  g = slp::Gradient(slp::hypot(2.0 * x, 3.0 * y), y);
  CHECK(g.get().value().coeff(0) ==
        9.0 * y.value() / std::hypot(2.0 * x.value(), 3.0 * y.value()));
  CHECK(g.value().coeff(0) ==
        9.0 * y.value() / std::hypot(2.0 * x.value(), 3.0 * y.value()));

  // Testing hypot function on (var, var, var)
  slp::Variable z;
  x.set_value(1.3);
  y.set_value(2.3);
  z.set_value(3.3);
  CHECK(slp::hypot(x, y, z).value() ==
        std::hypot(x.value(), y.value(), z.value()));

  g = slp::Gradient(slp::hypot(x, y, z), x);
  CHECK(g.get().value().coeff(0) ==
        x.value() / std::hypot(x.value(), y.value(), z.value()));
  CHECK(g.value().coeff(0) ==
        x.value() / std::hypot(x.value(), y.value(), z.value()));

  g = slp::Gradient(slp::hypot(x, y, z), y);
  CHECK(g.get().value().coeff(0) ==
        y.value() / std::hypot(x.value(), y.value(), z.value()));
  CHECK(g.value().coeff(0) ==
        y.value() / std::hypot(x.value(), y.value(), z.value()));

  g = slp::Gradient(slp::hypot(x, y, z), z);
  CHECK(g.get().value().coeff(0) ==
        z.value() / std::hypot(x.value(), y.value(), z.value()));
  CHECK(g.value().coeff(0) ==
        z.value() / std::hypot(x.value(), y.value(), z.value()));
}

TEST_CASE("Gradient - Miscellaneous", "[Gradient]") {
  slp::Variable x;

  // dx/dx
  x.set_value(3.0);
  CHECK(slp::abs(x).value() == std::abs(x.value()));

  auto g = slp::Gradient(x, x);
  CHECK(g.get().value().coeff(0) == 1.0);
  CHECK(g.value().coeff(0) == 1.0);

  // std::erf(x)
  x.set_value(0.5);
  CHECK(slp::erf(x).value() == std::erf(x.value()));

  g = slp::Gradient(slp::erf(x), x);
  CHECK(g.get().value().coeff(0) ==
        2.0 * std::numbers::inv_sqrtpi * std::exp(-x.value() * x.value()));
  CHECK(g.value().coeff(0) ==
        2.0 * std::numbers::inv_sqrtpi * std::exp(-x.value() * x.value()));
}

TEST_CASE("Gradient - Variable reuse", "[Gradient]") {
  slp::Variable a;
  a.set_value(10);

  slp::Variable b;
  b.set_value(20);

  slp::Variable x = a * b;

  auto g = slp::Gradient(x, a);

  CHECK(g.get().value().coeff(0) == 20.0);
  CHECK(g.value().coeff(0) == 20.0);

  b.set_value(10);
  CHECK(g.get().value().coeff(0) == 10.0);
  CHECK(g.value().coeff(0) == 10.0);
}

TEST_CASE("Gradient - sign()", "[Gradient]") {
  auto sign = [](double x) {
    if (x < 0.0) {
      return -1.0;
    } else if (x == 0.0) {
      return 0.0;
    } else {
      return 1.0;
    }
  };

  slp::Variable x;

  // sign(1.0)
  x.set_value(1.0);
  CHECK(slp::sign(x).value() == sign(x.value()));

  auto g = slp::Gradient(slp::sign(x), x);
  CHECK(g.get().value().coeff(0) == 0.0);
  CHECK(g.value().coeff(0) == 0.0);

  // sign(-1.0)
  x.set_value(-1.0);
  CHECK(slp::sign(x).value() == sign(x.value()));

  g = slp::Gradient(slp::sign(x), x);
  CHECK(g.get().value().coeff(0) == 0.0);
  CHECK(g.value().coeff(0) == 0.0);

  // sign(0.0)
  x.set_value(0.0);
  CHECK(slp::sign(x).value() == sign(x.value()));

  g = slp::Gradient(slp::sign(x), x);
  CHECK(g.get().value().coeff(0) == 0.0);
  CHECK(g.value().coeff(0) == 0.0);
}

TEST_CASE("Gradient - Non-scalar", "[Gradient]") {
  slp::VariableMatrix x{3};
  x[0].set_value(1);
  x[1].set_value(2);
  x[2].set_value(3);

  // y = [x₁ + 3x₂ − 5x₃]
  //
  // dy/dx = [1  3  −5]
  auto y = x[0] + 3 * x[1] - 5 * x[2];
  auto g = slp::Gradient(y, x);

  Eigen::MatrixXd expected_g{{1.0}, {3.0}, {-5.0}};

  auto g_get_value = g.get().value();
  CHECK(g_get_value.rows() == 3);
  CHECK(g_get_value.cols() == 1);
  CHECK(g_get_value == expected_g);

  auto g_value = g.value();
  CHECK(g_value.rows() == 3);
  CHECK(g_value.cols() == 1);
  CHECK(g_value.toDense() == expected_g);
}
