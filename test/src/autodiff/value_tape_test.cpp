// Copyright (c) Sleipnir contributors

#include <array>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/value_tape.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/util/pool.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("ValueTape - Shared nodes and mixed roots", "[ValueTape]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  slp::Variable<T> y;
  x.set_value(T(2));
  y.set_value(T(3));

  // a is shared by b and c, and a and b are also roots that c depends on
  slp::Variable a = x * y;
  slp::Variable b = a + x * x;
  slp::Variable c = a * b - y;
  slp::Variable<T> k{T(5)};

  // Roots include a duplicate, a leaf, and a constant
  std::array roots{c, a, b, c, x, k};
  slp::detail::ValueTape<T> tape{roots};

  auto expected = [](T x, T y) {
    T a = x * y;
    T b = a + x * x;
    return std::array{a * b - y, a, b, a * b - y, x, T(5)};
  };

  auto check = [&](T x_value, T y_value) {
    const auto& values = tape.value();
    auto expected_values = expected(x_value, y_value);
    for (size_t i = 0; i < roots.size(); ++i) {
      CHECK(values[i] == expected_values[i]);
    }
  };

  check(T(2), T(3));

  x.set_value(T(-1));
  y.set_value(T(4));
  check(T(-1), T(4));

  // The tape must restore each node's scratch, which Gradient relies on
  CHECK(slp::Gradient(c, x).value().coeff(0) ==
        y.value() * b.value() + a.value() * (y.value() + T(2) * x.value()));
}

TEMPLATE_TEST_CASE("ValueTape - Variable::value() after copies", "[ValueTape]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::Variable<T> x;
  x.set_value(T(2));

  slp::Variable f = slp::if_else(
      +[](T a, T b) { return a < b; }, x, slp::Variable<T>{T(0)}, -x, x * x);

  // The copy shares f's tape once f has built it
  CHECK(f.value() == T(4));
  slp::Variable g = f;

  x.set_value(T(-3));
  CHECK(f.value() == T(3));
  CHECK(g.value() == T(3));

  // Assigning a constant replaces the expression and drops the tape
  g = T(7);
  CHECK(g.value() == T(7));
  CHECK(f.value() == T(3));
}
