// Copyright (c) Sleipnir contributors

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/variable.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Variable - Default constructor", "[Variable]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Variable<T> a;

  CHECK(a.value() == T(0.0));
  CHECK(a.type() == slp::ExpressionType::LINEAR);
}
