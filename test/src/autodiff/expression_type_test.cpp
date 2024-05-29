// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sleipnir/autodiff/expression_type.hpp>

TEST_CASE("ExpressionType - Formatter", "[Formatter]") {
  CHECK(fmt::format("{}", slp::ExpressionType::NONE) == "none");
  CHECK(fmt::format("{}", slp::ExpressionType::CONSTANT) == "constant");
  CHECK(fmt::format("{}", slp::ExpressionType::LINEAR) == "linear");
  CHECK(fmt::format("{}", slp::ExpressionType::QUADRATIC) == "quadratic");
  CHECK(fmt::format("{}", slp::ExpressionType::NONLINEAR) == "nonlinear");
}
