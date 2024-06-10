// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/Variable.hpp>

#include "CatchStringConverters.hpp"

TEST_CASE("Variable - Default constructor", "[Variable]") {
  constexpr sleipnir::Variable a;

  CHECK(a.Value() == 0.0);
  CHECK(a.Type() == sleipnir::ExpressionType::kLinear);
}
