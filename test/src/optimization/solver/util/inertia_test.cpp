// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sleipnir/optimization/solver/util/inertia.hpp>

TEST_CASE("Inertia - Formatter", "[Formatter]") {
  CHECK(fmt::format("{}", slp::Inertia{1, -2, 0}) == "(1, -2, 0)");
  CHECK(fmt::format("{:+}", slp::Inertia{1, -2, 0}) == "(+1, -2, +0)");
  CHECK(fmt::format("{: }", slp::Inertia{1, -2, 0}) == "( 1, -2,  0)");
  CHECK(fmt::format("{: >2}", slp::Inertia{1, 20, 0}) == "( 1, 20,  0)");
}
