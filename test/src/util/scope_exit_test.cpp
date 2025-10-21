// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/util/scope_exit.hpp>

TEST_CASE("scope_exit - Scope exit", "[scope_exit]") {
  int exit_count = 0;

  {
    slp::scope_exit exit{[&] { ++exit_count; }};

    CHECK(exit_count == 0);
  }

  CHECK(exit_count == 1);
}

TEST_CASE("scope_exit - Release", "[scope_exit]") {
  int exit_count = 0;

  {
    slp::scope_exit exit1{[&] { ++exit_count; }};
    slp::scope_exit exit2 = std::move(exit1);
    slp::scope_exit exit3 =
        std::move(exit1);  // NOLINT (clang-analyzer-cplusplus.Move)
    CHECK(exit_count == 0);
  }
  CHECK(exit_count == 1);

  {
    slp::scope_exit exit{[&] { ++exit_count; }};
    exit.release();
  }
  CHECK(exit_count == 1);
}
