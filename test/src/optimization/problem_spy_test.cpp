// Copyright (c) Sleipnir contributors

#include <stdint.h>

#include <fstream>
#include <string>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "coord.hpp"
#include "scalar_types_under_test.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"

std::ifstream open(std::string_view filename) {
  std::ifstream file{std::string{filename}, std::ios::binary};
  if (!file.is_open()) {
    FAIL(fmt::format("Could not open {}", filename));
  }
  return file;
}

char read_char(std::ifstream& spy) {
  char value;
  spy.read(&value, 1);
  return value;
}

int32_t read_i32(std::ifstream& spy) {
  int32_t value;
  spy.read(reinterpret_cast<char*>(&value), 4);
  return value;
}

std::string read_str(std::ifstream& spy) {
  int32_t size = read_i32(spy);
  std::string value(size, '\0');
  spy.read(&value[0], size);
  return value;
}

Coord read_coord(std::ifstream& spy) {
  return Coord{read_i32(spy), read_i32(spy), read_char(spy)};
}

TEMPLATE_TEST_CASE("Problem - Spy", "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

#ifdef SLEIPNIR_DISABLE_DIAGNOSTICS
  SKIP("Spy is disabled");
#endif

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();
  x.set_value(T(20));
  y.set_value(T(20));

  problem.minimize(pow(x, T(4)) + pow(y, T(4)));

  problem.subject_to(x >= T(1));
  problem.subject_to(x <= T(10));
  problem.subject_to(y == T(2));

  int iterations = 0;
  problem.add_callback([&](const slp::IterationInfo<T>&) { ++iterations; });

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}, true) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(x.value(), WithinAbs(T(1), T(1e-8)));
  CHECK_THAT(y.value(), WithinAbs(T(2), T(1e-8)));

  // Check H.spy
  {
    auto spy = open("H.spy");

    REQUIRE(read_str(spy) == "Hessian");             // Title
    REQUIRE(read_str(spy) == "Decision variables");  // Row label
    REQUIRE(read_str(spy) == "Decision variables");  // Col label
    REQUIRE(read_i32(spy) == 2);                     // Rows
    REQUIRE(read_i32(spy) == 2);                     // Cols

    // Coords
    for (int i = 0; i < iterations; ++i) {
      REQUIRE(read_i32(spy) == 2);  // Num coords
      CHECK(read_coord(spy) == Coord{0, 0, '+'});
      CHECK(read_coord(spy) == Coord{1, 1, '+'});
      if (spy.eof()) {
        FAIL("Reached end of file prematurely");
      }
    }
  }

  // Check A_e.spy
  {
    auto spy = open("A_e.spy");

    REQUIRE(read_str(spy) == "Equality constraint Jacobian");  // Title
    REQUIRE(read_str(spy) == "Constraints");                   // Row label
    REQUIRE(read_str(spy) == "Decision variables");            // Col label
    REQUIRE(read_i32(spy) == 1);                               // Rows
    REQUIRE(read_i32(spy) == 2);                               // Cols

    // Coords
    for (int i = 0; i < iterations; ++i) {
      REQUIRE(read_i32(spy) == 1);  // Num coords
      CHECK(read_coord(spy) == Coord{0, 1, '+'});
      if (spy.eof()) {
        FAIL("Reached end of file prematurely");
      }
    }
  }

  // Check A_i.spy
  {
    auto spy = open("A_i.spy");

    REQUIRE(read_str(spy) == "Inequality constraint Jacobian");  // Title
    REQUIRE(read_str(spy) == "Constraints");                     // Row label
    REQUIRE(read_str(spy) == "Decision variables");              // Col label
    REQUIRE(read_i32(spy) == 2);                                 // Rows
    REQUIRE(read_i32(spy) == 2);                                 // Cols

    // Coords
    for (int i = 0; i < iterations; ++i) {
      REQUIRE(read_i32(spy) == 2);  // Num coords
      CHECK(read_coord(spy) == Coord{0, 0, '+'});
      CHECK(read_coord(spy) == Coord{1, 0, '-'});
      if (spy.eof()) {
        FAIL("Reached end of file prematurely");
      }
    }
  }
}
