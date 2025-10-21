// Copyright (c) Sleipnir contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Problem - Maximize", "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  x.set_value(T(1.0));

  auto y = problem.decision_variable();
  y.set_value(T(1.0));

  problem.maximize(T(50) * x + T(40) * y);

  problem.subject_to(x + T(1.5) * y <= T(750));
  problem.subject_to(T(2) * x + T(3) * y <= T(1500));
  problem.subject_to(T(2) * x + y <= T(1000));
  problem.subject_to(x >= T(0));
  problem.subject_to(y >= T(0));

  CHECK(problem.cost_function_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK(x.value() == Catch::Approx(T(375.0)).margin(T(1e-6)));
  CHECK(y.value() == Catch::Approx(T(250.0)).margin(T(1e-6)));
}

TEMPLATE_TEST_CASE("Problem - Free variable", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable(2);
  x[0].set_value(T(1.0));
  x[1].set_value(T(2.0));

  problem.subject_to(x[0] == T(0));

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK(x[0].value() == Catch::Approx(T(0.0)).margin(T(1e-6)));
  CHECK(x[1].value() == Catch::Approx(T(2.0)).margin(T(1e-6)));
}
