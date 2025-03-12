// Copyright (c) Sleipnir contributors

#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/multistart.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("multistart - Mishra's Bird function", "[multistart]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  struct DecisionVariables {
    T x;
    T y;
  };

  auto solve = [](const DecisionVariables& input)
      -> slp::MultistartResult<T, DecisionVariables> {
    slp::Problem<T> problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();
    x.set_value(input.x);
    y.set_value(input.y);

    // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    slp::Variable J = sin(y) * exp(pow(T(1) - cos(x), T(2))) +
                      cos(x) * exp(pow(T(1) - sin(y), T(2))) + pow(x - y, T(2));
    problem.minimize(J);

    problem.subject_to(pow(x + T(5), T(2)) + pow(y + T(5), T(2)) < T(25));

    return {problem.solve(), J.value(),
            DecisionVariables{x.value(), y.value()}};
  };

  auto [status, cost, variables] = slp::multistart<T, DecisionVariables>(
      solve, std::vector{DecisionVariables{T(-3), T(-8)},
                         DecisionVariables{T(-3), T(-1.5)}});

  CHECK(status == slp::ExitStatus::SUCCESS);

  CHECK_THAT(variables.x, WithinAbs(T(-3.13025), T(1e-15)));
  CHECK_THAT(variables.y, WithinAbs(T(-1.58214), T(1e-15)));
}
