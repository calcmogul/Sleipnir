// Copyright (c) Sleipnir contributors

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/multistart.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("multistart - Mishra's Bird function", "[nonlinear_problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  struct DecisionVariables {
    T x;
    T y;
  };

  auto Solve = [](const DecisionVariables& input)
      -> slp::MultistartResult<T, DecisionVariables> {
    slp::Problem<T> problem;

    auto x = problem.decision_variable();
    x.set_value(input.x);
    auto y = problem.decision_variable();
    y.set_value(input.y);

    // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    slp::Variable J =
        slp::sin(y) * slp::exp(slp::pow(T(1) - slp::cos(x), T(2))) +
        slp::cos(x) * slp::exp(slp::pow(T(1) - slp::sin(y), T(2))) +
        slp::pow(x - y, T(2));
    problem.minimize(J);

    problem.subject_to(slp::pow(x + T(5), T(2)) + slp::pow(y + T(5), T(2)) <
                       T(25));

    return {problem.solve(), J.value(),
            DecisionVariables{x.value(), y.value()}};
  };

  auto [status, cost, variables] = slp::Multistart<T, DecisionVariables>(
      Solve, std::vector{DecisionVariables{T(-3), T(-8)},
                         DecisionVariables{T(-3), T(-1.5)}});

  CHECK(status == slp::ExitStatus::SUCCESS);

  CHECK(variables.x == Catch::Approx(T(-3.130246803458174)).margin(T(1e-15)));
  CHECK(variables.y == Catch::Approx(T(-1.5821421769364057)).margin(T(1e-15)));
}
