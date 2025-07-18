// Copyright (c) Sleipnir contributors

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/multistart.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"

TEST_CASE("multistart - Mishra's Bird function", "[nonlinear_problem]") {
  struct DecisionVariables {
    double x;
    double y;
  };

  auto Solve = [](const DecisionVariables& input)
      -> slp::MultistartResult<DecisionVariables> {
    slp::Problem problem;

    auto x = problem.decision_variable();
    x.set_value(input.x);
    auto y = problem.decision_variable();
    y.set_value(input.y);

    // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    slp::Variable J = slp::sin(y) * slp::exp(slp::pow(1 - slp::cos(x), 2)) +
                      slp::cos(x) * slp::exp(slp::pow(1 - slp::sin(y), 2)) +
                      slp::pow(x - y, 2);
    problem.minimize(J);

    problem.subject_to(slp::pow(x + 5, 2) + slp::pow(y + 5, 2) < 25);

    return {problem.solve(), J.value(),
            DecisionVariables{x.value(), y.value()}};
  };

  auto [status, cost, variables] = slp::Multistart<DecisionVariables>(
      Solve,
      std::vector{DecisionVariables{-3, -8}, DecisionVariables{-3, -1.5}});

  CHECK(status == slp::ExitStatus::SUCCESS);

  CHECK(variables.x == Catch::Approx(-3.130246803458174).margin(1e-11));
  CHECK(variables.y == Catch::Approx(-1.5821421769364057).margin(1e-11));
}
