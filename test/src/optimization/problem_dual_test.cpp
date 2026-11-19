// Copyright (c) Sleipnir contributors

#include <format>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Problem dual - Equality constraint", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();

  problem.minimize(x * x + y * y);

  auto constraint = x + y == T(2);
  problem.subject_to(constraint);

  // The dual variable is zero before the first solve
  CHECK(problem.dual(constraint) == Eigen::Vector<T, 1>{T(0)});

  CHECK(problem.solve() == slp::ExitStatus::SUCCESS);

  CHECK_THAT(x.value(), WithinAbs(T(1), solution_tolerance<T>(1)));
  CHECK_THAT(y.value(), WithinAbs(T(1), solution_tolerance<T>(1)));

  // ∇f = [2x, 2y] = [2, 2] and ∇cₑ = [1, 1], so ∇f − y∇cₑ = 0 gives y = 2
  auto dual = problem.dual(constraint);
  REQUIRE(dual.rows() == 1);
  CHECK_THAT(dual[0], WithinAbs(T(2), solution_tolerance<T>(10)));
}

TEMPLATE_TEST_CASE("Problem dual - Inequality constraints", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();

  problem.minimize((x - T(2)) * (x - T(2)));

  auto upper_bound = x <= T(1);
  auto lower_bound = x >= T(-5);
  problem.subject_to(lower_bound);
  problem.subject_to(upper_bound);

  CHECK(problem.solve() == slp::ExitStatus::SUCCESS);

  CHECK_THAT(x.value(), WithinAbs(T(1), solution_tolerance<T>(1)));

  // The upper bound is active. ∇f = 2(x − 2) = −2 and ∇cᵢ = ∇(1 − x) = −1,
  // so ∇f − z∇cᵢ = 0 gives z = 2.
  auto upper_dual = problem.dual(upper_bound);
  REQUIRE(upper_dual.rows() == 1);
  CHECK_THAT(upper_dual[0], WithinAbs(T(2), solution_tolerance<T>(10)));

  // The lower bound is inactive
  auto lower_dual = problem.dual(lower_bound);
  REQUIRE(lower_dual.rows() == 1);
  CHECK_THAT(lower_dual[0], WithinAbs(T(0), solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem dual - Matrix constraints", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable(3);

  // Minimize distance to [2, 2, 2]
  slp::Variable<T> J = T(0);
  for (int i = 0; i < 3; ++i) {
    J += (x[i] - T(2)) * (x[i] - T(2));
  }
  problem.minimize(J);

  // Constraints the queried ones are interleaved with
  problem.subject_to(x[0] >= T(-10));

  auto sum = x[1] + x[2] == T(5);
  problem.subject_to(sum);

  auto upper_bounds = x <= Eigen::Vector3<T>{T(1), T(4), T(2.25)};
  problem.subject_to(upper_bounds);

  CHECK(problem.solve() == slp::ExitStatus::SUCCESS);

  // x₀ is at its upper bound. x₁ and x₂ split the sum evenly until x₂ hits its
  // upper bound.
  CHECK_THAT(x.value(0), WithinAbs(T(1), solution_tolerance<T>(1)));
  CHECK_THAT(x.value(1), WithinAbs(T(2.75), solution_tolerance<T>(1)));
  CHECK_THAT(x.value(2), WithinAbs(T(2.25), solution_tolerance<T>(1)));

  // Stationarity for x₁ gives 2(x₁ − 2) − y = 0, so y = 1.5
  auto y = problem.dual(sum);
  REQUIRE(y.rows() == 1);
  CHECK_THAT(y[0], WithinAbs(T(1.5), solution_tolerance<T>(10)));

  // Stationarity for x₀ gives 2(x₀ − 2) + z₀ = 0, so z₀ = 2. The upper bound
  // on x₁ is inactive. Stationarity for x₂ gives 2(x₂ − 2) − y + z₂ = 0, so
  // z₂ = 1.
  auto z = problem.dual(upper_bounds);
  REQUIRE(z.rows() == 3);
  CHECK_THAT(z[0], WithinAbs(T(2), solution_tolerance<T>(10)));
  CHECK_THAT(z[1], WithinAbs(T(0), solution_tolerance<T>(1)));
  CHECK_THAT(z[2], WithinAbs(T(1), solution_tolerance<T>(10)));
}

TEMPLATE_TEST_CASE("Problem set_dual - Equality constraint", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();

  problem.minimize(x * x + y * y);

  auto constraint = x + y == T(2);
  problem.subject_to(constraint);

  problem.set_dual(constraint, Eigen::Vector<T, 1>{T(2)});
  CHECK(problem.dual(constraint) == Eigen::Vector<T, 1>{T(2)});

  int iterations = 0;
  problem.add_callback([&](const slp::IterationInfo<T>&) { ++iterations; });

  // The primal-dual solution satisfies the KKT conditions, so the solver
  // starts there and stops immediately
  x.set_value(T(1));
  y.set_value(T(1));
  CHECK(problem.solve() == slp::ExitStatus::SUCCESS);
  CHECK(iterations == 0);

  CHECK_THAT(x.value(), WithinAbs(T(1), solution_tolerance<T>(1)));
  CHECK_THAT(y.value(), WithinAbs(T(1), solution_tolerance<T>(1)));

  // With zero dual variables, the solver has to take a step
  problem.set_dual(constraint, Eigen::Vector<T, 1>{T(0)});
  x.set_value(T(1));
  y.set_value(T(1));
  iterations = 0;
  CHECK(problem.solve() == slp::ExitStatus::SUCCESS);
  CHECK(iterations > 0);

  auto dual = problem.dual(constraint);
  REQUIRE(dual.rows() == 1);
  CHECK_THAT(dual[0], WithinAbs(T(2), solution_tolerance<T>(10)));
}

TEMPLATE_TEST_CASE("Problem set_dual - Matrix constraints", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable(3);

  slp::Variable<T> J = T(0);
  for (int i = 0; i < 3; ++i) {
    J += (x[i] - T(2)) * (x[i] - T(2));
  }
  problem.minimize(J);

  problem.subject_to(x[0] >= T(-10));

  auto sum = x[1] + x[2] == T(5);
  problem.subject_to(sum);

  auto upper_bounds = x <= Eigen::Vector3<T>{T(1), T(4), T(2.25)};
  problem.subject_to(upper_bounds);

  const Eigen::Vector<T, 1> y{T(-3)};
  const Eigen::Vector3<T> z{T(4), T(5), T(6)};
  problem.set_dual(sum, y);
  problem.set_dual(upper_bounds, z);
  CHECK(problem.dual(sum) == y);
  CHECK(problem.dual(upper_bounds) == z);

  int iterations = 0;
  problem.add_callback([&](const slp::IterationInfo<T>&) { ++iterations; });

  // Returns the iteration count of solving from the solution's primal
  // variables
  auto solve = [&] {
    x.set_value(Eigen::Vector3<T>{T(1), T(2.75), T(2.25)});
    iterations = 0;

    CHECK(problem.solve() == slp::ExitStatus::SUCCESS);

    CHECK_THAT(x.value(0), WithinAbs(T(1), solution_tolerance<T>(1)));
    CHECK_THAT(x.value(1), WithinAbs(T(2.75), solution_tolerance<T>(1)));
    CHECK_THAT(x.value(2), WithinAbs(T(2.25), solution_tolerance<T>(1)));

    return iterations;
  };

  solve();

  // Starting from the previous solve's dual variables takes fewer iterations
  // than starting from wrong ones
  const int warm_iterations = solve();
  problem.set_dual(sum, y);
  problem.set_dual(upper_bounds, z);
  const int wrong_dual_iterations = solve();
  INFO(std::format("warm = {}, wrong duals = {}", warm_iterations,
                   wrong_dual_iterations));
  CHECK(warm_iterations < wrong_dual_iterations);
}

TEMPLATE_TEST_CASE("Problem set_dual - Warm start new problem", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  // Returns the iteration count of solving from the solution's primal
  // variables, with the solution's dual variables if warm is true
  auto solve = [](bool warm) {
    slp::Problem<T> problem;

    auto x = problem.decision_variable(3);

    slp::Variable<T> J = T(0);
    for (int i = 0; i < 3; ++i) {
      J += (x[i] - T(2)) * (x[i] - T(2));
    }
    problem.minimize(J);

    problem.subject_to(x[0] >= T(-10));

    auto sum = x[1] + x[2] == T(5);
    problem.subject_to(sum);

    auto upper_bounds = x <= Eigen::Vector3<T>{T(1), T(4), T(2.25)};
    problem.subject_to(upper_bounds);

    x.set_value(Eigen::Vector3<T>{T(1), T(2.75), T(2.25)});
    if (warm) {
      problem.set_dual(sum, Eigen::Vector<T, 1>{T(1.5)});
      problem.set_dual(upper_bounds, Eigen::Vector3<T>{T(2), T(0), T(1)});
    }

    int iterations = 0;
    problem.add_callback([&](const slp::IterationInfo<T>&) { ++iterations; });

    CHECK(problem.solve() == slp::ExitStatus::SUCCESS);

    CHECK_THAT(x.value(0), WithinAbs(T(1), solution_tolerance<T>(1)));
    CHECK_THAT(x.value(1), WithinAbs(T(2.75), solution_tolerance<T>(1)));
    CHECK_THAT(x.value(2), WithinAbs(T(2.25), solution_tolerance<T>(1)));

    return iterations;
  };

  const int cold_iterations = solve(false);
  const int warm_iterations = solve(true);
  INFO(std::format("cold = {}, warm = {}", cold_iterations, warm_iterations));
  CHECK(warm_iterations < cold_iterations);
}
