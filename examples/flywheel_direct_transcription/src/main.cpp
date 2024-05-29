// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>

#include <fmt/base.h>
#include <sleipnir/optimization/problem.hpp>

#ifndef RUNNING_TESTS
int main() {
  using namespace std::chrono_literals;

  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 5ms;
  constexpr int N = T / dt;

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  Eigen::Matrix<double, 1, 1> A{std::exp(-dt.count())};
  Eigen::Matrix<double, 1, 1> B{1.0 - std::exp(-dt.count())};

  slp::Problem<double> problem;
  auto X = problem.decision_variable(1, N + 1);
  auto U = problem.decision_variable(1, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.subject_to(X.col(k + 1) == A * X.col(k) + B * U.col(k));
  }

  // State and input constraints
  problem.subject_to(X.col(0) == 0.0);
  problem.subject_to(slp::bounds(-12, U, 12));

  // Cost function - minimize error
  Eigen::Matrix<double, 1, 1> r{10.0};
  slp::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += (r - X.col(k)).T() * (r - X.col(k));
  }
  problem.minimize(J);

  problem.solve();

  // The first state
  fmt::println("x₀ = {}", X.value(0, 0));

  // The first input
  fmt::println("u₀ = {}", U.value(0, 0));
}
#endif
