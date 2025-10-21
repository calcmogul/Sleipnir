// Copyright (c) Sleipnir contributors

#include "sleipnir.hpp"

#include <cmath>

#include <Eigen/Core>

slp::Problem<double> flywheel_sleipnir(std::chrono::duration<double> dt,
                                       int N) {
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
  problem.subject_to(-12 <= U);
  problem.subject_to(U <= 12);

  // Cost function - minimize error
  Eigen::Matrix<double, 1, 1> r{10.0};
  slp::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += ((r - X.col(k)).T() * (r - X.col(k)));
  }
  problem.minimize(J);

  return problem;
}
