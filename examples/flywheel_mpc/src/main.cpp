// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>

#include <Eigen/Core>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/optimization/problem.hpp>

#ifndef RUNNING_TESTS
int main() {
  using namespace std::chrono_literals;

  constexpr std::chrono::milliseconds T = 4s;
  constexpr std::chrono::milliseconds T_horizon = 1s;
  constexpr std::chrono::milliseconds dt = 50ms;
  constexpr int N = T_horizon / dt;

  constexpr double dt_s = std::chrono::duration<double>{dt}.count();

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  Eigen::Matrix<double, 1, 1> A{std::exp(-dt_s)};
  Eigen::Matrix<double, 1, 1> B{1.0 - std::exp(-dt_s)};

  // The problem is set up once. Each timestep only changes the values of its
  // parameters, so every solve after the first is warm-started from the
  // previous solve's solution.
  slp::Problem<double> problem;
  auto X = problem.decision_variable(1, N + 1);
  auto U = problem.decision_variable(1, N);

  // Current state
  auto x_0 = problem.parameter();

  // Reference
  auto r = problem.parameter();

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.subject_to(X.col(k + 1) == A * X.col(k) + B * U.col(k));
  }

  // State and input constraints
  problem.subject_to(X[0, 0] == x_0);
  problem.subject_to(slp::bounds(-12, U, 12));

  // Cost function - minimize error
  slp::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += (r - X[0, k]) * (r - X[0, k]);
  }
  problem.minimize(J);

  double x_plant = 0.0;

  for (auto t = 0ms; t < T; t += dt) {
    // The reference changes halfway through
    r.set_value(t < 2s ? 10.0 : 5.0);

    // A disturbance slows the flywheel down
    if (t == 1s) {
      x_plant -= 3.0;
    }

    x_0.set_value(x_plant);

    problem.solve();

    // Apply the first input to the plant
    const double u = U.value(0, 0);
    x_plant = A[0, 0] * x_plant + B[0, 0] * u;
  }
}
#endif
