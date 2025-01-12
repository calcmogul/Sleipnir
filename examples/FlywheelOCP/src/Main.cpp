// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>

#include <Eigen/Core>
#include <fmt/base.h>
#include <sleipnir/control/OCPSolver.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#ifndef RUNNING_TESTS
int main() {
  using namespace std::chrono_literals;

  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 5ms;
  constexpr int N = T / dt;

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  Eigen::Matrix<double, 1, 1> A{-1.0};
  Eigen::Matrix<double, 1, 1> B{1.0};

  Eigen::Matrix<double, 1, 1> A_discrete{std::exp(A(0) * dt.count())};
  Eigen::Matrix<double, 1, 1> B_discrete{(1.0 - A_discrete(0)) * B(0)};

  auto f_discrete = [=](sleipnir::VariableMatrix x,
                        sleipnir::VariableMatrix u) {
    return A_discrete * x + B_discrete * u;
  };

  constexpr double r = 10.0;

  sleipnir::OCPSolver solver(
      1, 1, dt, N, f_discrete, sleipnir::DynamicsType::kDiscrete,
      sleipnir::TimestepMethod::kFixed,
      sleipnir::TranscriptionMethod::kDirectTranscription);
  solver.ConstrainInitialState(0.0);
  solver.SetUpperInputBound(12);
  solver.SetLowerInputBound(-12);

  // Set up cost
  Eigen::Matrix<double, 1, N + 1> r_mat =
      Eigen::Matrix<double, 1, N + 1>::Constant(r);
  solver.Minimize((r_mat - solver.X()) * (r_mat - solver.X()).T());

  solver.Solve();

  // The first state
  fmt::println("x₀ = {}", solver.X().Value(0, 0));

  // The first input
  fmt::println("u₀ = {}", solver.U().Value(0, 0));
}
#endif
