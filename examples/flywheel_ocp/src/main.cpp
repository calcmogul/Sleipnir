// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <print>

#include <Eigen/Core>
#include <sleipnir/optimization/ocp.hpp>

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

  auto f_discrete = [=](slp::VariableMatrix<double> x,
                        slp::VariableMatrix<double> u) {
    return A_discrete * x + B_discrete * u;
  };

  constexpr double r = 10.0;

  slp::OCP<double> solver(1, 1, dt, N, f_discrete, slp::DynamicsType::DISCRETE,
                          slp::TimestepMethod::FIXED,
                          slp::TranscriptionMethod::DIRECT_TRANSCRIPTION);
  solver.constrain_initial_state(0.0);
  solver.set_upper_input_bound(12);
  solver.set_lower_input_bound(-12);

  // Set up cost
  Eigen::Matrix<double, 1, N + 1> r_mat =
      Eigen::Matrix<double, 1, N + 1>::Constant(r);
  solver.minimize((r_mat - solver.X()) * (r_mat - solver.X()).T());

  solver.solve();

  // The first state
  std::println("x₀ = {}", solver.X().value(0, 0));

  // The first input
  std::println("u₀ = {}", solver.U().value(0, 0));
}
#endif
