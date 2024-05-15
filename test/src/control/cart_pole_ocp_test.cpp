// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/control/ocp.hpp>

#include "cart_pole_util.hpp"
#include "catch_string_converters.hpp"
#include "rk4.hpp"
#include "util/scope_exit.hpp"

TEST_CASE("OCP - Cart-pole", "[OCP]") {
  using namespace std::chrono_literals;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 50ms;
  constexpr int N = T / dt;

  constexpr double u_max = 20.0;  // N
  constexpr double d_max = 2.0;   // m

  constexpr Eigen::Vector<double, 4> x_initial{{0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 4> x_final{{1.0, std::numbers::pi, 0.0, 0.0}};

  slp::OCP problem(4, 1, dt, N, cart_pole_dynamics,
                   slp::DynamicsType::EXPLICIT_ODE,
                   slp::TimestepMethod::VARIABLE_SINGLE,
                   slp::TranscriptionMethod::DIRECT_COLLOCATION);

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.X();

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    X[0, k].set_value(
        std::lerp(x_initial(0), x_final(0), static_cast<double>(k) / N));
    X[1, k].set_value(
        std::lerp(x_initial(1), x_final(1), static_cast<double>(k) / N));
  }

  // u = f_x
  auto U = problem.U();

  // Initial conditions
  problem.constrain_initial_state(x_initial);

  // Final conditions
  problem.constrain_final_state(x_final);

  // Cart position constraints
  problem.for_each_step([&](const slp::VariableMatrix& x,
                            [[maybe_unused]]
                            const slp::VariableMatrix& u) {
    problem.subject_to(x[0] >= 0.0);
    problem.subject_to(x[0] <= d_max);
  });

  // Input constraints
  problem.set_lower_input_bound(-u_max);
  problem.set_upper_input_bound(u_max);

  // Minimize sum squared inputs
  slp::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U.col(k).T() * U.col(k);
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Verify initial state
  CHECK(X.value(0, 0) == Catch::Approx(x_initial(0)).margin(1e-8));
  CHECK(X.value(1, 0) == Catch::Approx(x_initial(1)).margin(1e-8));
  CHECK(X.value(2, 0) == Catch::Approx(x_initial(2)).margin(1e-8));
  CHECK(X.value(3, 0) == Catch::Approx(x_initial(3)).margin(1e-8));

  // FIXME: Replay diverges
  SKIP("Replay diverges");

  // Verify solution
  Eigen::Matrix<double, 4, 1> x{0.0, 0.0, 0.0, 0.0};
  Eigen::Matrix<double, 1, 1> u{0.0};
  for (int k = 0; k < N; ++k) {
    // Cart position constraints
    CHECK(X[0, k] >= 0.0);
    CHECK(X[0, k] <= d_max);

    // Input constraints
    CHECK(U[0, k] >= -u_max);
    CHECK(U[0, k] <= u_max);

    // Verify state
    CHECK(X.value(0, k) == Catch::Approx(x(0)).margin(1e-2));
    CHECK(X.value(1, k) == Catch::Approx(x(1)).margin(1e-2));
    CHECK(X.value(2, k) == Catch::Approx(x(2)).margin(1e-2));
    CHECK(X.value(3, k) == Catch::Approx(x(3)).margin(1e-2));
    INFO(std::format("  k = {}", k));

    // Project state forward
    x = rk4(cart_pole_dynamics_double, x, u, dt);
  }

  // Verify final state
  CHECK(X.value(0, N - 1) == Catch::Approx(x_final(0)).margin(1e-8));
  CHECK(X.value(1, N - 1) == Catch::Approx(x_final(1)).margin(1e-8));
  CHECK(X.value(2, N - 1) == Catch::Approx(x_final(2)).margin(1e-8));
  CHECK(X.value(3, N - 1) == Catch::Approx(x_final(3)).margin(1e-8));

  // Log states for offline viewing
  std::ofstream states{"OCP Cart-pole states.csv"};
  if (states.is_open()) {
    states << "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),"
              "Pole angular velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{},{},{},{}\n", k * dt.count(), X.value(0, k),
                            X.value(1, k), X.value(2, k), X.value(3, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"OCP Cart-pole inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Cart force (N)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{}\n", k * dt.count(),
                              problem.U().value(0, k));
      } else {
        inputs << std::format("{},{}\n", k * dt.count(), 0.0);
      }
    }
  }
}
