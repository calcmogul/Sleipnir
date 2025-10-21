// Copyright (c) Sleipnir contributors

#include <chrono>
#include <format>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "cart_pole_util.hpp"
#include "catch_string_converters.hpp"
#include "math_util.hpp"
#include "rk4.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("OCP - Cart-pole", "[OCP]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<T> TOTAL_TIME{T(5.0)};
  constexpr std::chrono::duration<T> dt{T(0.05)};
  constexpr int N = static_cast<int>(TOTAL_TIME / dt);

  constexpr T u_max{20.0};  // N
  constexpr T d_max{2.0};   // m

  constexpr Eigen::Vector<T, 4> x_initial{{T(0), T(0), T(0), T(0)}};
  constexpr Eigen::Vector<T, 4> x_final{
      {T(1), T(std::numbers::pi), T(0), T(0)}};

  slp::OCP<T> problem(4, 1, dt, N, CartPoleUtil<T>::dynamics_variable,
                      slp::DynamicsType::EXPLICIT_ODE,
                      slp::TimestepMethod::VARIABLE_SINGLE,
                      slp::TranscriptionMethod::DIRECT_COLLOCATION);

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.X();

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    X[0, k].set_value(lerp(x_initial[0], x_final[0], static_cast<T>(k) / N));
    X[1, k].set_value(lerp(x_initial[1], x_final[1], static_cast<T>(k) / N));
  }

  // Initial conditions
  problem.constrain_initial_state(x_initial);

  // Final conditions
  problem.constrain_final_state(x_final);

  // Cart position constraints
  problem.for_each_step([&](const slp::VariableMatrix<T>& x,
                            [[maybe_unused]]
                            const slp::VariableMatrix<T>& u) {
    problem.subject_to(x[0] >= T(0));
    problem.subject_to(x[0] <= d_max);
  });

  // Input constraints
  problem.set_lower_input_bound(-u_max);
  problem.set_upper_input_bound(u_max);

  // u = f_x
  auto U = problem.U();

  // Minimize sum squared inputs
  slp::Variable J = T(0.0);
  for (int k = 0; k < N; ++k) {
    J += U.col(k).T() * U.col(k);
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Verify initial state
  CHECK(X.value(0, 0) == Catch::Approx(x_initial[0]).margin(T(1e-8)));
  CHECK(X.value(1, 0) == Catch::Approx(x_initial[1]).margin(T(1e-8)));
  CHECK(X.value(2, 0) == Catch::Approx(x_initial[2]).margin(T(1e-8)));
  CHECK(X.value(3, 0) == Catch::Approx(x_initial[3]).margin(T(1e-8)));

  // FIXME: Replay diverges
  SKIP("Replay diverges");

  // Verify solution
  Eigen::Matrix<T, 4, 1> x{T(0), T(0), T(0), T(0)};
  Eigen::Matrix<T, 1, 1> u{T(0)};
  for (int k = 0; k < N; ++k) {
    // Cart position constraints
    CHECK(X[0, k] >= T(0));
    CHECK(X[0, k] <= d_max);

    // Input constraints
    CHECK(U[0, k] >= -u_max);
    CHECK(U[0, k] <= u_max);

    // Verify state
    CHECK(X.value(0, k) == Catch::Approx(x[0]).margin(T(1e-2)));
    CHECK(X.value(1, k) == Catch::Approx(x[1]).margin(T(1e-2)));
    CHECK(X.value(2, k) == Catch::Approx(x[2]).margin(T(1e-2)));
    CHECK(X.value(3, k) == Catch::Approx(x[3]).margin(T(1e-2)));
    INFO(std::format("  k = {}", k));

    // Project state forward
    x = rk4<T>(CartPoleUtil<T>::dynamics_scalar, x, u, dt);
  }

  // Verify final state
  CHECK(X.value(0, N) == Catch::Approx(x_final[0]).margin(T(1e-8)));
  CHECK(X.value(1, N) == Catch::Approx(x_final[1]).margin(T(1e-8)));
  CHECK(X.value(2, N) == Catch::Approx(x_final[2]).margin(T(1e-8)));
  CHECK(X.value(3, N) == Catch::Approx(x_final[3]).margin(T(1e-8)));

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
        inputs << std::format("{},{}\n", k * dt.count(), T(0.0));
      }
    }
  }
}
