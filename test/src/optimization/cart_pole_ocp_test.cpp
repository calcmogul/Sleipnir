// Copyright (c) Sleipnir contributors

#include <chrono>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "cart_pole_util.hpp"
#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "lerp.hpp"
#include "rk4.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("OCP - Cart-pole", "[OCP]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<T> TOTAL_TIME{T(5)};
  constexpr std::chrono::duration<T> dt{T(0.05)};
  constexpr int N = static_cast<int>(TOTAL_TIME / dt);

  constexpr T u_max(20);  // N
  constexpr T d_max(2);   // m

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
    X(0, k).set_value(lerp(x_initial[0], x_final[0], T(k) / T(N)));
    X(1, k).set_value(lerp(x_initial[1], x_final[1], T(k) / T(N)));
  }

  // Initial conditions
  problem.constrain_initial_state(x_initial);

  // Final conditions
  problem.constrain_final_state(x_final);

  // Cart position constraints
  problem.for_each_step([&](const slp::VariableMatrix<T>& x,
                            [[maybe_unused]]
                            const slp::VariableMatrix<T>& u) {
    problem.subject_to(slp::bounds(T(0), x[0], d_max));
  });

  // Input constraints
  problem.set_lower_input_bound(-u_max);
  problem.set_upper_input_bound(u_max);

  // u = f_x
  auto U = problem.U();

  // Minimize sum squared inputs
  slp::Variable J = T(0);
  for (int k = 0; k < N; ++k) {
    J += U.col(k).T() * U.col(k);
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  REQUIRE(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Verify initial state
  CHECK_THAT(X.value(0, 0), WithinAbs(x_initial[0], T(1e-8)));
  CHECK_THAT(X.value(1, 0), WithinAbs(x_initial[1], T(1e-8)));
  CHECK_THAT(X.value(2, 0), WithinAbs(x_initial[2], T(1e-8)));
  CHECK_THAT(X.value(3, 0), WithinAbs(x_initial[3], T(1e-8)));

  // FIXME: Replay diverges
  SKIP("Replay diverges");

  // Verify solution
  Eigen::Matrix<T, 4, 1> x{T(0), T(0), T(0), T(0)};
  Eigen::Matrix<T, 1, 1> u{T(0)};
  for (int k = 0; k < N; ++k) {
    // Cart position constraints
    CHECK(X(0, k) >= T(0));
    CHECK(X(0, k) <= d_max);

    // Input constraints
    CHECK(U(0, k) >= -u_max);
    CHECK(U(0, k) <= u_max);

    // Verify state
    CHECK_THAT(X.value(0, k), WithinAbs(x[0], T(1e-2)));
    CHECK_THAT(X.value(1, k), WithinAbs(x[1], T(1e-2)));
    CHECK_THAT(X.value(2, k), WithinAbs(x[2], T(1e-2)));
    CHECK_THAT(X.value(3, k), WithinAbs(x[3], T(1e-2)));
    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = rk4<T>(CartPoleUtil<T>::dynamics_scalar, x, u, dt);
  }

  // Verify final state
  CHECK_THAT(X.value(0, N), WithinAbs(x_final[0], T(1e-8)));
  CHECK_THAT(X.value(1, N), WithinAbs(x_final[1], T(1e-8)));
  CHECK_THAT(X.value(2, N), WithinAbs(x_final[2], T(1e-8)));
  CHECK_THAT(X.value(3, N), WithinAbs(x_final[3], T(1e-8)));

  // Log states for offline viewing
  std::ofstream states{"OCP - Cart-pole states.csv"};
  if (states.is_open()) {
    states << "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),"
              "Pole angular velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{},{},{},{}\n", T(k) * dt.count(),
                            X.value(0, k), X.value(1, k), X.value(2, k),
                            X.value(3, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"OCP - Cart-pole inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Cart force (N)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", T(k) * dt.count(),
                              problem.U().value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", T(k) * dt.count(), T(0));
      }
    }
  }
}
