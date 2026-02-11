// Copyright (c) Sleipnir contributors

#include <chrono>
#include <format>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "cart_pole_util.hpp"
#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "explicit_double.hpp"
#include "lerp.hpp"
#include "rk4.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Problem - Cart-pole", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
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

  slp::Problem<T> problem;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.decision_variable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    X[0, k].set_value(lerp(x_initial[0], x_final[0], T(k) / T(N)));
    X[1, k].set_value(lerp(x_initial[1], x_final[1], T(k) / T(N)));
  }

  // u = f_x
  auto U = problem.decision_variable(1, N);

  // Initial conditions
  problem.subject_to(X.col(0) == x_initial);

  // Final conditions
  problem.subject_to(X.col(N) == x_final);

  // Cart position constraints
  problem.subject_to(slp::bounds(T(0), X.row(0), d_max));

  // Input constraints
  problem.subject_to(slp::bounds(-u_max, U, u_max));

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.subject_to(
        X.col(k + 1) ==
        rk4<T, decltype(&CartPoleUtil<T>::dynamics_variable),
            slp::VariableMatrix<T>, slp::VariableMatrix<T>>(
            &CartPoleUtil<T>::dynamics_variable, X.col(k), U.col(k), dt));
  }

  // Minimize sum squared inputs
  slp::Variable J = T(0);
  for (int k = 0; k < N; ++k) {
    J += U.col(k).T() * U.col(k);
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

#if defined(__APPLE__) && defined(__aarch64__)
  if constexpr (std::same_as<T, ExplicitDouble>) {
    REQUIRE(problem.solve({.diagnostics = true}) ==
            slp::ExitStatus::FEASIBILITY_RESTORATION_FAILED);
    SKIP("Fails with \"feasibility restoration failed\"");
  } else {
    REQUIRE(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
  }
#else
  REQUIRE(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
#endif

  // Verify initial state
  CHECK_THAT(X.value(0, 0), WithinAbs(x_initial[0], T(1e-8)));
  CHECK_THAT(X.value(1, 0), WithinAbs(x_initial[1], T(1e-8)));
  CHECK_THAT(X.value(2, 0), WithinAbs(x_initial[2], T(1e-8)));
  CHECK_THAT(X.value(3, 0), WithinAbs(x_initial[3], T(1e-8)));

  // Verify solution
  for (int k = 0; k < N; ++k) {
    // Cart position constraints
    CHECK(X[0, k] >= T(0));
    CHECK(X[0, k] <= d_max);

    // Input constraints
    CHECK(U[0, k] >= -u_max);
    CHECK(U[0, k] <= u_max);

    // Dynamics constraints
    Eigen::Vector<T, Eigen::Dynamic> expected_x_k1 =
        rk4<T>(&CartPoleUtil<T>::dynamics_scalar, X.col(k).value(),
               U.col(k).value(), dt);
    Eigen::Vector<T, Eigen::Dynamic> actual_x_k1 = X.col(k + 1).value();
    for (int row = 0; row < actual_x_k1.rows(); ++row) {
      CHECK_THAT(actual_x_k1[row], WithinAbs(expected_x_k1[row], T(1e-8)));
      INFO(std::format("  x({} @ k = {}", row, k));
    }
  }

  // Verify final state
  CHECK_THAT(X.value(0, N), WithinAbs(x_final[0], T(1e-8)));
  CHECK_THAT(X.value(1, N), WithinAbs(x_final[1], T(1e-8)));
  CHECK_THAT(X.value(2, N), WithinAbs(x_final[2], T(1e-8)));
  CHECK_THAT(X.value(3, N), WithinAbs(x_final[3], T(1e-8)));

  // Log states for offline viewing
  std::ofstream states{"Problem - Cart-pole states.csv"};
  if (states.is_open()) {
    states << "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),"
              "Pole angular velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{},{},{},{}\n", T(k) * dt.count(),
                            X.value(0, k), X.value(1, k), X.value(2, k),
                            X.value(3, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Problem - Cart-pole inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Cart force (N)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{}\n", T(k) * dt.count(), U.value(0, k));
      } else {
        inputs << std::format("{},{}\n", T(k) * dt.count(), T(0));
      }
    }
  }
}
