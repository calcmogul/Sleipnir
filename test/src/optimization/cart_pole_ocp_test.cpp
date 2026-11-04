// Copyright (c) Sleipnir contributors

#include <chrono>
#include <format>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/optimization/ocp/dynamics_type.hpp>
#include <sleipnir/optimization/ocp/timestep_method.hpp>
#include <sleipnir/optimization/ocp/transcription_method.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/util/pool.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "cart_pole_util.hpp"
#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "lerp.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("OCP - Cart-pole", "[OCP]", SCALAR_TYPES) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<T> TOTAL_TIME{T(5)};
  constexpr std::chrono::duration<T> dt{T(0.05)};
  constexpr int N = static_cast<int>(TOTAL_TIME / dt);

  constexpr T u_max(20);  // N
  constexpr T d_max(2);   // m

  constexpr Eigen::Vector4<T> x_initial{{T(0), T(0), T(0), T(0)}};
  constexpr Eigen::Vector4<T> x_final{{T(1), T(std::numbers::pi), T(0), T(0)}};

  slp::OCP<T> problem(4, 1, dt, N, CartPoleUtil<T>::dynamics_variable,
                      slp::DynamicsType::EXPLICIT_ODE,
                      slp::TimestepMethod::VARIABLE_SINGLE,
                      slp::TranscriptionMethod::DIRECT_COLLOCATION);

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.X();

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    X[0, k].set_value(lerp(x_initial[0], x_final[0], T(k) / T(N)));
    X[1, k].set_value(lerp(x_initial[1], x_final[1], T(k) / T(N)));
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
  CHECK_THAT(X.value(0, 0), WithinAbs(x_initial[0], solution_tolerance<T>(1)));
  CHECK_THAT(X.value(1, 0), WithinAbs(x_initial[1], solution_tolerance<T>(1)));
  CHECK_THAT(X.value(2, 0), WithinAbs(x_initial[2], solution_tolerance<T>(1)));
  CHECK_THAT(X.value(3, 0), WithinAbs(x_initial[3], solution_tolerance<T>(1)));

  // Constraint residuals are part of the KKT error, so the solver only
  // guarantees bounds to within its tolerance. For example, the initial state
  // pins X(0, 0) to its lower bound, so its sign is rounding error.
  const T bound_tolerance = solution_tolerance<T>(1);

  // Verify solution
  for (int k = 0; k < N; ++k) {
    // Cart position constraints
    CHECK(X.value(0, k) >= -bound_tolerance);
    CHECK(X.value(0, k) <= d_max + bound_tolerance);

    // Input constraints
    CHECK(U.value(0, k) >= -u_max - bound_tolerance);
    CHECK(U.value(0, k) <= u_max + bound_tolerance);

    // Dynamics constraints
    //
    // Direct collocation constrains the system dynamics at the midpoint of a
    // cubic Hermite spline through each pair of adjacent states.
    constexpr auto f = &CartPoleUtil<T>::dynamics_scalar;
    T h = problem.dt().value(0, k);
    Eigen::Vector4<T> x_begin = X.col(k).value();
    Eigen::Vector4<T> x_end = X.col(k + 1).value();
    Eigen::Vector<T, 1> u_begin = U.col(k).value();
    Eigen::Vector<T, 1> u_end = U.col(k + 1).value();

    Eigen::Vector4<T> xdot_begin = f(x_begin, u_begin);
    Eigen::Vector4<T> xdot_end = f(x_end, u_end);
    Eigen::Vector4<T> xdot_c = T(-3) / (T(2) * h) * (x_begin - x_end) -
                               T(0.25) * (xdot_begin + xdot_end);

    Eigen::Vector4<T> x_c =
        T(0.5) * (x_begin + x_end) + h / T(8) * (xdot_begin - xdot_end);
    Eigen::Vector<T, 1> u_c = T(0.5) * (u_begin + u_end);

    Eigen::Vector4<T> expected_xdot_c = f(x_c, u_c);
    for (int row = 0; row < xdot_c.rows(); ++row) {
      INFO(std::format("  ẋ({}) @ k = {}", row, k));
      CHECK_THAT(xdot_c[row],
                 WithinAbs(expected_xdot_c[row], solution_tolerance<T>(1)));
    }
  }

  // Verify final state
  CHECK_THAT(X.value(0, N),
             WithinAbs(x_final[0],
                       solution_tolerance<T>(1, SolutionType::DEGENERATE)));
  CHECK_THAT(X.value(1, N), WithinAbs(x_final[1], solution_tolerance<T>(1)));
  CHECK_THAT(X.value(2, N), WithinAbs(x_final[2], solution_tolerance<T>(1)));
  CHECK_THAT(X.value(3, N), WithinAbs(x_final[3], solution_tolerance<T>(1)));

  // Log states for offline viewing
  std::ofstream states{"OCP - Cart-pole states.csv"};
  if (states.is_open()) {
    states << "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),"
              "Pole angular velocity (rad/s)\n";

    T time(0);
    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{},{},{},{}\n", time, X.value(0, k),
                            X.value(1, k), X.value(2, k), X.value(3, k));

      time += problem.dt().value(0, k);
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"OCP - Cart-pole inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Cart force (N)\n";

    T time(0);
    for (int k = 0; k < N + 1; ++k) {
      inputs << std::format("{},{}\n", time, U.value(0, k));

      time += problem.dt().value(0, k);
    }
  }
}
