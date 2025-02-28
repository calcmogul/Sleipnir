// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <string>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

namespace {
template <typename T>
bool near(T expected, T actual, T tolerance) {
  using std::abs;
  return abs(expected - actual) < tolerance;
}
}  // namespace

template <typename T>
void flywheel_test(
    std::string test_name, T A, T B,
    const slp::function_ref<slp::VariableMatrix<T>(
        const slp::VariableMatrix<T>& x, const slp::VariableMatrix<T>& u)>& f,
    slp::DynamicsType dynamics_type,
    slp::TranscriptionMethod transcription_method) {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<T> TOTAL_TIME{T(5)};
  constexpr std::chrono::duration<T> dt{T(0.005)};
  constexpr int N = static_cast<int>(TOTAL_TIME / dt);

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  using std::exp;
  T A_discrete = exp(A * T(dt.count()));
  T B_discrete = (T(1) - A_discrete) * B;

  constexpr T r(10);

  slp::OCP<T> problem(1, 1, dt, N, f, dynamics_type, slp::TimestepMethod::FIXED,
                      transcription_method);
  problem.constrain_initial_state(T(0));
  problem.set_upper_input_bound(T(12));
  problem.set_lower_input_bound(T(-12));

  // Set up cost
  Eigen::Matrix<T, 1, N + 1> r_mat = Eigen::Matrix<T, 1, N + 1>::Constant(r);
  problem.minimize((r_mat - problem.X()) * (r_mat - problem.X()).T());

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  REQUIRE(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Voltage for steady-state velocity:
  //
  // rₖ₊₁ = Arₖ + Buₖ
  // uₖ = B⁺(rₖ₊₁ − Arₖ)
  // uₖ = B⁺(rₖ − Arₖ)
  // uₖ = B⁺(I − A)rₖ
  T u_ss = T(1) / B_discrete * (T(1) - A_discrete) * r;

  // Verify initial state
  CHECK_THAT(problem.X().value(0, 0), WithinAbs(T(0), T(1e-8)));

  // Verify solution
  T x(0);
  T u(0);
  for (int k = 0; k < N; ++k) {
    // Verify state
    CHECK_THAT(problem.X().value(0, k), WithinAbs(x, T(1e-2)));

    // Determine expected input for this timestep
    T error = r - x;
    if (error > T(1e-2)) {
      // Max control input until the reference is reached
      u = T(12);
    } else {
      // Maintain speed
      u = u_ss;
    }

    // Verify input
    if (k > 0 && k < N - 1 &&
        near(T(12), problem.U().value(0, k - 1), T(1e-2)) &&
        near(u_ss, problem.U().value(0, k + 1), T(1e-2))) {
      // If control input is transitioning between 12 and u_ss, ensure it's
      // within (u_ss, 12)
      CHECK(problem.U().value(0, k) >= u_ss);
      CHECK(problem.U().value(0, k) <= T(12));
    } else {
      if (transcription_method ==
          slp::TranscriptionMethod::DIRECT_COLLOCATION) {
        // The tolerance is large because the trajectory is represented by a
        // spline, and splines chatter when transitioning quickly between
        // steady-states.
        CHECK_THAT(problem.U().value(0, k), WithinAbs(u, T(2)));
      } else {
        CHECK_THAT(problem.U().value(0, k), WithinAbs(u, T(3e-4)));
      }
    }

    INFO(std::format("  k = {}", k));

    // Project state forward
    x = A_discrete * x + B_discrete * u;
  }

  // Verify final state
  CHECK_THAT(problem.X().value(0, N), WithinAbs(r, T(1e-6)));

  // Log states for offline viewing
  std::ofstream states{std::format("{} states.csv", test_name)};
  if (states.is_open()) {
    states << "Time (s),Velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{}\n", T(k) * dt.count(),
                            problem.X().value(0, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{std::format("{} inputs.csv", test_name)};
  if (inputs.is_open()) {
    inputs << "Time (s),Voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{}\n", T(k) * dt.count(),
                              problem.U().value(0, k));
      } else {
        inputs << std::format("{},{}\n", T(k) * dt.count(), T(0));
      }
    }
  }
}

TEMPLATE_TEST_CASE("OCP - Flywheel (explicit)", "[OCP]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  constexpr T A(-1);
  constexpr T B(1);

  auto f_ode = [=](slp::VariableMatrix<T> x, slp::VariableMatrix<T> u) {
    return A * x + B * u;
  };

  flywheel_test<T>("OCP - Flywheel (explicit) direct collocation", A, B, f_ode,
                   slp::DynamicsType::EXPLICIT_ODE,
                   slp::TranscriptionMethod::DIRECT_COLLOCATION);
  flywheel_test<T>("OCP - Flywheel (explicit) direct transcription", A, B,
                   f_ode, slp::DynamicsType::EXPLICIT_ODE,
                   slp::TranscriptionMethod::DIRECT_TRANSCRIPTION);
  flywheel_test<T>("OCP - Flywheel (explicit) single-shooting", A, B, f_ode,
                   slp::DynamicsType::EXPLICIT_ODE,
                   slp::TranscriptionMethod::SINGLE_SHOOTING);
}

TEMPLATE_TEST_CASE("OCP - Flywheel (discrete)", "[OCP]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  constexpr T A(-1);
  constexpr T B(1);
  constexpr std::chrono::duration<T> dt{T(0.005)};

  using std::exp;
  T A_discrete = exp(A * T(dt.count()));
  T B_discrete = (T(1) - A_discrete) * B;

  auto f_discrete = [=](slp::VariableMatrix<T> x, slp::VariableMatrix<T> u) {
    return A_discrete * x + B_discrete * u;
  };

  flywheel_test<T>("OCP - Flywheel (discrete) direct transcription", A, B,
                   f_discrete, slp::DynamicsType::DISCRETE,
                   slp::TranscriptionMethod::DIRECT_TRANSCRIPTION);
  flywheel_test<T>("OCP - Flywheel (discrete) single-shooting", A, B,
                   f_discrete, slp::DynamicsType::DISCRETE,
                   slp::TranscriptionMethod::SINGLE_SHOOTING);
}
