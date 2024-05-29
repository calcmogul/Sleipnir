// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>
#include <string>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sleipnir/optimization/ocp.hpp>

#include "catch_string_converters.hpp"
#include "util/scope_exit.hpp"

using namespace std::chrono_literals;

namespace {
bool near(double expected, double actual, double tolerance) {
  return std::abs(expected - actual) < tolerance;
}
}  // namespace

void flywheel_test(
    std::string test_name, double A, double B,
    const slp::function_ref<slp::VariableMatrix(
        const slp::VariableMatrix& x, const slp::VariableMatrix& u)>& f,
    slp::DynamicsType dynamics_type,
    slp::TranscriptionMethod transcription_method) {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 5ms;
  constexpr int N = T / dt;

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  double A_discrete = std::exp(A * dt.count());
  double B_discrete = (1.0 - A_discrete) * B;

  constexpr double r = 10.0;

  slp::OCP problem(1, 1, dt, N, f, dynamics_type, slp::TimestepMethod::FIXED,
                   transcription_method);
  problem.constrain_initial_state(0.0);
  problem.set_upper_input_bound(12);
  problem.set_lower_input_bound(-12);

  // Set up cost
  Eigen::Matrix<double, 1, N + 1> r_mat =
      Eigen::Matrix<double, 1, N + 1>::Constant(r);
  problem.minimize((r_mat - problem.X()) * (r_mat - problem.X()).T());

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // Voltage for steady-state velocity:
  //
  // rₖ₊₁ = Arₖ + Buₖ
  // uₖ = B⁺(rₖ₊₁ − Arₖ)
  // uₖ = B⁺(rₖ − Arₖ)
  // uₖ = B⁺(I − A)rₖ
  double u_ss = 1.0 / B_discrete * (1.0 - A_discrete) * r;

  // Verify initial state
  CHECK(problem.X().value(0, 0) == Catch::Approx(0.0).margin(1e-8));

  // Verify solution
  double x = 0.0;
  double u = 0.0;
  for (int k = 0; k < N; ++k) {
    // Verify state
    CHECK(problem.X().value(0, k) == Catch::Approx(x).margin(1e-2));

    // Determine expected input for this timestep
    double error = r - x;
    if (error > 1e-2) {
      // Max control input until the reference is reached
      u = 12.0;
    } else {
      // Maintain speed
      u = u_ss;
    }

    // Verify input
    if (k > 0 && k < N - 1 && near(12.0, problem.U().value(0, k - 1), 1e-2) &&
        near(u_ss, problem.U().value(0, k + 1), 1e-2)) {
      // If control input is transitioning between 12 and u_ss, ensure it's
      // within (u_ss, 12)
      CHECK(problem.U().value(0, k) >= u_ss);
      CHECK(problem.U().value(0, k) <= 12.0);
    } else {
      if (transcription_method ==
          slp::TranscriptionMethod::DIRECT_COLLOCATION) {
        // The tolerance is large because the trajectory is represented by a
        // spline, and splines chatter when transitioning quickly between
        // steady-states.
        CHECK(problem.U().value(0, k) == Catch::Approx(u).margin(2.0));
      } else {
        CHECK(problem.U().value(0, k) == Catch::Approx(u).margin(1e-4));
      }
    }

    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = A_discrete * x + B_discrete * u;
  }

  // Verify final state
  CHECK(problem.X().value(0, N) == Catch::Approx(r).margin(1e-7));

  // Log states for offline viewing
  std::ofstream states{fmt::format("{} states.csv", test_name)};
  if (states.is_open()) {
    states << "Time (s),Velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{}\n", k * dt.count(), problem.X().value(0, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{fmt::format("{} inputs.csv", test_name)};
  if (inputs.is_open()) {
    inputs << "Time (s),Voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", k * dt.count(),
                              problem.U().value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", k * dt.count(), 0.0);
      }
    }
  }
}

TEST_CASE("OCP - Flywheel (explicit)", "[OCP]") {
  constexpr double A = -1.0;
  constexpr double B = 1.0;

  auto f_ode = [=](slp::VariableMatrix x, slp::VariableMatrix u) {
    return A * x + B * u;
  };

  flywheel_test("OCP Flywheel Explicit Collocation", A, B, f_ode,
                slp::DynamicsType::EXPLICIT_ODE,
                slp::TranscriptionMethod::DIRECT_COLLOCATION);
  flywheel_test("OCP Flywheel Explicit Transcription", A, B, f_ode,
                slp::DynamicsType::EXPLICIT_ODE,
                slp::TranscriptionMethod::DIRECT_TRANSCRIPTION);
  flywheel_test("OCP Flywheel Explicit Single-Shooting", A, B, f_ode,
                slp::DynamicsType::EXPLICIT_ODE,
                slp::TranscriptionMethod::SINGLE_SHOOTING);
}

TEST_CASE("OCP - Flywheel (discrete)", "[OCP]") {
  constexpr double A = -1.0;
  constexpr double B = 1.0;
  constexpr std::chrono::duration<double> dt = 5ms;

  double A_discrete = std::exp(A * dt.count());
  double B_discrete = (1.0 - A_discrete) * B;

  auto f_discrete = [=](slp::VariableMatrix x, slp::VariableMatrix u) {
    return A_discrete * x + B_discrete * u;
  };

  flywheel_test("OCP Flywheel Discrete Transcription", A, B, f_discrete,
                slp::DynamicsType::DISCRETE,
                slp::TranscriptionMethod::DIRECT_TRANSCRIPTION);
  flywheel_test("OCP Flywheel Discrete Single-Shooting", A, B, f_discrete,
                slp::DynamicsType::DISCRETE,
                slp::TranscriptionMethod::SINGLE_SHOOTING);
}
