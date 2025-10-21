// Copyright (c) Sleipnir contributors

#include <chrono>
#include <format>
#include <fstream>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_string_converters.hpp"
#include "differential_drive_util.hpp"
#include "rk4.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("OCP - Differential drive", "[OCP]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr int N = 50;

  constexpr std::chrono::duration<T> min_timestep{T(0.05)};
  constexpr Eigen::Vector<T, 5> x_initial{{T(0), T(0), T(0), T(0), T(0)}};
  constexpr Eigen::Vector<T, 5> x_final{{T(1), T(1), T(0), T(0), T(0)}};
  constexpr Eigen::Matrix<T, 2, 1> u_min{{T(-12.0), T(-12.0)}};
  constexpr Eigen::Matrix<T, 2, 1> u_max{{T(12.0), T(12.0)}};

  slp::OCP<T> problem(
      5, 2, min_timestep, N, DifferentialDriveUtil<T>::dynamics_variable,
      slp::DynamicsType::EXPLICIT_ODE, slp::TimestepMethod::VARIABLE_SINGLE,
      slp::TranscriptionMethod::DIRECT_TRANSCRIPTION);

  // Seed the min time formulation with lerp between waypoints
  for (int i = 0; i < N + 1; ++i) {
    problem.X()[0, i].set_value(static_cast<T>(i) / (N + 1));
    problem.X()[1, i].set_value(static_cast<T>(i) / (N + 1));
  }

  problem.constrain_initial_state(x_initial);
  problem.constrain_final_state(x_final);

  problem.set_lower_input_bound(u_min);
  problem.set_upper_input_bound(u_max);

  problem.set_min_timestep(min_timestep);
  problem.set_max_timestep(std::chrono::duration<T>{T(3.0)});

  // Set up cost
  problem.minimize(problem.dt() * Eigen::Matrix<T, N + 1, 1>::Ones());

  CHECK(problem.cost_function_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.max_iterations = 1000, .diagnostics = true}) ==
        slp::ExitStatus::SUCCESS);

  auto X = problem.X();
  auto U = problem.U();

  // Verify initial state
  CHECK(X.value(0, 0) == Catch::Approx(x_initial[0]).margin(T(1e-8)));
  CHECK(X.value(1, 0) == Catch::Approx(x_initial[1]).margin(T(1e-8)));
  CHECK(X.value(2, 0) == Catch::Approx(x_initial[2]).margin(T(1e-8)));
  CHECK(X.value(3, 0) == Catch::Approx(x_initial[3]).margin(T(1e-8)));
  CHECK(X.value(4, 0) == Catch::Approx(x_initial[4]).margin(T(1e-8)));

  // FIXME: Replay diverges
  SKIP("Replay diverges");

  // Verify solution
  Eigen::Vector<T, 5> x{T(0), T(0), T(0), T(0), T(0)};
  Eigen::Vector<T, 2> u{T(0), T(0)};
  for (int k = 0; k < N; ++k) {
    u = U.col(k).value();

    // Input constraints
    CHECK(U[0, k].value() >= -u_max[0]);
    CHECK(U[0, k].value() <= u_max[0]);
    CHECK(U[1, k].value() >= -u_max[1]);
    CHECK(U[1, k].value() <= u_max[1]);

    // Verify state
    CHECK(X.value(0, k) == Catch::Approx(x[0]).margin(T(1e-8)));
    CHECK(X.value(1, k) == Catch::Approx(x[1]).margin(T(1e-8)));
    CHECK(X.value(2, k) == Catch::Approx(x[2]).margin(T(1e-8)));
    CHECK(X.value(3, k) == Catch::Approx(x[3]).margin(T(1e-8)));
    CHECK(X.value(4, k) == Catch::Approx(x[4]).margin(T(1e-8)));

    INFO(std::format("  k = {}", k));

    // Project state forward
    x = rk4<T>(DifferentialDriveUtil<T>::dynamics_scalar, x, u,
               std::chrono::duration<T>{problem.dt().value(0, k)});
  }

  // Verify final state
  CHECK(X.value(0, N) == Catch::Approx(x_final[0]).margin(T(1e-8)));
  CHECK(X.value(1, N) == Catch::Approx(x_final[1]).margin(T(1e-8)));
  CHECK(X.value(2, N) == Catch::Approx(x_final[2]).margin(T(1e-8)));
  CHECK(X.value(3, N) == Catch::Approx(x_final[3]).margin(T(1e-8)));
  CHECK(X.value(4, N) == Catch::Approx(x_final[4]).margin(T(1e-8)));

  // Log states for offline viewing
  std::ofstream states{"OCP Differential drive states.csv"};
  if (states.is_open()) {
    states << "Time (s),X position (m),Y position (m),Heading (rad),Left "
              "velocity (m/s),Right velocity (m/s)\n";

    T time{0.0};
    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{},{},{},{},{}\n", time,
                            problem.X().value(0, k), problem.X().value(1, k),
                            problem.X().value(2, k), problem.X().value(3, k),
                            problem.X().value(4, k));

      time += problem.dt().value(0, k);
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"OCP Differential drive inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Left voltage (V),Right voltage (V)\n";

    T time{0.0};
    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{},{}\n", time, problem.U().value(0, k),
                              problem.U().value(1, k));
      } else {
        inputs << std::format("{},{},{}\n", time, T(0.0), T(0.0));
      }

      time += problem.dt().value(0, k);
    }
  }
}
