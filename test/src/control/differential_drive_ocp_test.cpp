// Copyright (c) Sleipnir contributors

#include <chrono>
#include <format>
#include <fstream>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/control/ocp.hpp>

#include "catch_string_converters.hpp"
#include "differential_drive_util.hpp"
#include "rk4.hpp"
#include "util/scope_exit.hpp"

TEST_CASE("OCP - Differential drive", "[OCP]") {
  using namespace std::chrono_literals;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr int N = 50;

  constexpr std::chrono::duration<double> min_timestep = 50ms;
  constexpr Eigen::Vector<double, 5> x_initial{{0.0, 0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 5> x_final{{1.0, 1.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Matrix<double, 2, 1> u_min{{-12.0, -12.0}};
  constexpr Eigen::Matrix<double, 2, 1> u_max{{12.0, 12.0}};

  slp::OCP problem(5, 2, min_timestep, N, differential_drive_dynamics,
                   slp::DynamicsType::EXPLICIT_ODE,
                   slp::TimestepMethod::VARIABLE_SINGLE,
                   slp::TranscriptionMethod::DIRECT_TRANSCRIPTION);

  // Seed the min time formulation with lerp between waypoints
  for (int i = 0; i < N + 1; ++i) {
    problem.X()[0, i].set_value(static_cast<double>(i) / (N + 1));
    problem.X()[1, i].set_value(static_cast<double>(i) / (N + 1));
  }

  problem.constrain_initial_state(x_initial);
  problem.constrain_final_state(x_final);

  problem.set_lower_input_bound(u_min);
  problem.set_upper_input_bound(u_max);

  problem.SetMinTimestep(min_timestep);
  problem.SetMaxTimestep(3s);

  // Set up cost
  problem.minimize(problem.dt() * Eigen::Matrix<double, N + 1, 1>::Ones());

  CHECK(problem.cost_function_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.max_iterations = 1000, .diagnostics = true}) ==
        slp::ExitStatus::SUCCESS);

  auto X = problem.X();
  auto U = problem.U();

  // Verify initial state
  CHECK(X.value(0, 0) == Catch::Approx(x_initial(0)).margin(1e-8));
  CHECK(X.value(1, 0) == Catch::Approx(x_initial(1)).margin(1e-8));
  CHECK(X.value(2, 0) == Catch::Approx(x_initial(2)).margin(1e-8));
  CHECK(X.value(3, 0) == Catch::Approx(x_initial(3)).margin(1e-8));
  CHECK(X.value(4, 0) == Catch::Approx(x_initial(4)).margin(1e-8));

  // FIXME: Replay diverges
  SKIP("Replay diverges");

  // Verify solution
  Eigen::Vector<double, 5> x{0.0, 0.0, 0.0, 0.0, 0.0};
  Eigen::Vector<double, 2> u{0.0, 0.0};
  for (int k = 0; k < N; ++k) {
    u = U.col(k).value();

    // Input constraints
    CHECK(U[0, k].value() >= -u_max[0]);
    CHECK(U[0, k].value() <= u_max[0]);
    CHECK(U[1, k].value() >= -u_max[1]);
    CHECK(U[1, k].value() <= u_max[1]);

    // Verify state
    CHECK(X.value(0, k) == Catch::Approx(x[0]).margin(1e-8));
    CHECK(X.value(1, k) == Catch::Approx(x[1]).margin(1e-8));
    CHECK(X.value(2, k) == Catch::Approx(x[2]).margin(1e-8));
    CHECK(X.value(3, k) == Catch::Approx(x[3]).margin(1e-8));
    CHECK(X.value(4, k) == Catch::Approx(x[4]).margin(1e-8));

    INFO(std::format("  k = {}", k));

    // Project state forward
    x = rk4(differential_drive_dynamics_double, x, u,
            std::chrono::duration<double>{problem.dt().value(0, k)});
  }

  // Verify final state
  CHECK(X.value(0, N) == Catch::Approx(x_final[0]).margin(1e-8));
  CHECK(X.value(1, N) == Catch::Approx(x_final[1]).margin(1e-8));
  CHECK(X.value(2, N) == Catch::Approx(x_final[2]).margin(1e-8));
  CHECK(X.value(3, N) == Catch::Approx(x_final[3]).margin(1e-8));
  CHECK(X.value(4, N) == Catch::Approx(x_final[4]).margin(1e-8));

  // Log states for offline viewing
  std::ofstream states{"OCP Differential drive states.csv"};
  if (states.is_open()) {
    states << "Time (s),X position (m),Y position (m),Heading (rad),Left "
              "velocity (m/s),Right velocity (m/s)\n";

    double time = 0.0;
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

    double time = 0.0;
    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{},{}\n", time, problem.U().value(0, k),
                              problem.U().value(1, k));
      } else {
        inputs << std::format("{},{},{}\n", time, 0.0, 0.0);
      }

      time += problem.dt().value(0, k);
    }
  }
}
