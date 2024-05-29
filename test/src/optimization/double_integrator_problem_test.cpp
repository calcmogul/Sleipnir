// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Problem - Double integrator", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::abs;

  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<T> TOTAL_TIME{T(3.5)};
  constexpr std::chrono::duration<T> dt{T(0.005)};
  constexpr int N = static_cast<int>(TOTAL_TIME / dt);

  constexpr T r(2);  // m

  slp::Problem<T> problem;

  // 2x1 state vector with N + 1 timesteps (includes last state)
  auto X = problem.decision_variable(2, N + 1);

  // 1x1 input vector with N timesteps (input at last state doesn't matter)
  auto U = problem.decision_variable(1, N);

  // Kinematics constraint assuming constant acceleration between timesteps
  for (int k = 0; k < N; ++k) {
    constexpr T t = dt.count();
    auto p_k1 = X[0, k + 1];
    auto v_k1 = X[1, k + 1];
    auto p_k = X[0, k];
    auto v_k = X[1, k];
    auto a_k = U[0, k];

    // pₖ₊₁ = pₖ + vₖt + 1/2aₖt²
    problem.subject_to(p_k1 == p_k + v_k * t + 0.5 * a_k * t * t);

    // vₖ₊₁ = vₖ + aₖt
    problem.subject_to(v_k1 == v_k + a_k * t);
  }

  // Start and end at rest
  problem.subject_to(X.col(0) == Eigen::Matrix<T, 2, 1>{{T(0)}, {T(0)}});
  problem.subject_to(X.col(N) == Eigen::Matrix<T, 2, 1>{{r}, {T(0)}});

  // Limit velocity
  problem.subject_to(slp::bounds(T(-1), X.row(1), T(1)));

  // Limit acceleration
  problem.subject_to(slp::bounds(T(-1), U, T(1)));

  // Cost function - minimize position error
  slp::Variable J = T(0);
  for (int k = 0; k < N + 1; ++k) {
    J += pow(r - X[0, k], 2);
  }
  problem.minimize(J);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  REQUIRE(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  Eigen::Matrix<T, 2, 2> A{{T(1), dt.count()}, {T(0), T(1)}};
  Eigen::Matrix<T, 2, 1> B{T(0.5) * dt.count() * dt.count(), dt.count()};

  // Verify initial state
  CHECK_THAT(X.value(0, 0), WithinAbs(T(0), T(1e-8)));
  CHECK_THAT(X.value(1, 0), WithinAbs(T(0), T(1e-8)));

  // Verify solution
  Eigen::Matrix<T, 2, 1> x{T(0), T(0)};
  Eigen::Matrix<T, 1, 1> u{T(0)};
  for (int k = 0; k < N; ++k) {
    // Verify state
    CHECK_THAT(X.value(0, k), WithinAbs(x[0], T(1e-2)));
    CHECK_THAT(X.value(1, k), WithinAbs(x[1], T(1e-2)));

    // Determine expected input for this timestep
    if (T(k) * dt < std::chrono::duration<T>{T(1)}) {
      // Accelerate
      u[0] = T(1);
    } else if (T(k) * dt < std::chrono::duration<T>{T(2.05)}) {
      // Maintain speed
      u[0] = T(0);
    } else if (T(k) * dt < std::chrono::duration<T>{T(3.275)}) {
      // Decelerate
      u[0] = T(-1);
    } else {
      // Accelerate
      u[0] = T(1);
    }

    // Verify input
    if (k > 0 && k < N - 1 &&
        abs(U.value(0, k - 1) - U.value(0, k + 1)) >= T(1.0 - 1e-2)) {
      // If control input is transitioning between -1, 0, or 1, ensure it's
      // within (-1, 1)
      CHECK(U.value(0, k) >= T(-1));
      CHECK(U.value(0, k) <= T(1));
    } else {
      CHECK_THAT(U.value(0, k), WithinAbs(u[0], T(1e-4)));
    }

    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = A * x + B * u;
  }

  // Verify final state
  CHECK_THAT(X.value(0, N), WithinAbs(r, T(1e-8)));
  CHECK_THAT(X.value(1, N), WithinAbs(T(0), T(1e-8)));

  // Log states for offline viewing
  std::ofstream states{"Problem - Double integrator states.csv"};
  if (states.is_open()) {
    states << "Time (s),Position (m),Velocity (m/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{},{}\n", T(k) * dt.count(), X.value(0, k),
                            X.value(1, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Problem - Double integrator inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Acceleration (m/s²)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", T(k) * dt.count(), U.value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", T(k) * dt.count(), T(0));
      }
    }
  }
}
