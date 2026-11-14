// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <utility>

#include <Eigen/Core>
#include <Eigen/LU>  // IWYU pragma: keep
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/util/pool.hpp>
#include <sleipnir/util/scope_exit.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

/// Discretizes the given continuous A and B matrices.
///
/// @tparam T Scalar type.
/// @tparam States Number of states.
/// @tparam Inputs Number of inputs.
/// @param A_c Continuous system matrix.
/// @param B_c Continuous input matrix.
/// @param dt Discretization timestep.
/// @return Discrete system and input matrices.
template <typename T, int States, int Inputs>
std::pair<Eigen::Matrix<T, States, States>, Eigen::Matrix<T, States, Inputs>>
discretize_ab(const Eigen::Matrix<T, States, States>& A_c,
              const Eigen::Matrix<T, States, Inputs>& B_c, T dt) {
  // M = [A_c  B_c]
  //     [ 0    0 ]
  Eigen::Matrix<T, States + Inputs, States + Inputs> M;
  M.template block<States, States>(0, 0) = A_c;
  M.template block<States, Inputs>(0, States) = B_c;
  M.template block<Inputs, States + Inputs>(States, 0).setZero();

  // ϕ = eᴹᵀ = [A_d  B_d]
  //           [ 0    I ]
  Eigen::Matrix<T, States + Inputs, States + Inputs> ϕ =
      slp::VariableMatrix<T>{M * dt}.exp().value();

  return {ϕ.template block<States, States>(0, 0),
          ϕ.template block<States, Inputs>(0, States)};
}

/// Returns the input trajectory that minimizes Σ(r − pₖ)² for k ∈ [0, N]
/// subject to double integrator kinematics with timestep dt, starting and
/// ending at rest at 0 and r respectively, |vₖ| ≤ 1, and |aₖ| ≤ 1.
///
/// @tparam T Scalar type.
/// @param r Reference position.
/// @param dt Timestep.
/// @param N Number of timesteps.
template <typename T>
Eigen::RowVectorX<T> optimal_double_integrator_inputs(T r, T dt, int N) {
  using std::floor;
  using std::round;
  using std::sqrt;

  T v_max = T(1);
  T a_max = T(1);
  T t_n = T(N) * dt;

  // The optimal input accelerates to the velocity limit, cruises, then
  // decelerates past r and accelerates back to rest at r. In continuous time,
  // let vₘₐₓ = 1 and aₘₐₓ = 1 be the velocity and acceleration limits and
  // tₙ = N dt be the total time. A phase with input u and duration t changes
  // the velocity by ut, and if it starts at velocity v₀, it covers
  // v₀t + ut²/2. The phases are
  //
  //   Phase      │ Duration │ u     │ Final v │ Distance
  //   ───────────┼──────────┼───────┼─────────┼───────────────────
  //   Accelerate │ t₁       │ aₘₐₓ  │ vₘₐₓ    │ aₘₐₓt₁²/2
  //   Cruise     │ t₂       │ 0     │ vₘₐₓ    │ vₘₐₓt₂
  //   Decelerate │ t₃       │ −aₘₐₓ │ −aₘₐₓt₄ │ vₘₐₓt₃ − aₘₐₓt₃²/2
  //   Accelerate │ t₄       │ aₘₐₓ  │ 0       │ −aₘₐₓt₄²/2
  //
  // where t₁ = vₘₐₓ/aₘₐₓ. The phases fill the total time. Deceleration
  // starts at vₘₐₓ and the final acceleration ends at rest, so the velocity
  // between them is both vₘₐₓ − aₘₐₓt₃ and −aₘₐₓt₄. Therefore,
  //
  //   t₁ + t₂ + t₃ + t₄ = tₙ
  //   vₘₐₓ − aₘₐₓt₃ = −aₘₐₓt₄
  //
  // Dividing the second equation by aₘₐₓ gives t₄ = t₃ − t₁, and substituting
  // that into the first gives t₂ = tₙ − 2t₃. Ending at r requires the
  // distances to sum to r. Dividing that sum by aₘₐₓ and substituting
  // vₘₐₓ/aₘₐₓ = t₁ gives
  //
  //   t₁²/2 + t₁t₂ + t₁t₃ − (t₃² + t₄²)/2 = r/aₘₐₓ
  //   t₁²/2 + t₁(tₙ − 2t₃) + t₁t₃ − (t₃² + (t₃ − t₁)²)/2 = r/aₘₐₓ
  //   t₁²/2 + t₁tₙ − t₁t₃ − t₃² + t₁t₃ − t₁²/2 = r/aₘₐₓ
  //   t₁tₙ − t₃² = r/aₘₐₓ
  //   t₃ = √(t₁tₙ − r/aₘₐₓ)
  //
  // so the phase durations are
  //
  //   t₁ = vₘₐₓ/aₘₐₓ
  //   t₂ = tₙ − 2t₃
  //   t₃ = √(t₁tₙ − r/aₘₐₓ)
  //   t₄ = t₃ − t₁
  T t_1 = v_max / a_max;
  T t_3 = sqrt(t_1 * t_n - r / a_max);
  T t_2 = t_n - T(2) * t_3;
  T t_4 = t_3 - t_1;

  // Each phase starts when the previous one ends, and the last ends at tₙ, so
  // the phases start at 0, t₁, t₁ + t₂, and tₙ − t₄.
  //
  // The kinematics are exact for piecewise constant acceleration, so the
  // discrete optimum switches at the same times. Timestep k applies uₖ over
  // [k dt, (k + 1) dt), so cruising, deceleration, and the final acceleration
  // start in timesteps
  //
  //   k₂ = t₁/dt
  //   k₃ = ⌊(t₁ + t₂)/dt⌋
  //   k₄ = ⌊(tₙ − t₄)/dt⌋
  //
  // t₁/dt is assumed to be an integer, so cruising starts on a timestep
  // boundary.
  int k_2 = static_cast<int>(round(t_1 / dt));
  int k_3 = static_cast<int>(floor((t_1 + t_2) / dt));
  int k_4 = static_cast<int>(floor((t_n - t_4) / dt));

  // Every timestep's input is its phase's u, except in timesteps k₃ and k₄.
  // Deceleration and the final acceleration generally start partway through
  // those timesteps, so they take fractional inputs, which start at zero and
  // are solved for below.
  Eigen::RowVectorX<T> U = Eigen::RowVectorX<T>::Zero(N);
  U.segment(0, k_2).setConstant(a_max);
  U.segment(k_3 + 1, k_4 - k_3 - 1).setConstant(-a_max);
  U.segment(k_4 + 1, N - 1 - k_4).setConstant(a_max);

  // Continuous dynamics ẋ = Ax + Bu with state x = [p  v]ᵀ and input u = a
  Eigen::Matrix2<T> A{{T(0), T(1)}, {T(0), T(0)}};
  Eigen::Matrix<T, 2, 1> B{T(0), T(1)};
  auto [A_d, B_d] = discretize_ab(A, B, dt);

  // Returns the state at timestep N reached from rest at 0 with the inputs
  auto final_state = [&](const Eigen::RowVectorX<T>& inputs) {
    Eigen::Vector2<T> x = Eigen::Vector2<T>::Zero();
    for (int k = 0; k < N; ++k) {
      x = A_d * x + B_d * inputs(k);
    }
    return x;
  };

  // The discrete dynamics xₖ₊₁ = A_d xₖ + B_d uₖ are exact for piecewise
  // constant inputs and linear in the inputs, so the final state is
  //
  //   xₙ = x_fixed + c₃uₖ₃ + c₄uₖ₄
  //
  // where x_fixed is the final state from the fixed inputs alone, and c₃ and
  // c₄ are the final states from unit inputs at timesteps k₃ and k₄. The
  // fractional inputs are still zero in U, so x_fixed is the final state
  // from U. Ending at rest at r requires
  //
  //           [uₖ₃]   [r]
  //   [c₃  c₄][uₖ₄] = [0] − x_fixed
  //
  // which is a 2×2 linear system for the fractional inputs.
  Eigen::Vector2<T> x_fixed = final_state(U);
  Eigen::Matrix2<T> C;
  C << final_state(Eigen::RowVectorX<T>::Unit(N, k_3)),
      final_state(Eigen::RowVectorX<T>::Unit(N, k_4));
  Eigen::Vector2<T> u = C.lu().solve(Eigen::Vector2<T>{r, T(0)} - x_fixed);
  U(k_3) = u(0);
  U(k_4) = u(1);

  return U;
}

TEMPLATE_TEST_CASE("Problem - Double integrator", "[Problem]", SCALAR_TYPES) {
  using T = TestType;

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
  problem.subject_to(X.col(0) == Eigen::Vector2<T>{{T(0)}, {T(0)}});
  problem.subject_to(X.col(N) == Eigen::Vector2<T>{{r}, {T(0)}});

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

  Eigen::Matrix2<T> A{{T(0), T(1)}, {T(0), T(0)}};
  Eigen::Matrix<T, 2, 1> B{T(0), T(1)};
  auto [A_d, B_d] = discretize_ab(A, B, dt.count());

  auto U_ref = optimal_double_integrator_inputs(r, dt.count(), N);

  // Only position is in the cost, and vₖ affects it only through pₖ₊₁, so
  // velocity is determined less precisely than position. uₖ affects only vₖ₊₁
  // through the dt uₖ it adds, so U is determined 1/dt times less precisely
  // than velocity.
  const T p_tolerance = solution_tolerance<T>(200);
  const T v_tolerance = solution_tolerance<T>(1000);
  const T u_tolerance = v_tolerance / dt.count();

  // Verify initial state
  CHECK_THAT(X.value(0, 0), WithinAbs(T(0), solution_tolerance<T>(1)));
  CHECK_THAT(X.value(1, 0), WithinAbs(T(0), solution_tolerance<T>(1)));

  // Verify solution
  Eigen::Vector2<T> x_ref{T(0), T(0)};
  for (int k = 0; k < N + 1; ++k) {
    INFO(std::format("  k = {}", k));

    CHECK_THAT(X.value(0, k), WithinAbs(x_ref[0], p_tolerance));
    CHECK_THAT(X.value(1, k), WithinAbs(x_ref[1], v_tolerance));

    if (k < N) {
      CHECK_THAT(U.value(0, k), WithinAbs(U_ref(k), u_tolerance));
      x_ref = A_d * x_ref + B_d * U_ref(k);
    }
  }

  // Verify final state
  CHECK_THAT(X.value(0, N), WithinAbs(r, solution_tolerance<T>(1)));
  CHECK_THAT(X.value(1, N), WithinAbs(T(0), solution_tolerance<T>(1)));

  // Log states for offline viewing
  std::ofstream states{"Problem - Double integrator states.csv"};
  if (states.is_open()) {
    states << "Time (s),Position (m),Velocity (m/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{},{}\n", T(k) * dt.count(), X.value(0, k),
                            X.value(1, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Problem - Double integrator inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Acceleration (m/s²)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{}\n", T(k) * dt.count(), U.value(0, k));
      } else {
        inputs << std::format("{},{}\n", T(k) * dt.count(), T(0));
      }
    }
  }
}
