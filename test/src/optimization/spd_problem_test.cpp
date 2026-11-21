// Copyright (c) Sleipnir contributors

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/manifold.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

namespace {

/// Adds an iteration callback that checks the n×n SPD block at offset stays
/// positive definite.
template <typename T>
void check_iterates_positive_definite(slp::Problem<T>& problem, int offset,
                                      int n) {
  problem.add_callback([=](const slp::IterationInfo<T>& info) {
    Eigen::MatrixX<T> P{n, n};
    int index = offset;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        P(i, j) = info.x[index];
        P(j, i) = info.x[index];
        ++index;
      }
    }
    CHECK(Eigen::LLT<Eigen::MatrixX<T>>{P}.info() == Eigen::Success);
  });
}

/// Returns the determinant of a 3×3 VariableMatrix.
template <typename T>
slp::Variable<T> det3(const slp::VariableMatrix<T>& P) {
  return P[0, 0] * (P[1, 1] * P[2, 2] - P[1, 2] * P[2, 1]) -
         P[0, 1] * (P[1, 0] * P[2, 2] - P[1, 2] * P[2, 0]) +
         P[0, 2] * (P[1, 0] * P[2, 1] - P[1, 1] * P[2, 0]);
}

}  // namespace

TEMPLATE_TEST_CASE("Problem - SPD log-det maximum likelihood",
                   "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  // The Gaussian maximum likelihood precision matrix for sample covariance S
  // minimizes tr(SP) − log det P, and it's P = S⁻¹. log det P is undefined
  // outside the positive definite matrices, so the iterates must stay there.
  const Eigen::Matrix3<T> S{
      {T(4), T(1), T(0.5)}, {T(1), T(2), T(-0.3)}, {T(0.5), T(-0.3), T(0.5)}};

  slp::Problem<T> problem;
  auto P = problem.decision_variable(slp::spd, 3);
  problem.minimize((S * P).trace() - slp::log(det3(P)));
  check_iterates_positive_definite(problem, 0, 3);

  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  Eigen::Matrix3<T> P_value = P.value();
  CHECK_THAT((P_value - S.inverse()).template lpNorm<Eigen::Infinity>(),
             WithinAbs(T(0), solution_tolerance<T>(10)));
}

TEMPLATE_TEST_CASE("Problem - SPD structured LQR", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  // Discrete LQR with u = −Kx and K's last column fixed at zero minimizes
  //
  //   J = tr(PW)
  //
  // where P solves the Lyapunov equation
  //
  //   P = Q + KᵀRK + (A − BK)ᵀP(A − BK)
  //
  // A positive definite solution exists only if A − BK is stable, so making P
  // an SPD decision variable constrains K to stabilize the system.
  //
  // The system is a triple integrator with a first-order lag on the last
  // state, discretized with dt = 0.1 s. The reference solution is from the
  // Levine–Athans iteration polished with Newton's method.
  const Eigen::Matrix3<T> A{{T(1), T(0.1), T(0.00426122638850534)},
                            {T(0), T(1), T(0.07869386805747332)},
                            {T(0), T(0), T(0.6065306597126334)}};
  const Eigen::Vector3<T> B{T(0.00073877361149466), T(0.02130613194252668),
                            T(0.3934693402873665)};
  const Eigen::Matrix3<T> Q =
      Eigen::Vector3<T>{T(1), T(0.1), T(0)}.asDiagonal();
  const T R{0.1};

  slp::Problem<T> problem;
  auto P = problem.decision_variable(slp::spd, 3);
  auto F = problem.decision_variable(1, 2);

  // Warm start from the full LQR gains with the last column dropped
  F[0].set_value(T(2.7790344855481206));
  F[1].set_value(T(3.0040260596389445));

  slp::VariableMatrix<T> K{{F[0], F[1], slp::Variable<T>{T(0)}}};
  auto A_cl = A - B * K;
  auto lyapunov = Q + K.T() * R * K + A_cl.T() * P * A_cl - P;

  // The residual is symmetric, so only its lower triangle is constrained
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j <= i; ++j) {
      problem.subject_to(lyapunov[i, j] == T(0));
    }
  }

  // W = I
  problem.minimize(P.trace());
  check_iterates_positive_definite(problem, 0, 3);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(F.value(0),
             WithinAbs(T(1.9140313971107417), solution_tolerance<T>(10)));
  CHECK_THAT(F.value(1),
             WithinAbs(T(2.2849602346968894), solution_tolerance<T>(10)));
  CHECK_THAT(Eigen::Matrix3<T>{P.value()}.trace(),
             WithinAbs(T(15.71201670662753), solution_tolerance<T>(10)));
}

TEMPLATE_TEST_CASE("Problem - SPD discrete algebraic Riccati equation",
                   "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  // The DARE
  //
  //   AᵀPA − P − AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q = 0
  //
  // has several symmetric solutions, but with (A, B) stabilizable and
  // (A, Q^½) detectable, only the stabilizing one is positive semidefinite,
  // so the solver finds it from any positive definite initial guess. From
  // P₀ = 0.01I, Newton's method on symmetric P converges to an indefinite
  // solution instead.
  //
  // The system is the one from the structured LQR test with a second input.
  // The reference solution is from scipy.linalg.solve_discrete_are().
  const Eigen::Matrix3<T> A{{T(1), T(0.1), T(0.00426122638850534)},
                            {T(0), T(1), T(0.07869386805747332)},
                            {T(0), T(0), T(0.6065306597126334)}};
  const Eigen::Matrix<T, 3, 2> B{{T(0.00073877361149466), T(0)},
                                 {T(0.02130613194252668), T(0.1)},
                                 {T(0.3934693402873665), T(0)}};
  const Eigen::Matrix3<T> Q =
      Eigen::Vector3<T>{T(1), T(0.1), T(0)}.asDiagonal();
  const Eigen::Matrix2<T> R = Eigen::Vector2<T>{T(0.1), T(0.2)}.asDiagonal();
  const Eigen::Matrix3<T> P_expected{
      {T(9.576584400466913), T(3.614691854994743), T(0.44953971039753926)},
      {T(3.614691854994743), T(2.9744755586284817), T(0.4238989679269632)},
      {T(0.44953971039753926), T(0.4238989679269632), T(0.06596987193320486)}};

  auto P_0 = GENERATE(0.01, 1.0, 100.0);

  slp::Problem<T> problem;
  auto P = problem.decision_variable(slp::spd, 3);
  P.set_value((T(P_0) * Eigen::Matrix3<T>::Identity()).eval());

  auto residual =
      A.transpose() * P * A - P -
      A.transpose() * P * B *
          slp::solve(R + B.transpose() * P * B, B.transpose() * P * A) +
      Q;

  // The residual is symmetric, so only its lower triangle is constrained
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j <= i; ++j) {
      problem.subject_to(residual[i, j] == T(0));
    }
  }
  check_iterates_positive_definite(problem, 0, 3);

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT((Eigen::Matrix3<T>{P.value()} - P_expected)
                 .template lpNorm<Eigen::Infinity>(),
             WithinAbs(T(0), solution_tolerance<T>(10)));
}
