// Copyright (c) Sleipnir contributors

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/spd_matrix.hpp>
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

/// Returns the solution P of the discrete algebraic Riccati equation
///
///   AᵀPA − P − AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q = 0
template <int States, int Inputs>
Eigen::Matrix<double, States, States> dare(
    const Eigen::Matrix<double, States, States>& A,
    const Eigen::Matrix<double, States, Inputs>& B,
    const Eigen::Matrix<double, States, States>& Q,
    const Eigen::Matrix<double, Inputs, Inputs>& R) {
  // [1] E. K.-W. Chu, H.-Y. Fan, W.-W. Lin & C.-S. Wang
  //     "Structure-Preserving Algorithms for Periodic Discrete-Time
  //     Algebraic Riccati Equations",
  //     International Journal of Control, 77:8, 767-788, 2004.
  //     DOI: 10.1080/00207170410001714988
  //
  // Implements SDA algorithm on p. 5 of [1] (A₀, G₀, H₀ are from (4)).
  using StateMatrix = Eigen::Matrix<double, States, States>;

  StateMatrix A_k = A;
  StateMatrix G_k = B * R.llt().solve(B.transpose());
  StateMatrix H_k;
  StateMatrix H_k1 = Q;

  do {
    H_k = H_k1;

    StateMatrix W = StateMatrix::Identity() + G_k * H_k;

    auto W_solver = W.lu();
    StateMatrix V_1 = W_solver.solve(A_k);
    StateMatrix V_2 = W_solver.solve(G_k.transpose()).transpose();

    G_k += A_k * V_2 * A_k.transpose();
    H_k1 = H_k + V_1.transpose() * H_k * A_k;
    A_k *= V_1;
  } while ((H_k1 - H_k).norm() > 1e-10 * H_k1.norm());

  return H_k1;
}

/// Returns the LQR gain K for u = −Kx.
template <int States, int Inputs>
Eigen::Matrix<double, Inputs, States> lqr(
    const Eigen::Matrix<double, States, States>& A,
    const Eigen::Matrix<double, States, Inputs>& B,
    const Eigen::Matrix<double, States, States>& Q,
    const Eigen::Matrix<double, Inputs, Inputs>& R) {
  Eigen::Matrix<double, States, States> P = dare<States, Inputs>(A, B, Q, R);
  return (B.transpose() * P * B + R).llt().solve(B.transpose() * P * A);
}

/// Returns the solution X of the discrete Lyapunov equation
///
///   AXAᵀ − X + Q = 0
template <int States>
Eigen::Matrix<double, States, States> dlyap(
    const Eigen::Matrix<double, States, States>& A,
    const Eigen::Matrix<double, States, States>& Q) {
  // [1] R. A. Smith
  //     "Matrix Equation XA + BX = C",
  //     SIAM Journal on Applied Mathematics, 16:1, 198-201, 1968.
  //     DOI: 10.1137/0116017
  //
  // Implements iteration (5) on p. 199 of [1] for X − UXV = W (2) with
  // U = A, V = Aᵀ, and W = Q.
  using StateMatrix = Eigen::Matrix<double, States, States>;

  StateMatrix A_k = A;
  StateMatrix X_k;
  StateMatrix X_k1 = Q;

  do {
    X_k = X_k1;
    X_k1 = X_k + A_k * X_k * A_k.transpose();
    A_k *= A_k;
  } while ((X_k1 - X_k).norm() > 1e-10 * X_k1.norm());

  return X_k1;
}

/// Returns the determinant of a 3×3 symmetric positive definite matrix.
template <typename T>
slp::Real<T> det3(const slp::SPDMatrix<T>& P) {
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
  auto P = problem.spd_decision_variable(3);
  problem.minimize((S * P).trace() - slp::log(det3(P)));
  check_iterates_positive_definite(problem, 0, 3);

  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  Eigen::Matrix3<T> P_value = P.value();
  CHECK_THAT((P_value - S.inverse()).template lpNorm<Eigen::Infinity>(),
             WithinAbs(T(0), solution_tolerance<T>(10)));
}

TEMPLATE_TEST_CASE("Problem - SPD parameter", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  // The sample covariance S is a parameter, so P = S⁻¹ follows it
  slp::Problem<T> problem;
  auto P = problem.spd_decision_variable(2);
  auto S = problem.spd_parameter(2);

  CHECK(S.value() == Eigen::Matrix2<T>::Identity());
  CHECK(S[1, 0].type() == slp::ExpressionType::PARAMETER);

  problem.minimize((slp::RealMatrix<T>{S} * slp::RealMatrix<T>{P}).trace() -
                   slp::log(P[0, 0] * P[1, 1] - P[1, 0] * P[0, 1]));

  for (const auto& value :
       {Eigen::Matrix2<T>{{T(4), T(1)}, {T(1), T(2)}},
        Eigen::Matrix2<T>{{T(0.5), T(-0.2)}, {T(-0.2), T(1)}}}) {
    S.set_value(value);
    CHECK(S.value() == value);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    Eigen::Matrix2<T> P_value = P.value();
    CHECK_THAT((P_value - value.inverse()).template lpNorm<Eigen::Infinity>(),
               WithinAbs(T(0), solution_tolerance<T>(10)));
  }
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
  //
  // F starts at the full LQR gains with the last column dropped, and P starts
  // at the solution of the Lyapunov equation for those gains. Starting P at
  // the identity instead leaves the initial iterate far from the constraints.
  const Eigen::Matrix3d A_d{{1.0, 0.1, 0.00426122638850534},
                            {0.0, 1.0, 0.07869386805747332},
                            {0.0, 0.0, 0.6065306597126334}};
  const Eigen::Vector3d B_d{0.00073877361149466, 0.02130613194252668,
                            0.3934693402873665};
  const Eigen::Matrix3d Q_d = Eigen::Vector3d{1.0, 0.1, 0.0}.asDiagonal();
  const Eigen::Matrix<double, 1, 1> R_d{0.1};

  Eigen::RowVector3d K_0 = lqr<3, 1>(A_d, B_d, Q_d, R_d);
  K_0[2] = 0.0;
  const Eigen::Matrix3d P_0 = dlyap<3>((A_d - B_d * K_0).transpose(),
                                       Q_d + K_0.transpose() * R_d * K_0);

  const Eigen::Matrix3<T> A = A_d.cast<T>();
  const Eigen::Vector3<T> B = B_d.cast<T>();
  const Eigen::Matrix3<T> Q = Q_d.cast<T>();
  const T R(R_d(0, 0));

  slp::Problem<T> problem;
  auto P = problem.spd_decision_variable(3);
  auto F = problem.real_decision_variable(1, 2);
  P.set_value(P_0.cast<T>().eval());
  F.set_value(K_0.leftCols<2>().cast<T>().eval());

  slp::RealMatrix<T> K{{F[0], F[1], slp::Real<T>{T(0)}}};
  auto A_cl = A - B * K;
  auto lyapunov = Q + K.T() * R * K + A_cl.T() * P * A_cl - P;

  problem.subject_to(lyapunov.symmetric_view() == T(0));

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
  auto P = problem.spd_decision_variable(3);
  P.set_value((T(P_0) * Eigen::Matrix3<T>::Identity()).eval());

  auto residual =
      A.transpose() * P * A - P -
      A.transpose() * P * B *
          slp::solve(R + B.transpose() * P * B, B.transpose() * P * A) +
      Q;

  problem.subject_to(residual.symmetric_view() == T(0));
  check_iterates_positive_definite(problem, 0, 3);

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT((Eigen::Matrix3<T>{P.value()} - P_expected)
                 .template lpNorm<Eigen::Infinity>(),
             WithinAbs(T(0), solution_tolerance<T>(10)));
}
