// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <utility>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/util/Eigen/SparseCholesky"
#include "sleipnir/optimization/solver/util/append_as_triplets.hpp"

namespace slp {

/// Lagrange multiplier estimates.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct LagrangeMultiplierEstimate {
  /// Equality constraint dual estimate.
  Eigen::Vector<Scalar, Eigen::Dynamic> y;
  /// Inequality constraint dual estimate.
  Eigen::Vector<Scalar, Eigen::Dynamic> z;
};

/// Estimates Lagrange multipliers for SQP.
///
/// @tparam Scalar Scalar type.
/// @param g Gradient of the cost function ∇f.
/// @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
///     current iterate.
template <typename Scalar>
Eigen::Vector<Scalar, Eigen::Dynamic> lagrange_multiplier_estimate(
    const Eigen::SparseVector<Scalar>& g,
    const Eigen::SparseMatrix<Scalar>& A_e) {
  // Lagrange multiplier estimates
  //
  //   ∇f − Aₑᵀy = 0
  //   Aₑᵀy = ∇f
  //   y = (AₑAₑᵀ)⁻¹Aₑ∇f
  return Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>>{
      A_e * A_e.transpose(), A_e.rows()}
      .solve(A_e * g);
}

/// Estimates Lagrange multipliers for interior-point method.
///
/// @tparam Scalar Scalar type.
/// @param g Gradient of the cost function ∇f.
/// @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
///     current iterate.
/// @param A_i The problem's inequality constraint Jacobian Aᵢ(x) evaluated at
///     the current iterate.
/// @param s Inequality constraint slack variables.
/// @param μ Barrier parameter.
template <typename Scalar>
LagrangeMultiplierEstimate<Scalar> lagrange_multiplier_estimate(
    const Eigen::SparseVector<Scalar>& g,
    const Eigen::SparseMatrix<Scalar>& A_e,
    const Eigen::SparseMatrix<Scalar>& A_i,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& s, Scalar μ) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  // Lagrange multiplier estimates
  //
  //   ∇f − Aₑᵀy − Aᵢᵀz = 0
  //   Sz − μe = 0
  //
  //   Aₑᵀy + Aᵢᵀz = ∇f
  //   −Sz = −μe
  //
  //   [Aₑᵀ  Aᵢᵀ][y] = [ ∇f]
  //   [ 0   −S ][z]   [−μe]
  //
  //   [Aₑ   0]ᵀ[y] = [ ∇f]
  //   [Aᵢ  −S] [z]   [−μe]
  //
  // Let Â = [Aₑ   0]
  //         [Aᵢ  −S]
  //
  //   Âᵀ[y] = [ ∇f]
  //     [z]   [−μe]
  //
  //   [y] = (ÂÂᵀ)⁻¹Â[ ∇f]
  //   [z]           [−μe]

  gch::small_vector<Eigen::Triplet<Scalar>> triplets;

  // Â = [Aₑ   0]
  //     [Aᵢ  −S]
  triplets.reserve(A_e.nonZeros() + A_i.nonZeros() + s.rows());
  append_as_triplets(triplets, 0, 0, {A_e, A_i});
  append_diagonal_as_triplets(triplets, A_e.rows(), A_i.cols(), (-s).eval());
  SparseMatrix A_hat{A_e.rows() + A_i.rows(), A_e.cols() + s.rows()};
  A_hat.setFromSortedTriplets(triplets.begin(), triplets.end());

  // lhs = ÂÂᵀ
  SparseMatrix lhs = A_hat * A_hat.transpose();

  // rhs = Â[ ∇f]
  //        [−μe]
  DenseVector rhs_temp{g.rows() + s.rows()};
  rhs_temp.segment(0, g.rows()) = g;
  rhs_temp.segment(g.rows(), s.rows()).setConstant(-μ);
  DenseVector rhs = A_hat * rhs_temp;

  Eigen::SimplicialLDLT<SparseMatrix> yz_estimator{lhs, lhs.rows()};
  DenseVector sol = yz_estimator.solve(rhs);
  DenseVector y = sol.segment(0, A_e.rows());
  DenseVector z = sol.segment(A_e.rows(), s.rows());

  // A requirement for the convergence proof is that the primal-dual barrier
  // term Hessian Σₖ₊₁ does not deviate arbitrarily much from the primal barrier
  // term Hessian μSₖ₊₁⁻².
  //
  //   Σₖ₊₁ = μSₖ₊₁⁻²
  //   Sₖ₊₁⁻¹Zₖ₊₁ = μSₖ₊₁⁻²
  //   Zₖ₊₁ = μSₖ₊₁⁻¹
  //
  // We ensure this by resetting
  //
  //   zₖ₊₁ = clamp(zₖ₊₁, 1/κ_Σ μ/sₖ₊₁, κ_Σ μ/sₖ₊₁)
  //
  // for some fixed κ_Σ ≥ 1 after each step. See equation (16) of [2].
  for (int row = 0; row < z.rows(); ++row) {
    constexpr Scalar κ_Σ(1e10);
    z[row] = std::clamp(z[row], Scalar(1) / κ_Σ * μ / s[row], κ_Σ * μ / s[row]);
  }

  return {std::move(y), std::move(z)};
}

}  // namespace slp
