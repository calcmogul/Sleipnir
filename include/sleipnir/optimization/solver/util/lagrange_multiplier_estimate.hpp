// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <utility>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/util/append_as_triplets.hpp"

namespace slp {

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
  return Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>>{A_e *
                                                            A_e.transpose()}
      .solve(A_e * g);
}

/// Estimates Lagrange multipliers for interior-point method.
///
/// @tparam Scalar Scalar type.
/// @param g Gradient of the cost function ∇f.
/// @param A_i The problem's inequality constraint Jacobian Aᵢ(x) evaluated at
///     the current iterate.
/// @param s Inequality constraint slack variables.
/// @param μ Barrier parameter.
template <typename Scalar>
Eigen::Vector<Scalar, Eigen::Dynamic> lagrange_multiplier_estimate(
    const Eigen::SparseVector<Scalar>& g,
    const Eigen::SparseMatrix<Scalar>& A_i,
    const Eigen::Vector<Scalar, Eigen::Dynamic>& s, Scalar μ) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  // Lagrange multiplier estimates
  //
  //   ∇f − Aᵢᵀz = 0
  //   Sz − μe = 0
  //
  //   Aᵢᵀz = ∇f
  //   −Sz = −μe
  //
  //   [Aᵢᵀ][z] = [ ∇f]
  //   [−S ]      [−μe]
  //
  //   [Aᵢ  −S]ᵀ[z] = [ ∇f]
  //                  [−μe]
  //
  // Let Â = [Aᵢ  −S]
  //
  //   Âᵀz = [ ∇f]
  //         [−μe]
  //
  //   z = (ÂÂᵀ)⁻¹Â[ ∇f]
  //               [−μe]

  gch::small_vector<Eigen::Triplet<Scalar>> triplets;

  // Â = [Aᵢ  −S]
  triplets.reserve(A_i.nonZeros() + s.rows());
  append_as_triplets(triplets, 0, 0, {A_i});
  append_diagonal_as_triplets(triplets, 0, A_i.cols(), (-s).eval());
  SparseMatrix A_hat{A_i.rows(), A_i.cols() + s.rows()};
  A_hat.setFromSortedTriplets(triplets.begin(), triplets.end());

  // lhs = ÂÂᵀ
  SparseMatrix lhs = A_hat * A_hat.transpose();

  // rhs = Â[ ∇f]
  //        [−μe]
  DenseVector rhs_temp{g.rows() + s.rows()};
  rhs_temp.segment(0, g.rows()) = g;
  rhs_temp.segment(g.rows(), s.rows()).setConstant(-μ);
  DenseVector rhs = A_hat * rhs_temp;

  Eigen::SimplicialLDLT<SparseMatrix> yz_estimator{lhs};
  DenseVector z = yz_estimator.solve(rhs);

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

  return std::move(z);
}

}  // namespace slp
