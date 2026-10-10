// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

namespace slp {

/// Estimates Lagrange multipliers for SQP.
///
/// @tparam Scalar Scalar type.
/// @param g Cost function gradient ∇f.
/// @param A_e Equality constraint Jacobian Aₑ(x).
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

}  // namespace slp
