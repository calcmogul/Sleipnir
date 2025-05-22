// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/util/all_finite.hpp"

namespace slp {

/// The cost function gradient and equality constraint Jacobian evaluated at a
/// Sequential Quadratic Programming iterate's decision variables.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SQPFirstDerivatives {
  /// The cost function gradient ∇f(x).
  Eigen::SparseVector<Scalar> g;

  /// The equality constraint Jacobian Aₑ(x).
  Eigen::SparseMatrix<Scalar> A_e;

  /// Returns whether every value is finite.
  ///
  /// @return Whether every value is finite.
  bool all_finite() const { return slp::all_finite(g) && slp::all_finite(A_e); }
};

/// The cost function gradient and inequality constraint Jacobian evaluated at
/// an interior-point method iterate's decision variables.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct IPMFirstDerivatives {
  /// The cost function gradient ∇f(x).
  Eigen::SparseVector<Scalar> g;

  /// The inequality constraint Jacobian Aᵢ(x).
  Eigen::SparseMatrix<Scalar> A_i;

  /// Returns whether every value is finite.
  ///
  /// @return Whether every value is finite.
  bool all_finite() const { return slp::all_finite(g) && slp::all_finite(A_i); }
};

}  // namespace slp
