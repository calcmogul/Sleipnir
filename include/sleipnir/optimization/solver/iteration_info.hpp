// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace slp {

/// Solver iteration information exposed to an iteration callback.
///
/// If some decision variables lie on a manifold (e.g., from
/// Problem::decision_variable(so3)), x is the point in the ambient space, and
/// the derivatives are with respect to the manifold's tangent space at x, so
/// they have fewer columns than x has rows.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct IterationInfo {
  /// The solver iteration.
  int iteration;

  /// The decision variables.
  const Eigen::VectorX<Scalar>& x;

  /// The inequality constraint slack variables.
  const Eigen::VectorX<Scalar>& s;

  /// The equality constraint dual variables.
  const Eigen::VectorX<Scalar>& y;

  /// The inequality constraint dual variables.
  const Eigen::VectorX<Scalar>& z;

  /// The gradient of the cost function.
  const Eigen::SparseVector<Scalar>& g;

  /// The Hessian of the Lagrangian.
  const Eigen::SparseMatrix<Scalar>& H;

  /// The equality constraint Jacobian.
  const Eigen::SparseMatrix<Scalar>& A_e;

  /// The inequality constraint Jacobian.
  const Eigen::SparseMatrix<Scalar>& A_i;
};

}  // namespace slp
