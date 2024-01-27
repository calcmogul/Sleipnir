// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

// See docs/algorithms.md#Works_cited for citation definitions

namespace sleipnir {

/**
 * Returns the KKT error for the interior-point method.
 *
 * @param g Gradient of the cost function ∇f.
 * @param A_i The problem's inequality constraint Jacobian Aᵢ(x) evaluated at
 *   the current iterate.
 * @param c_i The problem's inequality constraints cᵢ(x) evaluated at the
 *   current iterate.
 * @param s Inequality constraint slack variables.
 * @param y Inequality constraint dual variables.
 * @param μ Barrier parameter.
 */
inline double KKTError(const Eigen::VectorXd& g,
                       const Eigen::SparseMatrix<double>& A_i,
                       const Eigen::VectorXd& c_i, const Eigen::VectorXd& s,
                       const Eigen::VectorXd& y, double μ) {
  // Compute the KKT error as the 1-norm of the KKT conditions from equations
  // (19.5a) through (19.5d) of [1].
  //
  //   ∇f − Aᵢᵀy = 0
  //   Sy − μe = 0
  //   cᵢ − s = 0

  const auto S = s.asDiagonal();
  const Eigen::VectorXd e = Eigen::VectorXd::Ones(s.rows());

  return (g - A_i.transpose() * y).lpNorm<1>() + (S * y - μ * e).lpNorm<1>() +
         (c_i - s).lpNorm<1>();
}

}  // namespace sleipnir
