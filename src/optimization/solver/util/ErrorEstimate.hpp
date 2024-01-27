// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>

#include <Eigen/Core>
#include <Eigen/SparseCore>

// See docs/algorithms.md#Works_cited for citation definitions

namespace sleipnir {

/**
 * Returns the error estimate using the KKT conditions for the interior-point
 * method.
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
inline double ErrorEstimate(const Eigen::VectorXd& g,
                            const Eigen::SparseMatrix<double>& A_i,
                            const Eigen::VectorXd& c_i,
                            const Eigen::VectorXd& s, const Eigen::VectorXd& y,
                            double μ) {
  int numInequalityConstraints = A_i.rows();

  // Update the error estimate using the KKT conditions from equations (19.5a)
  // through (19.5d) of [1].
  //
  //   ∇f − Aᵢᵀy = 0
  //   Sy − μe = 0
  //   cᵢ − s = 0
  //
  // The error tolerance is the max of the following infinity norms scaled by
  // s_d and s_c (see equation (5) of [2]).
  //
  //   ‖∇f − Aₑᵀy − Aᵢᵀz‖_∞ / s_d
  //   ‖Sy − μe‖_∞ / s_c
  //   ‖cₑ‖_∞
  //   ‖cᵢ − s‖_∞

  // s_d = max(sₘₐₓ, ‖y‖₁ / n) / sₘₐₓ
  constexpr double s_max = 100.0;
  double s_d =
      std::max(s_max, y.lpNorm<1>() / numInequalityConstraints) / s_max;

  const auto S = s.asDiagonal();
  const Eigen::VectorXd e = Eigen::VectorXd::Ones(s.rows());

  return std::max({(g - A_i.transpose() * y).lpNorm<Eigen::Infinity>() / s_d,
                   (S * y - μ * e).lpNorm<Eigen::Infinity>() / s_d,
                   (c_i - s).lpNorm<Eigen::Infinity>()});
}

}  // namespace sleipnir
