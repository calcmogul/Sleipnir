// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>

#include <Eigen/Core>
#include <Eigen/SparseCore>

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

/**
 * Returns the error estimate using the KKT conditions for Newton's method.
 *
 * @param g Gradient of the cost function ∇f.
 */
inline double error_estimate(const Eigen::VectorXd& g) {
  // Update the error estimate using the KKT conditions from equations (19.5a)
  // through (19.5d) of [1].
  //
  //   ∇f = 0

  return g.lpNorm<Eigen::Infinity>();
}

/**
 * Returns the error estimate using the KKT conditions for SQP.
 *
 * @param g Gradient of the cost function ∇f.
 * @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
 *   current iterate.
 * @param c_e The problem's equality constraints cₑ(x) evaluated at the current
 *   iterate.
 * @param y Equality constraint dual variables.
 */
inline double error_estimate(const Eigen::VectorXd& g,
                             const Eigen::SparseMatrix<double>& A_e,
                             const Eigen::VectorXd& c_e,
                             const Eigen::VectorXd& y) {
  // Update the error estimate using the KKT conditions from equations (19.5a)
  // through (19.5d) of [1].
  //
  //   ∇f − Aₑᵀy = 0
  //   cₑ = 0
  //
  // The error tolerance is the max of the following infinity norms scaled by
  // s_d (see equation (5) of [2]).
  //
  //   ‖∇f − Aₑᵀy‖_∞ / s_d
  //   ‖cₑ‖_∞

  // s_d = max(sₘₐₓ, ‖y‖₁ / m) / sₘₐₓ
  constexpr double s_max = 100.0;
  double s_d = std::max(s_max, y.lpNorm<1>() / y.rows()) / s_max;

  return std::max({(g - A_e.transpose() * y).lpNorm<Eigen::Infinity>() / s_d,
                   c_e.lpNorm<Eigen::Infinity>()});
}

/**
 * Returns the error estimate using the KKT conditions for the interior-point
 * method.
 *
 * @param g Gradient of the cost function ∇f.
 * @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
 *   current iterate.
 * @param c_e The problem's equality constraints cₑ(x) evaluated at the current
 *   iterate.
 * @param A_i The problem's inequality constraint Jacobian Aᵢ(x) evaluated at
 *   the current iterate.
 * @param c_i The problem's inequality constraints cᵢ(x) evaluated at the
 *   current iterate.
 * @param y Equality constraint dual variables.
 * @param v Log-domain variables.
 * @param sqrt_μ Square root of the barrier parameter.
 */
inline double error_estimate(const Eigen::VectorXd& g,
                             const Eigen::SparseMatrix<double>& A_e,
                             const Eigen::VectorXd& c_e,
                             const Eigen::SparseMatrix<double>& A_i,
                             const Eigen::VectorXd& c_i,
                             const Eigen::VectorXd& y, const Eigen::VectorXd& v,
                             double sqrt_μ) {
  // Update the error estimate using the KKT conditions.
  //
  //   ∇f − Aₑᵀy − Aᵢᵀz = 0
  //   cₑ = 0
  //   cᵢ − s = 0
  //
  // where
  //
  //   s = √(μ)e⁻ᵛ
  //   z = √(μ)eᵛ
  //
  // The error tolerance is the max of the following infinity norms scaled by
  // s_d (see equation (5) of [2]).
  //
  //   ‖∇f − Aₑᵀy − Aᵢᵀz‖_∞ / s_d
  //   ‖cₑ‖_∞
  //   ‖cᵢ − s‖_∞

  Eigen::VectorXd s = sqrt_μ * (-v).array().exp().matrix();
  Eigen::VectorXd z = sqrt_μ * v.array().exp().matrix();

  // s_d = max(sₘₐₓ, (‖y‖₁ + ‖z‖₁) / (m + n)) / sₘₐₓ
  constexpr double s_max = 100.0;
  double s_d =
      std::max(s_max, (y.lpNorm<1>() + z.lpNorm<1>()) / (y.rows() + z.rows())) /
      s_max;

  return std::max({(g - A_e.transpose() * y - A_i.transpose() * z)
                           .lpNorm<Eigen::Infinity>() /
                       s_d,
                   c_e.lpNorm<Eigen::Infinity>(),
                   (c_i - s).lpNorm<Eigen::Infinity>()});
}

}  // namespace slp
