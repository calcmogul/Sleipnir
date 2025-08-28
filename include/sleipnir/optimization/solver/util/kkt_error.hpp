// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

/// Returns the KKT error for Newton's method.
///
/// @tparam Scalar Scalar type.
/// @param g Gradient of the cost function ∇f.
template <typename Scalar>
Scalar kkt_error(const Eigen::Vector<Scalar, Eigen::Dynamic>& g) {
  // Compute the KKT error as the 1-norm of the KKT conditions from equations
  // (19.5a) through (19.5d) of [1].
  //
  //   ∇f = 0

  return g.template lpNorm<1>();
}

/// Returns the KKT error for Sequential Quadratic Programming.
///
/// @tparam Scalar Scalar type.
/// @param g Gradient of the cost function ∇f.
/// @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
///     current iterate.
/// @param c_e The problem's equality constraints cₑ(x) evaluated at the current
///     iterate.
/// @param y Equality constraint dual variables.
template <typename Scalar>
Scalar kkt_error(const Eigen::Vector<Scalar, Eigen::Dynamic>& g,
                 const Eigen::SparseMatrix<Scalar>& A_e,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& c_e,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& y) {
  // Compute the KKT error as the 1-norm of the KKT conditions from equations
  // (19.5a) through (19.5d) of [1].
  //
  //   ∇f − Aₑᵀy = 0
  //   cₑ = 0

  return (g - A_e.transpose() * y).template lpNorm<1>() +
         c_e.template lpNorm<1>();
}

/// Returns the KKT error for the interior-point method.
///
/// @tparam Scalar Scalar type.
/// @param g Gradient of the cost function ∇f.
/// @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
///     current iterate.
/// @param c_e The problem's equality constraints cₑ(x) evaluated at the current
///     iterate.
/// @param A_i The problem's inequality constraint Jacobian Aᵢ(x) evaluated at
///     the current iterate.
/// @param c_i The problem's inequality constraints cᵢ(x) evaluated at the
///     current iterate.
/// @param y Equality constraint dual variables.
/// @param u Log-domain slack variables.
/// @param v Log-domain dual variables.
/// @param sqrt_μ Square root of the barrier parameter.
template <typename Scalar>
Scalar kkt_error(const Eigen::Vector<Scalar, Eigen::Dynamic>& g,
                 const Eigen::SparseMatrix<Scalar>& A_e,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& c_e,
                 const Eigen::SparseMatrix<Scalar>& A_i,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& c_i,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& y,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& u,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& v,
                 Scalar sqrt_μ) {
  // Compute the KKT error as the 1-norm of the KKT conditions.
  //
  //   ∇f − Aₑᵀy − Aᵢᵀz = 0
  //   cₑ = 0
  //   cᵢ − s = 0
  //
  // where
  //
  //   s = √(μ)e⁻ᵛ
  //   z = √(μ)eᵛ

  const Eigen::Vector<Scalar, Eigen::Dynamic> s =
      sqrt_μ * (-u).array().exp().matrix();
  const Eigen::Vector<Scalar, Eigen::Dynamic> z =
      sqrt_μ * v.array().exp().matrix();

  return (g - A_e.transpose() * y - A_i.transpose() * z).template lpNorm<1>() +
         c_e.template lpNorm<1>() + (c_i - s).template lpNorm<1>();
}

}  // namespace slp
