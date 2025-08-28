// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>

#include <Eigen/Core>
#include <Eigen/SparseCore>

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

/// Type of KKT error to compute.
enum class KKTErrorType {
  /// ∞-norm of scaled KKT condition errors.
  INF_NORM_SCALED,
  /// 1-norm of KKT condition errors.
  ONE_NORM
};

/// Returns the KKT error for Newton's method.
///
/// @tparam Scalar Scalar type.
/// @tparam T The type of KKT error to compute.
/// @param g Gradient of the cost function ∇f.
template <typename Scalar, KKTErrorType T>
Scalar kkt_error(const Eigen::Vector<Scalar, Eigen::Dynamic>& g) {
  // The KKT conditions from docs/algorithms.md:
  //
  //   ∇f = 0

  if constexpr (T == KKTErrorType::INF_NORM_SCALED) {
    return g.template lpNorm<Eigen::Infinity>();
  } else if constexpr (T == KKTErrorType::ONE_NORM) {
    return g.template lpNorm<1>();
  }
}

/// Returns the KKT error for Sequential Quadratic Programming.
///
/// @tparam Scalar Scalar type.
/// @tparam T The type of KKT error to compute.
/// @param g Gradient of the cost function ∇f.
/// @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
///     current iterate.
/// @param c_e The problem's equality constraints cₑ(x) evaluated at the current
///     iterate.
/// @param y Equality constraint dual variables.
template <typename Scalar, KKTErrorType T>
Scalar kkt_error(const Eigen::Vector<Scalar, Eigen::Dynamic>& g,
                 const Eigen::SparseMatrix<Scalar>& A_e,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& c_e,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& y) {
  // The KKT conditions from docs/algorithms.md:
  //
  //   ∇f − Aₑᵀy = 0
  //   cₑ = 0

  if constexpr (T == KKTErrorType::INF_NORM_SCALED) {
    // See equation (5) of [2].

    // s_d = max(sₘₐₓ, ‖y‖₁ / m) / sₘₐₓ
    constexpr Scalar s_max(100);
    Scalar s_d =
        std::max(s_max, y.template lpNorm<1>() / Scalar(y.rows())) / s_max;

    // ‖∇f − Aₑᵀy‖_∞ / s_d
    // ‖cₑ‖_∞
    return std::max(
        {(g - A_e.transpose() * y).template lpNorm<Eigen::Infinity>() / s_d,
         c_e.template lpNorm<Eigen::Infinity>()});
  } else if constexpr (T == KKTErrorType::ONE_NORM) {
    return (g - A_e.transpose() * y).template lpNorm<1>() +
           c_e.template lpNorm<1>();
  }
}

/// Returns the KKT error for the interior-point method.
///
/// @tparam Scalar Scalar type.
/// @tparam T The type of KKT error to compute.
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
template <typename Scalar, KKTErrorType T>
Scalar kkt_error(const Eigen::Vector<Scalar, Eigen::Dynamic>& g,
                 const Eigen::SparseMatrix<Scalar>& A_e,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& c_e,
                 const Eigen::SparseMatrix<Scalar>& A_i,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& c_i,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& y,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& u,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& v,
                 Scalar sqrt_μ) {
  // The KKT conditions from docs/algorithms.md:
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

  if constexpr (T == KKTErrorType::INF_NORM_SCALED) {
    // See equation (5) of [2].

    // s_d = max(sₘₐₓ, (‖y‖₁ + ‖z‖₁) / (m + n)) / sₘₐₓ
    constexpr Scalar s_max(100);
    Scalar s_d =
        std::max(s_max, (y.template lpNorm<1>() + z.template lpNorm<1>()) /
                            Scalar(y.rows() + z.rows())) /
        s_max;

    // ‖∇f − Aₑᵀy − Aᵢᵀz‖_∞ / s_d
    // ‖cₑ‖_∞
    // ‖cᵢ − s‖_∞
    return std::max({(g - A_e.transpose() * y - A_i.transpose() * z)
                             .template lpNorm<Eigen::Infinity>() /
                         s_d,
                     c_e.template lpNorm<Eigen::Infinity>(),
                     (c_i - s).template lpNorm<Eigen::Infinity>()});
  } else if constexpr (T == KKTErrorType::ONE_NORM) {
    return (g - A_e.transpose() * y - A_i.transpose() * z)
               .template lpNorm<1>() +
           c_e.template lpNorm<1>() + (c_i - s).template lpNorm<1>();
  }
}

}  // namespace slp
