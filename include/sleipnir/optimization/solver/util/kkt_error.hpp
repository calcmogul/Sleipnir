// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/util/problem_scaling.hpp"

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

/// Coefficient β₁ of the interior-point method's μβ₁e stationarity
/// perturbation.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
constexpr Scalar ipm_β_1(1e-4);

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
/// @tparam T Type of KKT error to compute.
/// @param g Cost function gradient ∇f.
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
/// @tparam T Type of KKT error to compute.
/// @param g Cost function gradient ∇f.
/// @param A_e Equality constraint Jacobian Aₑ(x).
/// @param c_e Equality constraints cₑ(x).
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
/// @tparam T Type of KKT error to compute.
/// @param g Cost function gradient ∇f.
/// @param A_i Inequality constraint Jacobian Aᵢ(x).
/// @param c_i Inequality constraints cᵢ(x).
/// @param v Log-domain variables.
/// @param sqrt_μ Square root of the barrier parameter for the iterate.
/// @param μ_target The barrier parameter against which to measure
///     complementarity and the μw and μβ₁e perturbations. Pass zero for the
///     error of the original problem and μ for the error of the barrier
///     subproblem.
template <typename Scalar, KKTErrorType T>
Scalar kkt_error(const Eigen::Vector<Scalar, Eigen::Dynamic>& g,
                 const Eigen::SparseMatrix<Scalar>& A_i,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& c_i,
                 const Eigen::Vector<Scalar, Eigen::Dynamic>& v, Scalar sqrt_μ,
                 Scalar μ_target) {
  // The KKT conditions from docs/algorithms.md:
  //
  //   ∇f − Aᵢᵀz − μβ₁e = 0
  //   Sz − μe = 0
  //   cᵢ − s + μw = 0
  //
  // where
  //
  //   s = √(μ)e⁻ᵛ
  //   z = √(μ)eᵛ
  //   w = e
  //
  // The log-domain parameterization satisfies Sz = μe exactly, so the
  // complementarity error relative to μ_target is |μ − μ_target|.

  using std::abs;

  const Eigen::Vector<Scalar, Eigen::Dynamic> s =
      sqrt_μ * (-v).array().exp().matrix();
  const Eigen::Vector<Scalar, Eigen::Dynamic> z =
      sqrt_μ * v.array().exp().matrix();

  // ∇f − Aᵢᵀz − μβ₁e
  const Eigen::Vector<Scalar, Eigen::Dynamic> r_d =
      ((g - A_i.transpose() * z).array() - μ_target * ipm_β_1<Scalar>).matrix();
  // cᵢ − s + μw
  const Eigen::Vector<Scalar, Eigen::Dynamic> r_p =
      ((c_i - s).array() + μ_target).matrix();

  if constexpr (T == KKTErrorType::INF_NORM_SCALED) {
    // See equation (5) of [2].

    // s_d = max(sₘₐₓ, ‖z‖₁ / n) / sₘₐₓ
    constexpr Scalar s_max(100);
    Scalar s_d =
        std::max(s_max, z.template lpNorm<1>() / Scalar(z.rows())) / s_max;

    // s_c = max(sₘₐₓ, ‖z‖₁ / n) / sₘₐₓ
    Scalar s_c =
        std::max(s_max, z.template lpNorm<1>() / Scalar(z.rows())) / s_max;

    // ‖∇f − Aᵢᵀz − μβ₁e‖_∞ / s_d
    // ‖Sz − μe‖_∞ / s_c
    // ‖cᵢ − s + μw‖_∞
    return std::max(
        {r_d.template lpNorm<Eigen::Infinity>() / s_d,
         z.rows() > 0 ? abs(sqrt_μ * sqrt_μ - μ_target) / s_c : Scalar(0),
         r_p.template lpNorm<Eigen::Infinity>()});
  } else if constexpr (T == KKTErrorType::ONE_NORM) {
    return r_d.template lpNorm<1>() +
           Scalar(z.rows()) * abs(sqrt_μ * sqrt_μ - μ_target) +
           r_p.template lpNorm<1>();
  }
}

/// Returns the unscaled KKT error for Newton's method.
///
/// @tparam Scalar Scalar type.
/// @tparam T Type of KKT error to compute.
/// @param scaling Problem scaling.
/// @param g Scaled cost function gradient d_f·∇f.
template <typename Scalar, KKTErrorType T>
Scalar unscaled_kkt_error(const ProblemScaling<Scalar>& scaling,
                          const Eigen::Vector<Scalar, Eigen::Dynamic>& g) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  if (scaling.is_identity()) {
    return kkt_error<Scalar, T>(g);
  }

  const DenseVector g_unscaled = (Scalar(1) / scaling.f) * g;

  return kkt_error<Scalar, T>(g_unscaled);
}

/// Returns the unscaled KKT error for Sequential Quadratic Programming.
///
/// @tparam Scalar Scalar type.
/// @tparam T Type of KKT error to compute.
/// @param scaling Problem scaling.
/// @param g Scaled cost function gradient d_f·∇f.
/// @param A_e Scaled equality constraint Jacobian D_cₑ·Aₑ(x).
/// @param c_e Scaled equality constraints D_cₑ·cₑ(x).
/// @param y Scaled equality constraint dual variables.
template <typename Scalar, KKTErrorType T>
Scalar unscaled_kkt_error(const ProblemScaling<Scalar>& scaling,
                          const Eigen::Vector<Scalar, Eigen::Dynamic>& g,
                          const Eigen::SparseMatrix<Scalar>& A_e,
                          const Eigen::Vector<Scalar, Eigen::Dynamic>& c_e,
                          const Eigen::Vector<Scalar, Eigen::Dynamic>& y) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  if (scaling.is_identity()) {
    return kkt_error<Scalar, T>(g, A_e, c_e, y);
  }

  const Scalar inv_d_f = Scalar(1) / scaling.f;
  const DenseVector inv_d_c_e = scaling.c_e.cwiseInverse();

  const DenseVector g_unscaled = inv_d_f * g;
  const SparseMatrix A_e_unscaled = inv_d_c_e.asDiagonal() * A_e;
  const DenseVector c_e_unscaled = inv_d_c_e.cwiseProduct(c_e);
  const DenseVector y_unscaled = scaling.c_e.cwiseProduct(y) * inv_d_f;

  return kkt_error<Scalar, T>(g_unscaled, A_e_unscaled, c_e_unscaled,
                              y_unscaled);
}

/// Returns the unscaled KKT error for the interior-point method.
///
/// @tparam Scalar Scalar type.
/// @tparam T Type of KKT error to compute.
/// @param scaling Problem scaling.
/// @param g Scaled cost function gradient d_f·∇f.
/// @param A_i Scaled inequality constraint Jacobian D_cᵢ·Aᵢ(x).
/// @param c_i Scaled inequality constraints D_cᵢ·cᵢ(x).
/// @param v Scaled log-domain variables.
/// @param sqrt_μ Square root of the scaled barrier parameter.
/// @param μ_target The scaled barrier parameter against which to measure
///     complementarity.
template <typename Scalar, KKTErrorType T>
Scalar unscaled_kkt_error(const ProblemScaling<Scalar>& scaling,
                          const Eigen::Vector<Scalar, Eigen::Dynamic>& g,
                          const Eigen::SparseMatrix<Scalar>& A_i,
                          const Eigen::Vector<Scalar, Eigen::Dynamic>& c_i,
                          const Eigen::Vector<Scalar, Eigen::Dynamic>& v,
                          Scalar sqrt_μ, Scalar μ_target) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  using std::log;
  using std::sqrt;

  if (scaling.is_identity()) {
    return kkt_error<Scalar, T>(g, A_i, c_i, v, sqrt_μ, μ_target);
  }

  const Scalar inv_d_f = Scalar(1) / scaling.f;
  const DenseVector inv_d_c_i = scaling.c_i.cwiseInverse();

  const DenseVector g_unscaled = inv_d_f * g;
  const SparseMatrix A_i_unscaled = inv_d_c_i.asDiagonal() * A_i;
  const DenseVector c_i_unscaled = inv_d_c_i.cwiseProduct(c_i);

  // The unscaled slacks and duals are
  //
  //   sᵤ = D_cᵢ⁻¹s
  //   zᵤ = D_cᵢz/d_f
  //
  // so sᵤzᵤ = μ/d_f. Matching sᵤ = √(μᵤ)exp(−vᵤ) and zᵤ = √(μᵤ)exp(vᵤ) gives
  //
  //   √(μᵤ) = √(μ/d_f)
  //   vᵤ = v + ln(D_cᵢ) − ½ln(d_f)
  const DenseVector v_unscaled =
      (v.array() + scaling.c_i.array().log() - Scalar(0.5) * log(scaling.f))
          .matrix();
  const Scalar sqrt_μ_unscaled = sqrt_μ * sqrt(inv_d_f);

  const Scalar μ_target_unscaled = μ_target * inv_d_f;

  return kkt_error<Scalar, T>(g_unscaled, A_i_unscaled, c_i_unscaled,
                              v_unscaled, sqrt_μ_unscaled, μ_target_unscaled);
}

}  // namespace slp
