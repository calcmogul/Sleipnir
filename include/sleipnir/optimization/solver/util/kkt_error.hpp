// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/util/first_derivatives.hpp"
#include "sleipnir/optimization/solver/util/function_values.hpp"
#include "sleipnir/optimization/solver/util/iterates.hpp"
#include "sleipnir/optimization/solver/util/problem_scaling.hpp"

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

/// Returns the ∞-norm of the KKT condition errors for Newton's method.
///
/// @tparam Scalar Scalar type.
/// @param g Cost function gradient ∇f.
template <typename Scalar>
Scalar kkt_error_inf_norm(const Eigen::SparseVector<Scalar>& g) {
  // The KKT conditions from docs/algorithms.md:
  //
  //   ∇f = 0
  return g.coeffs().template lpNorm<Eigen::Infinity>();
}

/// Returns the 1-norm of the KKT condition errors for Newton's method.
///
/// @tparam Scalar Scalar type.
/// @param g Cost function gradient ∇f.
template <typename Scalar>
Scalar kkt_error_one_norm(const Eigen::SparseVector<Scalar>& g) {
  // The KKT conditions from docs/algorithms.md:
  //
  //   ∇f = 0
  return g.coeffs().template lpNorm<1>();
}

/// Returns the ∞-norm of the scaled KKT condition errors for Sequential
/// Quadratic Programming.
///
/// @tparam Scalar Scalar type.
/// @param iterates The iterates.
/// @param function_values Cost function and equality constraint values.
/// @param first_derivatives Cost function gradient and equality constraint
///     Jacobian.
/// @param tolerance Solver tolerance.
template <typename Scalar>
Scalar kkt_error_inf_norm(const SQPIterates<Scalar>& iterates,
                          const SQPFunctionValues<Scalar>& function_values,
                          const SQPFirstDerivatives<Scalar>& first_derivatives,
                          Scalar tolerance) {
  using std::pow;

  const auto& y = iterates.y;
  const auto& c_e = function_values.c_e;
  const auto& [g, A_e] = first_derivatives;

  // The KKT conditions from docs/algorithms.md:
  //
  //   ∇f − Aₑᵀy = 0
  //   cₑ = 0

  constexpr Scalar ε = std::numeric_limits<Scalar>::epsilon();

  // The stationarity residual must not go below its rounding error floor κεω,
  // where κ is how much the problem amplifies rounding error and ω is the size
  // of the largest stationarity terms. If κεω > tol, the solver stalls instead
  // of terminating.
  //
  // κ is unknown, so assume κ ≤ ε^(-3/8) (≈ 7.4e5 for double, 395 for float).
  // Then the floor is at most
  //
  //   κεω ≤ ε^(-3/8) εω = ε^(5/8) ω
  //
  // and dividing the residual by ε^(5/8) ω/tol makes it meet tol at the floor.
  // Dividing by less than 1 would tighten tol instead, so
  //
  //   s_d = max(1, ε^(5/8) ω/tol)

  // ω = ‖|∇f| + |Aₑ|ᵀ|y|‖_∞
  const Scalar ω = (g.cwiseAbs() + A_e.cwiseAbs().transpose() * y.cwiseAbs())
                       .toDense()
                       .template lpNorm<Eigen::Infinity>();

  Scalar s_d = std::max(Scalar(1), pow(ε, Scalar(0.625)) * ω / tolerance);

  // ‖∇f − Aₑᵀy‖_∞ / s_d
  // ‖cₑ‖_∞
  return std::max(
      {(g - A_e.transpose() * y).toDense().template lpNorm<Eigen::Infinity>() /
           s_d,
       c_e.template lpNorm<Eigen::Infinity>()});
}

/// Returns the 1-norm of the KKT condition errors for Sequential Quadratic
/// Programming.
///
/// @tparam Scalar Scalar type.
/// @param iterates The iterates.
/// @param function_values Cost function and equality constraint values.
/// @param first_derivatives Cost function gradient and equality constraint
///     Jacobian.
template <typename Scalar>
Scalar kkt_error_one_norm(
    const SQPIterates<Scalar>& iterates,
    const SQPFunctionValues<Scalar>& function_values,
    const SQPFirstDerivatives<Scalar>& first_derivatives) {
  const auto& y = iterates.y;
  const auto& c_e = function_values.c_e;
  const auto& [g, A_e] = first_derivatives;

  // The KKT conditions from docs/algorithms.md:
  //
  //   ∇f − Aₑᵀy = 0
  //   cₑ = 0
  return (g - A_e.transpose() * y).toDense().template lpNorm<1>() +
         c_e.template lpNorm<1>();
}

/// Returns the ∞-norm of the scaled KKT condition errors for the interior-point
/// method.
///
/// @tparam Scalar Scalar type.
/// @param iterates The iterates.
/// @param function_values Cost function and constraint values.
/// @param first_derivatives Cost function gradient and constraint Jacobians.
/// @param sqrt_μ Square root of the barrier parameter for the iterate.
/// @param μ_target The barrier parameter against which to measure
///     complementarity. Pass zero for the error of the original problem and μ
///     for the error of the barrier subproblem.
/// @param tolerance Solver tolerance.
template <typename Scalar>
Scalar kkt_error_inf_norm(const IPMIterates<Scalar>& iterates,
                          const IPMFunctionValues<Scalar>& function_values,
                          const IPMFirstDerivatives<Scalar>& first_derivatives,
                          Scalar sqrt_μ, Scalar μ_target, Scalar tolerance) {
  using std::pow;

  const auto& [x, y, u, v] = iterates;
  const auto& [f, c_e, c_i] = function_values;
  const auto& [g, A_e, A_i] = first_derivatives;

  // The KKT conditions from docs/algorithms.md:
  //
  //   ∇f − Aₑᵀy − Aᵢᵀz = 0
  //   Sz − μe = 0
  //   cₑ = 0
  //   cᵢ − s = 0
  //
  // where
  //
  //   s = √(μ)e⁻ᵘ
  //   z = √(μ)eᵛ

  const Eigen::VectorX<Scalar> s = sqrt_μ * (-u).array().exp().matrix();
  const Eigen::VectorX<Scalar> z = sqrt_μ * v.array().exp().matrix();

  constexpr Scalar ε = std::numeric_limits<Scalar>::epsilon();

  // The stationarity residual must not go below its rounding error floor κεω,
  // where κ is how much the problem amplifies rounding error and ω is the size
  // of the largest stationarity terms. If κεω > tol, the solver stalls instead
  // of terminating.
  //
  // κ is unknown, so assume κ ≤ ε^(-3/8) (≈ 7.4e5 for double, 395 for float).
  // Then the floor is at most
  //
  //   κεω ≤ ε^(-3/8) εω = ε^(5/8) ω
  //
  // and dividing the residual by ε^(5/8) ω/tol makes it meet tol at the floor.
  // Dividing by less than 1 would tighten tol instead, so
  //
  //   s_d = max(1, ε^(5/8) ω/tol)

  // ω = ‖|∇f| + |Aₑ|ᵀ|y| + |Aᵢ|ᵀ|z|‖_∞
  const Scalar ω = (g.cwiseAbs() + A_e.cwiseAbs().transpose() * y.cwiseAbs() +
                    A_i.cwiseAbs().transpose() * z.cwiseAbs())
                       .toDense()
                       .template lpNorm<Eigen::Infinity>();

  Scalar s_d = std::max(Scalar(1), pow(ε, Scalar(0.625)) * ω / tolerance);

  // s_c = max(sₘₐₓ, ‖z‖₁ / n) / sₘₐₓ
  //
  // See equation (5) of [2].
  constexpr Scalar s_max(100);
  Scalar s_c =
      std::max(s_max, z.template lpNorm<1>() / Scalar(z.rows())) / s_max;

  // ‖∇f − Aₑᵀy − Aᵢᵀz‖_∞ / s_d
  // ‖Sz − μe‖_∞ / s_c
  // ‖cₑ‖_∞
  // ‖cᵢ − s‖_∞
  return std::max({(g - A_e.transpose() * y - A_i.transpose() * z)
                           .toDense()
                           .template lpNorm<Eigen::Infinity>() /
                       s_d,
                   (s.cwiseProduct(z).array() - μ_target)
                           .matrix()
                           .template lpNorm<Eigen::Infinity>() /
                       s_c,
                   c_e.template lpNorm<Eigen::Infinity>(),
                   (c_i - s).template lpNorm<Eigen::Infinity>()});
}

/// Returns the 1-norm of the KKT condition errors for the interior-point
/// method.
///
/// @tparam Scalar Scalar type.
/// @param iterates The iterates.
/// @param function_values Cost function and constraint values.
/// @param first_derivatives Cost function gradient and constraint Jacobians.
/// @param sqrt_μ Square root of the barrier parameter for the iterate.
/// @param μ_target The barrier parameter against which to measure
///     complementarity. Pass zero for the error of the original problem and μ
///     for the error of the barrier subproblem.
template <typename Scalar>
Scalar kkt_error_one_norm(const IPMIterates<Scalar>& iterates,
                          const IPMFunctionValues<Scalar>& function_values,
                          const IPMFirstDerivatives<Scalar>& first_derivatives,
                          Scalar sqrt_μ, Scalar μ_target) {
  const auto& [x, y, u, v] = iterates;
  const auto& [f, c_e, c_i] = function_values;
  const auto& [g, A_e, A_i] = first_derivatives;

  // The KKT conditions from docs/algorithms.md:
  //
  //   ∇f − Aₑᵀy − Aᵢᵀz = 0
  //   Sz − μe = 0
  //   cₑ = 0
  //   cᵢ − s = 0
  //
  // where
  //
  //   s = √(μ)e⁻ᵘ
  //   z = √(μ)eᵛ

  const Eigen::VectorX<Scalar> s = sqrt_μ * (-u).array().exp().matrix();
  const Eigen::VectorX<Scalar> z = sqrt_μ * v.array().exp().matrix();

  return (g - A_e.transpose() * y - A_i.transpose() * z)
             .toDense()
             .template lpNorm<1>() +
         (s.cwiseProduct(z).array() - μ_target).matrix().template lpNorm<1>() +
         c_e.template lpNorm<1>() + (c_i - s).template lpNorm<1>();
}

/// Returns the ∞-norm of the unscaled KKT condition errors for Newton's method.
///
/// @tparam Scalar Scalar type.
/// @param scaling Problem scaling.
/// @param g Scaled cost function gradient d_f·∇f.
template <typename Scalar>
Scalar unscaled_kkt_error_inf_norm(const ProblemScaling<Scalar>& scaling,
                                   const Eigen::SparseVector<Scalar>& g) {
  using SparseVector = Eigen::SparseVector<Scalar>;

  if (scaling.is_identity()) {
    return kkt_error_inf_norm<Scalar>(g);
  }

  const SparseVector g_unscaled = (Scalar(1) / scaling.f) * g;

  return kkt_error_inf_norm<Scalar>(g_unscaled);
}

/// Returns the ∞-norm of the unscaled KKT condition errors for Sequential
/// Quadratic Programming.
///
/// @tparam Scalar Scalar type.
/// @param scaling Problem scaling.
/// @param iterates Scaled iterates.
/// @param function_values Scaled cost function and equality constraint values.
/// @param first_derivatives Scaled cost function gradient and equality
///     constraint Jacobian.
/// @param tolerance Solver tolerance.
template <typename Scalar>
Scalar unscaled_kkt_error_inf_norm(
    const ProblemScaling<Scalar>& scaling, const SQPIterates<Scalar>& iterates,
    const SQPFunctionValues<Scalar>& function_values,
    const SQPFirstDerivatives<Scalar>& first_derivatives, Scalar tolerance) {
  using DenseVector = Eigen::VectorX<Scalar>;

  if (scaling.is_identity()) {
    return kkt_error_inf_norm<Scalar>(iterates, function_values,
                                      first_derivatives, tolerance);
  }

  const Scalar inv_d_f = Scalar(1) / scaling.f;
  const DenseVector inv_d_c_e = scaling.c_e.cwiseInverse();

  const SQPFunctionValues<Scalar> function_values_unscaled{
      inv_d_f * function_values.f, inv_d_c_e.cwiseProduct(function_values.c_e)};
  const SQPFirstDerivatives<Scalar> first_derivatives_unscaled{
      inv_d_f * first_derivatives.g,
      inv_d_c_e.asDiagonal() * first_derivatives.A_e};
  const SQPIterates<Scalar> iterates_unscaled{
      iterates.x, scaling.c_e.cwiseProduct(iterates.y) * inv_d_f};

  return kkt_error_inf_norm<Scalar>(iterates_unscaled, function_values_unscaled,
                                    first_derivatives_unscaled, tolerance);
}

/// Returns the ∞-norm of the unscaled KKT condition errors for the
/// interior-point method.
///
/// @tparam Scalar Scalar type.
/// @param scaling Problem scaling.
/// @param iterates Scaled iterates.
/// @param function_values Scaled cost function and constraint values.
/// @param first_derivatives Scaled cost function gradient and constraint
///     Jacobians.
/// @param sqrt_μ Square root of the scaled barrier parameter.
/// @param μ_target The scaled barrier parameter against which to measure
///     complementarity.
/// @param tolerance Solver tolerance.
template <typename Scalar>
Scalar unscaled_kkt_error_inf_norm(
    const ProblemScaling<Scalar>& scaling, const IPMIterates<Scalar>& iterates,
    const IPMFunctionValues<Scalar>& function_values,
    const IPMFirstDerivatives<Scalar>& first_derivatives, Scalar sqrt_μ,
    Scalar μ_target, Scalar tolerance) {
  using DenseVector = Eigen::VectorX<Scalar>;

  using std::log;
  using std::sqrt;

  if (scaling.is_identity()) {
    return kkt_error_inf_norm<Scalar>(iterates, function_values,
                                      first_derivatives, sqrt_μ, μ_target,
                                      tolerance);
  }

  const Scalar inv_d_f = Scalar(1) / scaling.f;
  const DenseVector inv_d_c_e = scaling.c_e.cwiseInverse();
  const DenseVector inv_d_c_i = scaling.c_i.cwiseInverse();

  const IPMFunctionValues<Scalar> function_values_unscaled{
      inv_d_f * function_values.f, inv_d_c_e.cwiseProduct(function_values.c_e),
      inv_d_c_i.cwiseProduct(function_values.c_i)};
  const IPMFirstDerivatives<Scalar> first_derivatives_unscaled{
      inv_d_f * first_derivatives.g,
      inv_d_c_e.asDiagonal() * first_derivatives.A_e,
      inv_d_c_i.asDiagonal() * first_derivatives.A_i};

  // The unscaled slacks and duals are
  //
  //   sᵤ = D_cᵢ⁻¹s
  //   zᵤ = D_cᵢz/d_f
  //
  // Matching sᵤ = √(μᵤ)exp(−uᵤ) and zᵤ = √(μᵤ)exp(vᵤ) gives
  //
  //   √(μᵤ) = √(μ/d_f)
  //   uᵤ = u + ln(D_cᵢ) − ½ln(d_f)
  //   vᵤ = v + ln(D_cᵢ) − ½ln(d_f)
  const DenseVector log_shift =
      (scaling.c_i.array().log() - Scalar(0.5) * log(scaling.f)).matrix();
  const IPMIterates<Scalar> iterates_unscaled{
      iterates.x, scaling.c_e.cwiseProduct(iterates.y) * inv_d_f,
      iterates.u + log_shift, iterates.v + log_shift};
  const Scalar sqrt_μ_unscaled = sqrt_μ * sqrt(inv_d_f);

  const Scalar μ_target_unscaled = μ_target * inv_d_f;

  return kkt_error_inf_norm<Scalar>(iterates_unscaled, function_values_unscaled,
                                    first_derivatives_unscaled, sqrt_μ_unscaled,
                                    μ_target_unscaled, tolerance);
}

}  // namespace slp
