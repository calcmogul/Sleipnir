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
/// If the manifold has a boundary, the boundary barrier term μ_b b(x) (see
/// Manifold::barrier()) is part of the problem. Its implicit dual variable for
/// an SPD block P is Z = μ_b P⁻¹, so the stationarity residual includes μ_b∇b,
/// and the complementarity residual PZ − μI = (μ_b − μ)I has ∞-norm
/// |μ_b − μ|.
///
/// @tparam Scalar Scalar type.
/// @param iterates The iterates.
/// @param function_values Cost function and constraint values.
/// @param first_derivatives Cost function gradient and constraint Jacobians.
/// @param g_b Boundary barrier function gradient ∇b, or empty if the manifold
///     has no boundary.
/// @param μ_b Boundary barrier parameter.
/// @param μ Barrier parameter.
/// @param tolerance Solver tolerance.
template <typename Scalar>
Scalar kkt_error_inf_norm(const IPMIterates<Scalar>& iterates,
                          const IPMFunctionValues<Scalar>& function_values,
                          const IPMFirstDerivatives<Scalar>& first_derivatives,
                          const Eigen::VectorX<Scalar>& g_b, Scalar μ_b,
                          Scalar μ, Scalar tolerance) {
  using std::abs;
  using std::pow;

  const auto& [x, s, y, z] = iterates;
  const auto& [f, c_e, c_i] = function_values;
  const auto& [g, A_e, A_i] = first_derivatives;

  // The KKT conditions from docs/algorithms.md:
  //
  //   ∇f + μ_b∇b − Aₑᵀy − Aᵢᵀz = 0
  //   Sz − μe = 0
  //   cₑ = 0
  //   cᵢ − s = 0
  //
  // and (μ_b − μ)I = 0 if the manifold has a boundary.
  const bool has_barrier = g_b.size() > 0;

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

  // ω = ‖|∇f| + μ_b|∇b| + |Aₑ|ᵀ|y| + |Aᵢ|ᵀ|z|‖_∞
  Eigen::VectorX<Scalar> terms =
      (g.cwiseAbs() + A_e.cwiseAbs().transpose() * y.cwiseAbs() +
       A_i.cwiseAbs().transpose() * z.cwiseAbs())
          .toDense();
  if (has_barrier) {
    terms += μ_b * g_b.cwiseAbs();
  }
  const Scalar ω = terms.template lpNorm<Eigen::Infinity>();

  Scalar s_d = std::max(Scalar(1), pow(ε, Scalar(0.625)) * ω / tolerance);

  // ∇f + μ_b∇b − Aₑᵀy − Aᵢᵀz
  Eigen::VectorX<Scalar> r_d =
      (g - A_e.transpose() * y - A_i.transpose() * z).toDense();
  if (has_barrier) {
    r_d += μ_b * g_b;
  }

  // s_c = max(sₘₐₓ, ‖z‖₁ / n) / sₘₐₓ
  //
  // See equation (5) of [2].
  constexpr Scalar s_max(100);
  Scalar s_c =
      std::max(s_max, z.template lpNorm<1>() / Scalar(z.rows())) / s_max;

  const auto S = s.asDiagonal();
  const Eigen::VectorX<Scalar> μe =
      Eigen::VectorX<Scalar>::Constant(s.rows(), μ);

  // ‖∇f + μ_b∇b − Aₑᵀy − Aᵢᵀz‖_∞ / s_d
  // ‖Sz − μe‖_∞ / s_c
  // |μ_b − μ|
  // ‖cₑ‖_∞
  // ‖cᵢ − s‖_∞
  return std::max({r_d.template lpNorm<Eigen::Infinity>() / s_d,
                   (S * z - μe).template lpNorm<Eigen::Infinity>() / s_c,
                   has_barrier ? abs(μ_b - μ) : Scalar(0),
                   c_e.template lpNorm<Eigen::Infinity>(),
                   (c_i - s).template lpNorm<Eigen::Infinity>()});
}

/// Returns the 1-norm of the KKT condition errors for the interior-point
/// method.
///
/// If the manifold has a boundary, the boundary barrier term μb(x) (see
/// Manifold::barrier()) is part of the problem. Its implicit dual variable for
/// an SPD block P is Z = μP⁻¹, so the stationarity residual includes μ∇b, and
/// the complementarity residual PZ − μI is zero.
///
/// @tparam Scalar Scalar type.
/// @param iterates The iterates.
/// @param function_values Cost function and constraint values.
/// @param first_derivatives Cost function gradient and constraint Jacobians.
/// @param g_b Boundary barrier function gradient ∇b, or empty if the manifold
///     has no boundary.
/// @param μ Barrier parameter.
template <typename Scalar>
Scalar kkt_error_one_norm(const IPMIterates<Scalar>& iterates,
                          const IPMFunctionValues<Scalar>& function_values,
                          const IPMFirstDerivatives<Scalar>& first_derivatives,
                          const Eigen::VectorX<Scalar>& g_b, Scalar μ) {
  const auto& [x, s, y, z] = iterates;
  const auto& [f, c_e, c_i] = function_values;
  const auto& [g, A_e, A_i] = first_derivatives;

  // The KKT conditions from docs/algorithms.md:
  //
  //   ∇f + μ∇b − Aₑᵀy − Aᵢᵀz = 0
  //   Sz − μe = 0
  //   cₑ = 0
  //   cᵢ − s = 0

  // ∇f + μ∇b − Aₑᵀy − Aᵢᵀz
  Eigen::VectorX<Scalar> r_d =
      (g - A_e.transpose() * y - A_i.transpose() * z).toDense();
  if (g_b.size() > 0) {
    r_d += μ * g_b;
  }

  const auto S = s.asDiagonal();
  const Eigen::VectorX<Scalar> μe =
      Eigen::VectorX<Scalar>::Constant(s.rows(), μ);

  return r_d.template lpNorm<1>() + (S * z - μe).template lpNorm<1>() +
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
/// @param g_b Boundary barrier function gradient ∇b, or empty if the manifold
///     has no boundary.
/// @param μ_b Scaled boundary barrier parameter.
/// @param μ Scaled barrier parameter.
/// @param tolerance Solver tolerance.
template <typename Scalar>
Scalar unscaled_kkt_error_inf_norm(
    const ProblemScaling<Scalar>& scaling, const IPMIterates<Scalar>& iterates,
    const IPMFunctionValues<Scalar>& function_values,
    const IPMFirstDerivatives<Scalar>& first_derivatives,
    const Eigen::VectorX<Scalar>& g_b, Scalar μ_b, Scalar μ, Scalar tolerance) {
  using DenseVector = Eigen::VectorX<Scalar>;

  if (scaling.is_identity()) {
    return kkt_error_inf_norm<Scalar>(
        iterates, function_values, first_derivatives, g_b, μ_b, μ, tolerance);
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
  const IPMIterates<Scalar> iterates_unscaled{
      iterates.x, inv_d_c_i.cwiseProduct(iterates.s),
      scaling.c_e.cwiseProduct(iterates.y) * inv_d_f,
      scaling.c_i.cwiseProduct(iterates.z) * inv_d_f};
  const Scalar μ_unscaled = inv_d_f * μ;

  // b(x) isn't scaled, so the unscaled barrier term is d_f⁻¹μ_b∇b
  const Scalar μ_b_unscaled = inv_d_f * μ_b;

  return kkt_error_inf_norm<Scalar>(iterates_unscaled, function_values_unscaled,
                                    first_derivatives_unscaled, g_b,
                                    μ_b_unscaled, μ_unscaled, tolerance);
}

}  // namespace slp
