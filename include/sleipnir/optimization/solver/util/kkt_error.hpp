// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>

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
/// @param g Cost function gradient ∇f.
/// @param A_e Equality constraint Jacobian Aₑ(x).
/// @param c_e Equality constraints cₑ(x).
/// @param y Equality constraint dual variables.
/// @param tolerance Solver tolerance.
template <typename Scalar>
Scalar kkt_error_inf_norm(const Eigen::SparseVector<Scalar>& g,
                          const Eigen::SparseMatrix<Scalar>& A_e,
                          const Eigen::VectorX<Scalar>& c_e,
                          const Eigen::VectorX<Scalar>& y, Scalar tolerance) {
  using std::pow;

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
/// @param g Cost function gradient ∇f.
/// @param A_e Equality constraint Jacobian Aₑ(x).
/// @param c_e Equality constraints cₑ(x).
/// @param y Equality constraint dual variables.
template <typename Scalar>
Scalar kkt_error_one_norm(const Eigen::SparseVector<Scalar>& g,
                          const Eigen::SparseMatrix<Scalar>& A_e,
                          const Eigen::VectorX<Scalar>& c_e,
                          const Eigen::VectorX<Scalar>& y) {
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
/// @param g Cost function gradient ∇f.
/// @param A_e Equality constraint Jacobian Aₑ(x).
/// @param c_e Equality constraints cₑ(x).
/// @param A_i Inequality constraint Jacobian Aᵢ(x).
/// @param c_i Inequality constraints cᵢ(x).
/// @param s Inequality constraint slack variables.
/// @param y Equality constraint dual variables.
/// @param z Inequality constraint dual variables.
/// @param g_b Boundary barrier function gradient ∇b, or empty if the manifold
///     has no boundary.
/// @param μ_b Boundary barrier parameter.
/// @param μ Barrier parameter.
/// @param tolerance Solver tolerance.
template <typename Scalar>
Scalar kkt_error_inf_norm(
    const Eigen::SparseVector<Scalar>& g,
    const Eigen::SparseMatrix<Scalar>& A_e, const Eigen::VectorX<Scalar>& c_e,
    const Eigen::SparseMatrix<Scalar>& A_i, const Eigen::VectorX<Scalar>& c_i,
    const Eigen::VectorX<Scalar>& s, const Eigen::VectorX<Scalar>& y,
    const Eigen::VectorX<Scalar>& z, const Eigen::VectorX<Scalar>& g_b,
    Scalar μ_b, Scalar μ, Scalar tolerance) {
  using std::abs;
  using std::pow;

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
  Eigen::VectorX<Scalar> terms = (g.cwiseAbs() +
                                  A_e.cwiseAbs().transpose() * y.cwiseAbs() +
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
/// @param g Cost function gradient ∇f.
/// @param A_e Equality constraint Jacobian Aₑ(x).
/// @param c_e Equality constraints cₑ(x).
/// @param A_i Inequality constraint Jacobian Aᵢ(x).
/// @param c_i Inequality constraints cᵢ(x).
/// @param s Inequality constraint slack variables.
/// @param y Equality constraint dual variables.
/// @param z Inequality constraint dual variables.
/// @param g_b Boundary barrier function gradient ∇b, or empty if the manifold
///     has no boundary.
/// @param μ Barrier parameter.
template <typename Scalar>
Scalar kkt_error_one_norm(
    const Eigen::SparseVector<Scalar>& g,
    const Eigen::SparseMatrix<Scalar>& A_e, const Eigen::VectorX<Scalar>& c_e,
    const Eigen::SparseMatrix<Scalar>& A_i, const Eigen::VectorX<Scalar>& c_i,
    const Eigen::VectorX<Scalar>& s, const Eigen::VectorX<Scalar>& y,
    const Eigen::VectorX<Scalar>& z, const Eigen::VectorX<Scalar>& g_b,
    Scalar μ) {
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
/// @param g Scaled cost function gradient d_f·∇f.
/// @param A_e Scaled equality constraint Jacobian D_cₑ·Aₑ(x).
/// @param c_e Scaled equality constraints D_cₑ·cₑ(x).
/// @param y Scaled equality constraint dual variables.
/// @param tolerance Solver tolerance.
template <typename Scalar>
Scalar unscaled_kkt_error_inf_norm(const ProblemScaling<Scalar>& scaling,
                                   const Eigen::SparseVector<Scalar>& g,
                                   const Eigen::SparseMatrix<Scalar>& A_e,
                                   const Eigen::VectorX<Scalar>& c_e,
                                   const Eigen::VectorX<Scalar>& y,
                                   Scalar tolerance) {
  using DenseVector = Eigen::VectorX<Scalar>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using SparseVector = Eigen::SparseVector<Scalar>;

  if (scaling.is_identity()) {
    return kkt_error_inf_norm<Scalar>(g, A_e, c_e, y, tolerance);
  }

  const Scalar inv_d_f = Scalar(1) / scaling.f;
  const DenseVector inv_d_c_e = scaling.c_e.cwiseInverse();

  const SparseVector g_unscaled = inv_d_f * g;
  const SparseMatrix A_e_unscaled = inv_d_c_e.asDiagonal() * A_e;
  const DenseVector c_e_unscaled = inv_d_c_e.cwiseProduct(c_e);
  const DenseVector y_unscaled = scaling.c_e.cwiseProduct(y) * inv_d_f;

  return kkt_error_inf_norm<Scalar>(g_unscaled, A_e_unscaled, c_e_unscaled,
                                    y_unscaled, tolerance);
}

/// Returns the ∞-norm of the unscaled KKT condition errors for the
/// interior-point method.
///
/// @tparam Scalar Scalar type.
/// @param scaling Problem scaling.
/// @param g Scaled cost function gradient d_f·∇f.
/// @param A_e Scaled equality constraint Jacobian D_cₑ·Aₑ(x).
/// @param c_e Scaled equality constraints D_cₑ·cₑ(x).
/// @param A_i Scaled inequality constraint Jacobian D_cᵢ·Aᵢ(x).
/// @param c_i Scaled inequality constraints D_cᵢ·cᵢ(x).
/// @param s Scaled inequality constraint slack variables.
/// @param y Scaled equality constraint dual variables.
/// @param z Scaled inequality constraint dual variables.
/// @param g_b Boundary barrier function gradient ∇b, or empty if the manifold
///     has no boundary.
/// @param μ_b Scaled boundary barrier parameter.
/// @param μ Scaled barrier parameter.
/// @param tolerance Solver tolerance.
template <typename Scalar>
Scalar unscaled_kkt_error_inf_norm(
    const ProblemScaling<Scalar>& scaling, const Eigen::SparseVector<Scalar>& g,
    const Eigen::SparseMatrix<Scalar>& A_e, const Eigen::VectorX<Scalar>& c_e,
    const Eigen::SparseMatrix<Scalar>& A_i, const Eigen::VectorX<Scalar>& c_i,
    const Eigen::VectorX<Scalar>& s, const Eigen::VectorX<Scalar>& y,
    const Eigen::VectorX<Scalar>& z, const Eigen::VectorX<Scalar>& g_b,
    Scalar μ_b, Scalar μ, Scalar tolerance) {
  using DenseVector = Eigen::VectorX<Scalar>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using SparseVector = Eigen::SparseVector<Scalar>;

  if (scaling.is_identity()) {
    return kkt_error_inf_norm<Scalar>(g, A_e, c_e, A_i, c_i, s, y, z, g_b, μ_b,
                                      μ, tolerance);
  }

  const Scalar inv_d_f = Scalar(1) / scaling.f;
  const DenseVector inv_d_c_e = scaling.c_e.cwiseInverse();
  const DenseVector inv_d_c_i = scaling.c_i.cwiseInverse();

  const SparseVector g_unscaled = inv_d_f * g;
  const SparseMatrix A_e_unscaled = inv_d_c_e.asDiagonal() * A_e;
  const DenseVector c_e_unscaled = inv_d_c_e.cwiseProduct(c_e);
  const SparseMatrix A_i_unscaled = inv_d_c_i.asDiagonal() * A_i;
  const DenseVector c_i_unscaled = inv_d_c_i.cwiseProduct(c_i);
  const DenseVector s_unscaled = inv_d_c_i.cwiseProduct(s);
  const DenseVector y_unscaled = scaling.c_e.cwiseProduct(y) * inv_d_f;
  const DenseVector z_unscaled = scaling.c_i.cwiseProduct(z) * inv_d_f;
  const Scalar μ_unscaled = inv_d_f * μ;

  // b(x) isn't scaled, so the unscaled barrier term is d_f⁻¹μ_b∇b
  const Scalar μ_b_unscaled = inv_d_f * μ_b;

  return kkt_error_inf_norm<Scalar>(g_unscaled, A_e_unscaled, c_e_unscaled,
                                    A_i_unscaled, c_i_unscaled, s_unscaled,
                                    y_unscaled, z_unscaled, g_b, μ_b_unscaled,
                                    μ_unscaled, tolerance);
}

}  // namespace slp
