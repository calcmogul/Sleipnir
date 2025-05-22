// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/interior_point_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/sqp_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/util/append_as_triplets.hpp"
#include "sleipnir/optimization/solver/util/lagrange_multiplier_estimate.hpp"

namespace slp {

template <typename Scalar>
ExitStatus interior_point(
    const InteriorPointMatrixCallbacks<Scalar>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options& options, bool in_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::Vector<Scalar, Eigen::Dynamic>& x,
    Eigen::Vector<Scalar, Eigen::Dynamic>& s,
    Eigen::Vector<Scalar, Eigen::Dynamic>& y,
    Eigen::Vector<Scalar, Eigen::Dynamic>& z, Scalar& μ);

/// Computes initial values for p and n in feasibility restoration.
///
/// @tparam Scalar Scalar type.
/// @param[in] c The constraint violation.
/// @param[in] ρ Scaling parameter.
/// @param[in] μ Barrier parameter.
/// @return Tuple of positive and negative constraint relaxation slack variables
///     respectively.
template <typename Scalar>
std::tuple<Eigen::Vector<Scalar, Eigen::Dynamic>,
           Eigen::Vector<Scalar, Eigen::Dynamic>>
compute_p_n(const Eigen::Vector<Scalar, Eigen::Dynamic>& c, Scalar ρ,
            Scalar μ) {
  // From equation (33) of [2]:
  //                       ______________________
  //       μ − ρ c(x)     /(μ − ρ c(x))²   μ c(x)
  //   n = −−−−−−−−−− +  / (−−−−−−−−−−)  + −−−−−−     (1)
  //           2ρ       √  (    2ρ    )      2ρ
  //
  // The quadratic formula:
  //             ________
  //       -b + √b² - 4ac
  //   x = −−−−−−−−−−−−−−                             (2)
  //             2a
  //
  // Rearrange (1) to fit the quadratic formula better:
  //                     _________________________
  //       μ - ρ c(x) + √(μ - ρ c(x))² + 2ρ μ c(x)
  //   n = −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
  //                         2ρ
  //
  // Solve for coefficients:
  //
  //   a = ρ                                          (3)
  //   b = ρ c(x) - μ                                 (4)
  //
  //   -4ac = 2ρ μ c(x)
  //   -4(ρ)c = 2ρ μ c(x)
  //   -4c = 2μ c(x)
  //   c = -μ c(x)/2                                  (5)
  //
  //   p = c(x) + n                                   (6)

  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  using std::sqrt;

  DenseVector p{c.rows()};
  DenseVector n{c.rows()};
  for (int row = 0; row < p.rows(); ++row) {
    Scalar _a = ρ;
    Scalar _b = ρ * c[row] - μ;
    Scalar _c = -μ * c[row] / Scalar(2);

    n[row] = (-_b + sqrt(_b * _b - Scalar(4) * _a * _c)) / (Scalar(2) * _a);
    p[row] = c[row] + n[row];
  }

  return {std::move(p), std::move(n)};
}

/// Finds the iterate that minimizes the constraint violation while not
/// deviating too far from the starting point. This is a fallback procedure when
/// the normal Sequential Quadratic Programming method fails to converge to a
/// feasible point.
///
/// @tparam Scalar Scalar type.
/// @param[in] matrix_callbacks Matrix callbacks.
/// @param[in] is_nlp If true, the solver uses a more conservative barrier
///     parameter reduction strategy that's more reliable on NLPs. Pass false
///     for problems with quadratic or lower-order cost and linear or
///     lower-order constraints.
/// @param[in] iteration_callbacks The list of callbacks to call at the
///     beginning of each iteration.
/// @param[in] options Solver options.
/// @param[in,out] x The decision variables from the normal solve.
/// @param[in,out] y The equality constraint dual variables from the normal
///     solve.
/// @return The exit status.
template <typename Scalar>
ExitStatus feasibility_restoration(
    [[maybe_unused]] const SQPMatrixCallbacks<Scalar>& matrix_callbacks,
    [[maybe_unused]] bool is_nlp,
    [[maybe_unused]] std::span<
        std::function<bool(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    [[maybe_unused]] const Options& options,
    [[maybe_unused]] Eigen::Vector<Scalar, Eigen::Dynamic>& x,
    [[maybe_unused]] Eigen::Vector<Scalar, Eigen::Dynamic>& y) {
  return ExitStatus::FEASIBILITY_RESTORATION_FAILED;
}

/// Finds the iterate that minimizes the constraint violation while not
/// deviating too far from the starting point. This is a fallback procedure when
/// the normal interior-point method fails to converge to a feasible point.
///
/// @tparam Scalar Scalar type.
/// @param[in] matrix_callbacks Matrix callbacks.
/// @param[in] is_nlp If true, the solver uses a more conservative barrier
///     parameter reduction strategy that's more reliable on NLPs. Pass false
///     for problems with quadratic or lower-order cost and linear or
///     lower-order constraints.
/// @param[in] iteration_callbacks The list of callbacks to call at the
///     beginning of each iteration.
/// @param[in] options Solver options.
/// @param[in,out] x The current decision variables from the normal solve.
/// @param[in,out] v The current log-domain variables from the normal solve.
/// @param[in,out] sqrt_μ Barrier parameter.
/// @return The exit status.
template <typename Scalar>
ExitStatus feasibility_restoration(
    const InteriorPointMatrixCallbacks<Scalar>& matrix_callbacks, bool is_nlp,
    std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options& options, Eigen::Vector<Scalar, Eigen::Dynamic>& x,
    Eigen::Vector<Scalar, Eigen::Dynamic>& v, Scalar sqrt_μ) {
  // Feasibility restoration
  //
  //        min  ρ Σ (pₑ + nₑ + pᵢ + nᵢ) + ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)
  //         x
  //       pₑ,nₑ
  //       pᵢ,nᵢ
  //
  //   s.t. cₑ(x) - pₑ + nₑ = 0
  //        cᵢ(x) - pᵢ + nᵢ ≥ 0
  //        pₑ ≥ 0
  //        nₑ ≥ 0
  //        pᵢ ≥ 0
  //        nᵢ ≥ 0
  //
  // where ρ = 1000, ζ = √μ where μ is the barrier parameter, xᵣ is original
  // iterate before feasibility restoration, and Dᵣ is a scaling matrix defined
  // by
  //
  //   Dᵣ = diag(min(1, 1/xᵣ[i]²) for i in x.rows())

  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using DiagonalMatrix = Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using SparseVector = Eigen::SparseVector<Scalar>;

  using std::sqrt;

  const auto& matrices = matrix_callbacks;
  const auto& num_vars = matrices.num_decision_variables;
  const auto& num_ineq = matrices.num_inequality_constraints;

  constexpr Scalar ρ(1e3);
  const Scalar ζ = sqrt_μ;
  const Scalar μ = sqrt_μ * sqrt_μ;

  // eᵛ
  DenseVector exp_v{v.array().exp().matrix()};
  // e⁻ᵛ
  DenseVector exp_neg_v = exp_v.cwiseInverse();
  // s = √(μ)e⁻ᵛ
  DenseVector s = sqrt_μ * exp_neg_v;
  // z = √(μ)eᵛ
  DenseVector z = sqrt_μ * exp_v;

  const DenseVector c_i = matrices.c_i(x);

  const auto& x_r = x;
  const auto [p_i_0, n_i_0] = compute_p_n((c_i - s).eval(), ρ, μ);

  // Dᵣ = diag(min(1, 1/xᵣ[i]²) for i in x.rows())
  const DiagonalMatrix D_r =
      x.cwiseSquare().cwiseInverse().cwiseMin(Scalar(1)).asDiagonal();

  DenseVector fr_x{num_vars + 2 * num_ineq};
  fr_x << x, p_i_0, n_i_0;

  DenseVector fr_v{v.rows() + 2 * num_ineq};
  fr_v.segment(0, v.rows()) = v;
  fr_v.segment(v.rows(), 2 * num_ineq).setZero();

  Scalar fr_sqrt_μ =
      std::max(sqrt_μ, sqrt((c_i - s).template lpNorm<Eigen::Infinity>()));

  InteriorPointMatrixCallbacks<Scalar> fr_matrix_callbacks{
      static_cast<int>(fr_x.rows()),
      0,
      static_cast<int>(fr_v.rows()),
      [&](const DenseVector& x_p) -> Scalar {
        auto x = x_p.segment(0, num_vars);

        // Cost function
        //
        //   ρ Σ (pₑ + nₑ + pᵢ + nᵢ) + ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)
        auto diff = x - x_r;
        return ρ * x_p.segment(num_vars, 2 * num_ineq).array().sum() +
               ζ / Scalar(2) * diff.transpose() * D_r * diff;
      },
      [&](const DenseVector& x_p) -> SparseVector {
        auto x = x_p.segment(0, num_vars);

        // Cost function gradient
        //
        //   [ζDᵣ(x − xᵣ)]
        //   [     ρ     ]
        //   [     ρ     ]
        //   [     ρ     ]
        //   [     ρ     ]
        DenseVector g{x_p.rows()};
        g.segment(0, num_vars) = ζ * D_r * (x - x_r);
        g.segment(num_vars, 2 * num_ineq).setConstant(ρ);
        return g.sparseView();
      },
      [&](const DenseVector& x_p, const DenseVector& v_p,
          Scalar sqrt_μ) -> SparseMatrix {
        auto x = x_p.segment(0, num_vars);
        auto v = v_p.segment(0, num_ineq);

        // Cost function Hessian
        //
        //   [ζDᵣ  0  0  0  0]
        //   [ 0   0  0  0  0]
        //   [ 0   0  0  0  0]
        //   [ 0   0  0  0  0]
        //   [ 0   0  0  0  0]
        gch::small_vector<Eigen::Triplet<Scalar>> triplets;
        triplets.reserve(x_p.rows());
        append_as_triplets(triplets, 0, 0, {SparseMatrix{ζ * D_r}});
        SparseMatrix d2f_dx2{x_p.rows(), x_p.rows()};
        d2f_dx2.setFromSortedTriplets(triplets.begin(), triplets.end());

        // Constraint part of original problem's Lagrangian Hessian
        //
        //   −∇ₓₓ²zᵀcᵢ(x)
        auto H_c = matrices.H_c(x, v, sqrt_μ);
        H_c.resize(x_p.rows(), x_p.rows());

        // Lagrangian Hessian
        //
        //   [ζDᵣ  0  0  0  0]
        //   [ 0   0  0  0  0]
        //   [ 0   0  0  0  0] − ∇ₓₓ²yᵀcₑ(x) − ∇ₓₓ²zᵀcᵢ(x)
        //   [ 0   0  0  0  0]
        //   [ 0   0  0  0  0]
        return d2f_dx2 + H_c;
      },
      [&](const DenseVector& x_p, [[maybe_unused]] const DenseVector& v_p,
          [[maybe_unused]] Scalar sqrt_μ) -> SparseMatrix {
        return SparseMatrix{x_p.rows(), x_p.rows()};
      },
      [&](const DenseVector& x_p) -> DenseVector {
        auto x = x_p.segment(0, num_vars);
        auto p_i = x_p.segment(num_vars, num_ineq);
        auto n_i = x_p.segment(num_vars + num_ineq, num_ineq);

        // Inequality constraints
        //
        //   cᵢ(x) - pᵢ + nᵢ ≥ 0
        //   pₑ ≥ 0
        //   nₑ ≥ 0
        //   pᵢ ≥ 0
        //   nᵢ ≥ 0
        DenseVector c_i_p{c_i.rows() + 2 * num_ineq};
        c_i_p.segment(0, num_ineq) = matrices.c_i(x) - p_i + n_i;
        c_i_p.segment(p_i.rows(), 2 * num_ineq) =
            x_p.segment(num_vars, 2 * num_ineq);
        return c_i_p;
      },
      [&](const DenseVector& x_p) -> SparseMatrix {
        auto x = x_p.segment(0, num_vars);

        // Inequality constraint Jacobian
        //
        //   [Aᵢ  0  0  −I  I]
        //   [0   I  0   0  0]
        //   [0   0  I   0  0]
        //   [0   0  0   I  0]
        //   [0   0  0   0  I]

        SparseMatrix A_i = matrices.A_i(x);

        gch::small_vector<Eigen::Triplet<Scalar>> triplets;
        triplets.reserve(A_i.nonZeros() + 4 * num_ineq);

        // Column 0
        append_as_triplets(triplets, 0, 0, {A_i});

        // Columns 1 and 2
        append_diagonal_as_triplets(triplets, num_ineq, num_vars,
                                    DenseVector::Constant(0, Scalar(1)).eval());

        SparseMatrix I_ineq{
            DenseVector::Constant(num_ineq, Scalar(1)).asDiagonal()};

        // Column 3
        SparseMatrix Z_col3{0, num_ineq};
        append_as_triplets(triplets, 0, num_vars,
                           {(-I_ineq).eval(), Z_col3, I_ineq});

        // Column 4
        SparseMatrix Z_col4{num_ineq, num_ineq};
        append_as_triplets(triplets, 0, num_vars + num_ineq,
                           {I_ineq, Z_col4, I_ineq});

        SparseMatrix A_i_p{3 * num_ineq, x_p.rows()};
        A_i_p.setFromSortedTriplets(triplets.begin(), triplets.end());
        return A_i_p;
      }};

  auto status = interior_point<Scalar>(fr_matrix_callbacks, is_nlp,
                                       iteration_callbacks, options, true,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
                                       {},
#endif
                                       fr_x, fr_v, fr_sqrt_μ);

  x = fr_x.segment(0, x.rows());
  v = fr_v.segment(0, v.rows());

  // eᵛ
  exp_v = v.array().exp().matrix();
  // e⁻ᵛ
  exp_neg_v = exp_v.cwiseInverse();
  // s = √(μ)e⁻ᵛ
  s = sqrt_μ * exp_neg_v;
  // z = √(μ)eᵛ
  z = sqrt_μ * exp_v;

  if (status == ExitStatus::CALLBACK_REQUESTED_STOP) {
    auto g = matrices.g(x);
    auto A_i = matrices.A_i(x);

    auto z_estimate = lagrange_multiplier_estimate(g, A_i, s, μ);

    // v = ln(1/√(μ) z)
    v = (z_estimate / sqrt_μ).array().log().matrix();

    return ExitStatus::SUCCESS;
  } else if (status == ExitStatus::SUCCESS) {
    return ExitStatus::LOCALLY_INFEASIBLE;
  } else {
    return ExitStatus::FEASIBILITY_RESTORATION_FAILED;
  }
}

}  // namespace slp

#include "sleipnir/optimization/solver/interior_point.hpp"
