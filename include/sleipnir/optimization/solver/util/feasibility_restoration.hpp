// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/interior_point_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/sqp_matrix_callbacks.hpp"

namespace slp {

template <typename Scalar>
ExitStatus interior_point(
    const InteriorPointMatrixCallbacks<Scalar>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options& options, bool use_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::Vector<Scalar, Eigen::Dynamic>& x,
    Eigen::Vector<Scalar, Eigen::Dynamic>& s);

/// Computes initial values for p and n in feasibility restoration.
///
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
  //   -4ac = μ c(x) 2ρ
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
    Scalar _a = Scalar(2) * ρ;
    Scalar _b = ρ * c[row] - μ;
    Scalar _c = -μ * c[row] / Scalar(2);

    n[row] = -_b * sqrt(_b * _b - Scalar(4) * _a * _c) / (Scalar(2) * _a);
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
/// @param[in] iteration_callbacks The list of callbacks to call at the
///     beginning of each iteration.
/// @param[in] options Solver options.
/// @param[in,out] x The current decision variables from the normal solve.
/// @param[in,out] y The current equality constraint duals from the normal
///     solve.
/// @return The exit status.
template <typename Scalar>
ExitStatus feasibility_restoration(
    const SQPMatrixCallbacks<Scalar>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options& options, Eigen::Vector<Scalar, Eigen::Dynamic>& x,
    Eigen::Vector<Scalar, Eigen::Dynamic>& y) {
  // Feasibility restoration
  //
  //        min  ρ Σ (pₑ + nₑ) + ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)
  //         x
  //       pₑ,nₑ
  //
  //   s.t. cₑ(x) - pₑ + nₑ = 0
  //        pₑ ≥ 0
  //        nₑ ≥ 0
  //
  // where ρ = 1000, ζ = √μ where μ is the barrier parameter, xᵣ is original
  // iterate before feasibility restoration, and Dᵣ is a scaling matrix defined
  // by
  //
  //   Dᵣ = diag(min(1, 1/|xᵣ⁽¹⁾|), …, min(1, 1/|xᵣ|⁽ⁿ⁾))

  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using SparseVector = Eigen::SparseVector<Scalar>;

  using std::abs;
  using std::sqrt;

  constexpr Scalar ρ(1000.0);
  Scalar μ(options.tolerance / 10.0);
  Scalar ζ = sqrt(μ);

  DenseVector c_e = matrix_callbacks.c_e(x);

  int num_decision_variables = x.rows();
  int num_equality_constraints = c_e.rows();

  auto [p_e, n_e] = compute_p_n(c_e, ρ, μ);

  // Dᵣ = diag(min(1, 1/|xᵣ⁽¹⁾|), …, min(1, 1/|xᵣ|⁽ⁿ⁾)
  DenseVector D_r = x.cwiseAbs().cwiseInverse().cwiseMin(Scalar(1));

  DenseVector fr_x{num_decision_variables + 2 * num_equality_constraints};
  fr_x.segment(0, num_decision_variables) = x;
  fr_x.segment(num_decision_variables, num_equality_constraints) = p_e;
  fr_x.segment(num_decision_variables + num_equality_constraints,
               num_equality_constraints) = n_e;

  DenseVector fr_s{2 * num_equality_constraints};
  fr_s.setOnes();

  auto status = interior_point<Scalar>(
      InteriorPointMatrixCallbacks<Scalar>{
          [&](const DenseVector& x_p) -> Scalar {
            auto _x = x_p.segment(0, num_decision_variables);

            // Cost function
            //
            //   ρ Σ (pₑ + nₑ) + ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)

            DenseVector diff = _x - x;
            return ρ * (p_e.array().sum() + n_e.array().sum()) +
                   ζ / Scalar(2) * diff.transpose() * D_r.asDiagonal() * diff;
          },
          [&](const DenseVector& x_p) -> SparseVector {
            auto _x = x_p.segment(0, num_decision_variables);

            // Cost function gradient
            //
            //   [ζDᵣ(x − xᵣ)]
            //   [     ρ     ]
            //   [     ρ     ]
            DenseVector g{x_p.rows()};
            g.segment(0, num_decision_variables) =
                ζ * D_r.asDiagonal() * (_x - x);
            g.segment(num_decision_variables, 2 * num_equality_constraints)
                .setConstant(ρ);
            return g.sparseView();
          },
          [&](const DenseVector& x_p, const DenseVector& y,
              [[maybe_unused]] const DenseVector& z) -> SparseMatrix {
            auto _x = x_p.segment(0, num_decision_variables);

            // Cost function Hessian
            //
            //   [ζDᵣ  0  0]
            //   [ 0   0  0]
            //   [ 0   0  0]
            SparseMatrix d2f_dx2{D_r.asDiagonal() * ζ};
            d2f_dx2.resize(x_p.rows(), x_p.rows());

            // Constraint part of original problem's Lagrangian Hessian
            //
            //   −∇ₓₓ²yᵀcₑ(x)
            auto H_c = matrix_callbacks.H_c(_x, y);
            H_c.resize(x_p.rows(), x_p.rows());

            // Lagrangian Hessian
            //
            //   [ζDᵣ  0  0]
            //   [ 0   0  0] − ∇ₓₓ²yᵀcₑ(x)
            //   [ 0   0  0]
            return d2f_dx2 + H_c;
          },
          [&](const DenseVector& x_p, [[maybe_unused]] const DenseVector& y,
              [[maybe_unused]] const DenseVector& z) -> SparseMatrix {
            return SparseMatrix{x_p.rows(), x_p.rows()};
          },
          [&](const DenseVector& x_p) -> DenseVector {
            auto _x = x_p.segment(0, num_decision_variables);
            auto _p_e =
                x_p.segment(num_decision_variables, num_equality_constraints);
            auto _n_e =
                x_p.segment(num_decision_variables + num_equality_constraints,
                            num_equality_constraints);

            // Equality constraints
            //
            //   cₑ(x) - pₑ + nₑ = 0
            return matrix_callbacks.c_e(_x) - _p_e + _n_e;
          },
          [&](const DenseVector& x_p) -> SparseMatrix {
            auto _x = x_p.segment(0, num_decision_variables);

            // Equality constraint Jacobian
            //
            //   [Aₑ  −I  I]

            SparseMatrix A_e = matrix_callbacks.A_e(_x);

            gch::small_vector<Eigen::Triplet<Scalar>> triplets;
            triplets.reserve(A_e.nonZeros() + 2 * num_equality_constraints);

            for (int col = 0; col < A_e.cols(); ++col) {
              for (typename SparseMatrix::InnerIterator it{A_e, col}; it;
                   ++it) {
                triplets.emplace_back(it.row(), it.col(), it.value());
              }
            }
            for (int row = 0; row < num_equality_constraints; ++row) {
              triplets.emplace_back(row, num_decision_variables + row,
                                    Scalar(-1));
            }
            for (int row = 0; row < num_equality_constraints; ++row) {
              triplets.emplace_back(
                  row, num_decision_variables + num_equality_constraints + row,
                  Scalar(1));
            }

            SparseMatrix A_e_p{num_equality_constraints, x_p.rows()};
            A_e_p.setFromSortedTriplets(triplets.begin(), triplets.end());
            return A_e_p;
          },
          [&](const DenseVector& x_p) -> DenseVector {
            // Inequality constraints
            //
            //   pₑ ≥ 0
            //   nₑ ≥ 0
            return x_p.segment(num_decision_variables,
                               2 * num_equality_constraints);
          },
          [&](const DenseVector& x_p) -> SparseMatrix {
            // Inequality constraint Jacobian
            //
            //   [0  I  0]
            //   [0  0  I]

            gch::small_vector<Eigen::Triplet<Scalar>> triplets;
            triplets.reserve(2 * num_equality_constraints);

            for (int row = 0; row < 2 * num_equality_constraints; ++row) {
              triplets.emplace_back(row, num_decision_variables + row,
                                    Scalar(1));
            }

            SparseMatrix A_i_p{2 * num_equality_constraints, x_p.rows()};
            A_i_p.setFromSortedTriplets(triplets.begin(), triplets.end());
            return A_i_p;
          }},
      iteration_callbacks, options, true,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
      {},
#endif
      fr_x, fr_s);

  x = fr_x.segment(0, num_decision_variables);

  if (status == ExitStatus::CALLBACK_REQUESTED_STOP) {
    // Lagrange multiplier estimates
    //
    //   y = (AₑAₑᵀ)⁻¹Aₑ∇f
    //
    // See equation (19.37) of [1].

    auto A_e = matrix_callbacks.A_e(x);
    auto g = matrix_callbacks.g(x);

    // lhs = AₑAₑᵀ
    SparseMatrix lhs = A_e * A_e.transpose();

    // rhs = Aₑ∇f
    DenseVector rhs = A_e * g;

    Eigen::SimplicialLDLT<SparseMatrix> yEstimator{lhs};
    y = yEstimator.solve(rhs);

    return ExitStatus::SUCCESS;
  } else if (status == ExitStatus::SUCCESS) {
    return ExitStatus::LOCALLY_INFEASIBLE;
  } else {
    return ExitStatus::FEASIBILITY_RESTORATION_FAILED;
  }
}

/// Finds the iterate that minimizes the constraint violation while not
/// deviating too far from the starting point. This is a fallback procedure when
/// the normal interior-point method fails to converge to a feasible point.
///
/// @param[in] matrix_callbacks Matrix callbacks.
/// @param[in] μ Barrier parameter.
/// @param[in] iteration_callbacks The list of callbacks to call at the
///     beginning of each iteration.
/// @param[in] options Solver options.
/// @param[in,out] x The current decision variables from the normal solve.
/// @param[in,out] s The current inequality constraint slack variables from the
///     normal solve.
/// @param[in,out] y The current equality constraint duals from the normal
///     solve.
/// @param[in,out] z The current inequality constraint duals from the normal
///     solve.
/// @return The exit status.
template <typename Scalar>
ExitStatus feasibility_restoration(
    const InteriorPointMatrixCallbacks<Scalar>& matrix_callbacks, Scalar μ,
    std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options& options, Eigen::Vector<Scalar, Eigen::Dynamic>& x,
    Eigen::Vector<Scalar, Eigen::Dynamic>& s,
    Eigen::Vector<Scalar, Eigen::Dynamic>& y,
    Eigen::Vector<Scalar, Eigen::Dynamic>& z) {
  // Feasibility restoration
  //
  //        min  ρ Σ (pₑ + nₑ + pᵢ + nᵢ) + ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)
  //         x
  //       pₑ,nₑ
  //       pᵢ,nᵢ
  //
  //   s.t. cₑ(x) - pₑ + nₑ = 0
  //        cᵢ(x) - s - pᵢ + nᵢ = 0
  //        pₑ ≥ 0
  //        nₑ ≥ 0
  //        pᵢ ≥ 0
  //        nᵢ ≥ 0
  //
  // where ρ = 1000, ζ = √μ where μ is the barrier parameter, xᵣ is original
  // iterate before feasibility restoration, and Dᵣ is a scaling matrix defined
  // by
  //
  //   Dᵣ = diag(min(1, 1/|xᵣ⁽¹⁾|), …, min(1, 1/|xᵣ|⁽ⁿ⁾)

  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using SparseVector = Eigen::SparseVector<Scalar>;

  using std::abs;
  using std::sqrt;

  constexpr Scalar ρ(1000.0);
  Scalar ζ = sqrt(μ);

  DenseVector c_e = matrix_callbacks.c_e(x);
  DenseVector c_i = matrix_callbacks.c_i(x);

  int num_decision_variables = x.rows();
  int num_equality_constraints = c_e.rows();
  int num_inequality_constraints = c_i.rows();

  auto [p_e, n_e] = compute_p_n(c_e, ρ, μ);
  auto [p_i, n_i] = compute_p_n((c_i - s).eval(), ρ, μ);

  // Dᵣ = diag(min(1, 1/|xᵣ⁽¹⁾|), …, min(1, 1/|xᵣ|⁽ⁿ⁾))
  DenseVector D_r = x.cwiseAbs().cwiseInverse().cwiseMin(Scalar(1));

  DenseVector fr_x{num_decision_variables + 2 * num_equality_constraints +
                   2 * num_inequality_constraints};
  fr_x.segment(0, num_decision_variables) = x;
  fr_x.segment(num_decision_variables, num_equality_constraints) = p_e;
  fr_x.segment(num_decision_variables + num_equality_constraints,
               num_equality_constraints) = n_e;
  fr_x.segment(num_decision_variables + 2 * num_equality_constraints,
               num_inequality_constraints) = p_i;
  fr_x.segment(num_decision_variables + 2 * num_equality_constraints +
                   num_inequality_constraints,
               num_inequality_constraints) = n_i;

  DenseVector fr_s{2 * num_equality_constraints +
                   2 * num_inequality_constraints};
  fr_s.setOnes();

  auto status = interior_point<Scalar>(
      InteriorPointMatrixCallbacks<Scalar>{
          [&](const DenseVector& x_p) -> Scalar {
            auto _x = x_p.segment(0, num_decision_variables);

            // Cost function
            //
            //   ρ Σ (pₑ + nₑ + pᵢ + nᵢ) + ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)
            auto diff = _x - x;
            return ρ * (p_e.array().sum() + n_e.array().sum() +
                        p_i.array().sum() + n_i.array().sum()) +
                   ζ / Scalar(2) * diff.transpose() * D_r.asDiagonal() * diff;
          },
          [&](const DenseVector& x_p) -> SparseVector {
            auto _x = x_p.segment(0, num_decision_variables);

            // Cost function gradient
            //
            //   [ζDᵣ(x − xᵣ)]
            //   [     ρ     ]
            //   [     ρ     ]
            //   [     ρ     ]
            //   [     ρ     ]
            DenseVector g{x_p.rows()};
            g.segment(0, num_decision_variables) =
                ζ * D_r.asDiagonal() * (_x - x);
            g.segment(
                 num_decision_variables,
                 2 * num_equality_constraints + 2 * num_inequality_constraints)
                .setConstant(ρ);
            return g.sparseView();
          },
          [&](const DenseVector& x_p, const DenseVector& y,
              [[maybe_unused]] const DenseVector& z) -> SparseMatrix {
            auto _x = x_p.segment(0, num_decision_variables);
            auto _y = y.segment(0, num_equality_constraints);
            auto _z =
                y.segment(num_equality_constraints, num_inequality_constraints);

            // Cost function Hessian
            //
            //   [ζDᵣ  0  0  0  0]
            //   [ 0   0  0  0  0]
            //   [ 0   0  0  0  0]
            //   [ 0   0  0  0  0]
            //   [ 0   0  0  0  0]
            SparseMatrix d2f_dx2{D_r.asDiagonal() * ζ};
            d2f_dx2.resize(x_p.rows(), x_p.rows());

            // Constraint part of original problem's Lagrangian Hessian
            //
            //   −∇ₓₓ²yᵀcₑ(x) − ∇ₓₓ²zᵀcᵢ(x)
            auto H_c = matrix_callbacks.H_c(_x, _y, _z);
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
          [&](const DenseVector& x_p, [[maybe_unused]] const DenseVector& y,
              [[maybe_unused]] const DenseVector& z) -> SparseMatrix {
            return SparseMatrix{x_p.rows(), x_p.rows()};
          },
          [&](const DenseVector& x_p) -> DenseVector {
            auto _x = x_p.segment(0, num_decision_variables);
            auto _p_e =
                x_p.segment(num_decision_variables, num_equality_constraints);
            auto _n_e =
                x_p.segment(num_decision_variables + num_equality_constraints,
                            num_equality_constraints);
            auto _p_i = x_p.segment(
                num_decision_variables + 2 * num_equality_constraints,
                num_inequality_constraints);
            auto _n_i = x_p.segment(num_decision_variables +
                                        2 * num_equality_constraints +
                                        num_inequality_constraints,
                                    num_inequality_constraints);

            // Equality constraints
            //
            //   cₑ(x) - pₑ + nₑ = 0
            //   cᵢ(x) - s - pᵢ + nᵢ = 0
            DenseVector c_e_p{num_equality_constraints +
                              num_inequality_constraints};
            auto c_e = matrix_callbacks.c_e(_x);
            c_e_p.segment(0, num_equality_constraints) = c_e - _p_e + _n_e;
            auto c_i = matrix_callbacks.c_e(_x);
            c_e_p.segment(num_equality_constraints,
                          num_inequality_constraints) = c_i - s - _p_i + _n_i;
            return c_e_p;
          },
          [&](const DenseVector& x_p) -> SparseMatrix {
            auto _x = x_p.segment(0, num_decision_variables);

            // Equality constraint Jacobian
            //
            //   [Aₑ  −I  I   0  0]
            //   [Aᵢ   0  0  −I  I]

            SparseMatrix A_e = matrix_callbacks.A_e(_x);
            SparseMatrix A_i = matrix_callbacks.A_i(_x);

            gch::small_vector<Eigen::Triplet<Scalar>> triplets;
            triplets.reserve(A_e.nonZeros() + A_i.nonZeros() +
                             2 * num_equality_constraints +
                             2 * num_inequality_constraints);

            for (int col = 0; col < num_decision_variables; ++col) {
              // Append column of Aₑ
              for (typename SparseMatrix::InnerIterator it{A_e, col}; it;
                   ++it) {
                triplets.emplace_back(it.row(), it.col(), it.value());
              }

              // Append column of Aᵢ
              for (typename SparseMatrix::InnerIterator it{A_i, col}; it;
                   ++it) {
                triplets.emplace_back(num_equality_constraints + it.row(),
                                      it.col(), it.value());
              }
            }

            for (int row = 0; row < num_equality_constraints; ++row) {
              triplets.emplace_back(row, num_decision_variables + row,
                                    Scalar(-1));
            }
            for (int row = 0; row < num_equality_constraints; ++row) {
              triplets.emplace_back(
                  row, num_decision_variables + num_equality_constraints + row,
                  Scalar(1));
            }
            for (int row = 0; row < num_inequality_constraints; ++row) {
              triplets.emplace_back(
                  num_equality_constraints + row,
                  num_decision_variables + 2 * num_equality_constraints + row,
                  Scalar(-1));
            }
            for (int row = 0; row < num_inequality_constraints; ++row) {
              triplets.emplace_back(num_equality_constraints + row,
                                    num_decision_variables +
                                        2 * num_equality_constraints +
                                        num_inequality_constraints + row,
                                    Scalar(1));
            }

            SparseMatrix A_e_p{
                num_equality_constraints + num_inequality_constraints,
                x_p.rows()};
            A_e_p.setFromSortedTriplets(triplets.begin(), triplets.end());
            return A_e_p;
          },
          [&](const DenseVector& x_p) -> DenseVector {
            // Inequality constraints
            //
            //   pₑ ≥ 0
            //   nₑ ≥ 0
            //   pᵢ ≥ 0
            //   nᵢ ≥ 0
            return x_p.segment(
                num_decision_variables,
                2 * num_equality_constraints + 2 * num_inequality_constraints);
          },
          [&](const DenseVector& x_p) -> SparseMatrix {
            // Inequality constraint Jacobian
            //
            //   [0  I  0  0  0]
            //   [0  0  I  0  0]
            //   [0  0  0  I  0]
            //   [0  0  0  0  I]

            gch::small_vector<Eigen::Triplet<Scalar>> triplets;
            triplets.reserve(2 * num_equality_constraints +
                             2 * num_inequality_constraints);

            for (int row = 0; row < 2 * num_equality_constraints +
                                        2 * num_inequality_constraints;
                 ++row) {
              triplets.emplace_back(row, num_decision_variables + row,
                                    Scalar(1));
            }

            SparseMatrix A_i_p{
                2 * num_equality_constraints + 2 * num_inequality_constraints,
                x_p.rows()};
            A_i_p.setFromSortedTriplets(triplets.begin(), triplets.end());
            return A_i_p;
          }},
      iteration_callbacks, options, true,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
      {},
#endif
      fr_x, fr_s);

  x = fr_x.segment(0, x.rows());
  // s = fr_s.segment(0, inequality_constraints.size());

  if (status == ExitStatus::CALLBACK_REQUESTED_STOP) {
    // Lagrange multiplier estimates
    //
    //   [y] = (ÂÂᵀ)⁻¹Â[ ∇f]
    //   [z]           [−μe]
    //
    //   where Â = [Aₑ   0]
    //             [Aᵢ  −S]
    //
    // See equation (19.37) of [1].

    auto A_e = matrix_callbacks.A_e(x);
    auto A_i = matrix_callbacks.A_i(x);
    auto g = matrix_callbacks.g(x);

    // Â = [Aₑ   0]
    //     [Aᵢ  −S]
    gch::small_vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(A_e.nonZeros() + A_i.nonZeros() + s.rows());
    for (int col = 0; col < A_e.cols(); ++col) {
      // Append column of Aₑ in top-left quadrant
      for (typename SparseMatrix::InnerIterator it{A_e, col}; it; ++it) {
        triplets.emplace_back(it.row(), it.col(), it.value());
      }
      // Append column of Aᵢ in bottom-left quadrant
      for (typename SparseMatrix::InnerIterator it{A_i, col}; it; ++it) {
        triplets.emplace_back(A_e.rows() + it.row(), it.col(), it.value());
      }
    }
    // Append −S in bottom-right quadrant
    for (int i = 0; i < s.rows(); ++i) {
      triplets.emplace_back(A_e.rows() + i, A_e.cols() + i, -s(i));
    }
    SparseMatrix Ahat{A_e.rows() + A_i.rows(), A_e.cols() + s.rows()};
    Ahat.setFromSortedTriplets(triplets.begin(), triplets.end(),
                               [](const auto&, const auto& b) { return b; });

    // lhs = ÂÂᵀ
    SparseMatrix lhs = Ahat * Ahat.transpose();

    // rhs = Â[ ∇f]
    //        [−μe]
    DenseVector rhsTemp{g.rows() + s.rows()};
    rhsTemp.block(0, 0, g.rows(), 1) = g;
    rhsTemp.block(g.rows(), 0, s.rows(), 1) = -μ * DenseVector::Ones(s.rows());
    DenseVector rhs = Ahat * rhsTemp;

    Eigen::SimplicialLDLT<SparseMatrix> yzEstimator{lhs};
    DenseVector sol = yzEstimator.solve(rhs);

    y = sol.block(0, 0, y.rows(), 1);
    z = sol.block(y.rows(), 0, z.rows(), 1);

    return ExitStatus::SUCCESS;
  } else if (status == ExitStatus::SUCCESS) {
    return ExitStatus::LOCALLY_INFEASIBLE;
  } else {
    return ExitStatus::FEASIBILITY_RESTORATION_FAILED;
  }
}

}  // namespace slp

#include "sleipnir/optimization/solver/interior_point.hpp"
