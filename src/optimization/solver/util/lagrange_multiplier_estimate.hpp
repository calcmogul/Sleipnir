// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <utility>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "sleipnir/util/small_vector.hpp"

namespace slp {

/**
 * Lagrange multiplier estimates.
 */
struct LagrangeMultiplierEstimate {
  /// Equality constraint dual estimate.
  Eigen::VectorXd y;
  /// Inequality constraint dual estimate.
  Eigen::VectorXd z;
};

/**
 * Estimates Lagrange multipliers for SQP.
 *
 * @param g Gradient of the cost function ∇f.
 * @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
 *   current iterate.
 */
inline Eigen::VectorXd lagrange_multiplier_estimate(
    const Eigen::SparseVector<double>& g,
    const Eigen::SparseMatrix<double>& A_e) {
  // Lagrange multiplier estimates
  //
  //   y = (AₑAₑᵀ)⁻¹Aₑ∇f
  //
  // See equation (19.37) of [1].
  return Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>{A_e *
                                                            A_e.transpose()}
      .solve(A_e * g);
}

/**
 * Estimates Lagrange multipliers for interior-point method.
 *
 * @param g Gradient of the cost function ∇f.
 * @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
 *   current iterate.
 * @param A_i The problem's inequality constraint Jacobian Aᵢ(x) evaluated at
 *   the current iterate.
 * @param s Inequality constraint slack variables.
 * @param μ Barrier parameter.
 */
inline LagrangeMultiplierEstimate lagrange_multiplier_estimate(
    const Eigen::SparseVector<double>& g,
    const Eigen::SparseMatrix<double>& A_e,
    const Eigen::SparseMatrix<double>& A_i, const Eigen::VectorXd& s,
    double μ) {
  // Lagrange multiplier estimates
  //
  //   [y] = (ÂÂᵀ)⁻¹Â[ ∇f]
  //   [z]           [−μe]
  //
  //   where Â = [Aₑ   0]
  //             [Aᵢ  −S]
  //
  // See equation (19.37) of [1].

  small_vector<Eigen::Triplet<double>> triplets;

  // Â = [Aₑ   0]
  //     [Aᵢ  −S]
  triplets.clear();
  triplets.reserve(A_e.nonZeros() + A_i.nonZeros() + s.rows());
  for (int col = 0; col < A_e.cols(); ++col) {
    // Append column of Aₑ in top-left quadrant
    for (Eigen::SparseMatrix<double>::InnerIterator it{A_e, col}; it; ++it) {
      triplets.emplace_back(it.row(), it.col(), it.value());
    }
    // Append column of Aᵢ in bottom-left quadrant
    for (Eigen::SparseMatrix<double>::InnerIterator it{A_i, col}; it; ++it) {
      triplets.emplace_back(A_e.rows() + it.row(), it.col(), it.value());
    }
  }
  // Append −S in bottom-right quadrant
  for (int i = 0; i < s.rows(); ++i) {
    triplets.emplace_back(A_e.rows() + i, A_e.cols() + i, -s(i));
  }
  Eigen::SparseMatrix<double> A_hat{A_e.rows() + A_i.rows(),
                                    A_e.cols() + s.rows()};
  A_hat.setFromSortedTriplets(triplets.begin(), triplets.end(),
                              [](const auto&, const auto& b) { return b; });

  // lhs = ÂÂᵀ
  Eigen::SparseMatrix<double> lhs = A_hat * A_hat.transpose();

  // rhs = Â[ ∇f]
  //        [−μe]
  Eigen::VectorXd rhs_temp{g.rows() + s.rows()};
  rhs_temp.segment(0, g.rows()) = g;
  rhs_temp.segment(g.rows(), s.rows()).setConstant(-μ);
  Eigen::VectorXd rhs = A_hat * rhs_temp;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> yz_estimator{lhs};
  Eigen::VectorXd sol = yz_estimator.solve(rhs);
  Eigen::VectorXd y = sol.segment(0, A_e.rows());
  Eigen::VectorXd z = sol.segment(A_e.rows(), s.rows());

  // A requirement for the convergence proof is that the "primal-dual barrier
  // term Hessian" Σₖ does not deviate arbitrarily much from the "primal
  // Hessian" μⱼSₖ⁻². We ensure this by resetting
  //
  //   zₖ₊₁⁽ⁱ⁾ = max(min(zₖ₊₁⁽ⁱ⁾, κ_Σ μⱼ/sₖ₊₁⁽ⁱ⁾), μⱼ/(κ_Σ sₖ₊₁⁽ⁱ⁾))
  //
  // for some fixed κ_Σ ≥ 1 after each step. See equation (16) of [2].
  for (int row = 0; row < z.rows(); ++row) {
    // Barrier parameter scale factor for inequality constraint Lagrange
    // multiplier safeguard
    constexpr double κ_Σ = 1e10;

    z[row] = std::max(std::min(z[row], κ_Σ * μ / s[row]), μ / (κ_Σ * s[row]));
  }

  return {std::move(y), std::move(z)};
}

}  // namespace slp
