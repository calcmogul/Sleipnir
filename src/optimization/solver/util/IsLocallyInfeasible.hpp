// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

// See docs/algorithms.md#Works_cited for citation definitions

namespace sleipnir {

/**
 * Returns true if the problem's equality constraints are locally infeasible.
 *
 * @param A_e The problem's equality constraint Jacobian Aₑ(x) evaluated at the
 *   current iterate.
 * @param c_e The problem's equality constraints cₑ(x) evaluated at the current
 *   iterate.
 * @param y The problem's equality constraint duals.
 */
inline bool IsEqualityLocallyInfeasible(const Eigen::SparseMatrix<double>& A_e,
                                        const Eigen::VectorXd& c_e,
                                        const Eigen::VectorXd& y) {
  // The equality constraints are locally infeasible if
  //
  //   cₑᵀy > 0
  //   Γ_far ≤ ε_far
  //   Γ_inf ≤ ε_inf
  //
  // where
  //
  //           |Aₑᵀy|₁
  //   Γ_far = -------
  //            cₑᵀy
  //
  //           |Aₑᵀy|₁
  //   Γ_inf = -------
  //            |y|₁
  //
  // See equations (18a-c) of [5].

  if (A_e.rows() == 0) {
    return false;
  }

  constexpr double ε_far = 1e-3;
  constexpr double ε_inf = 1e-6;

  double c_e_y = (c_e.transpose() * y)(0);
  if (c_e_y == 0.0) {
    return false;
  }

  double A_e_y_lp1 = (A_e.transpose() * y).lpNorm<1>();
  double Γ_far = A_e_y_lp1 / c_e_y;
  double Γ_inf = A_e_y_lp1 / y.lpNorm<1>();

  return Γ_far <= ε_far && Γ_inf <= ε_inf;
}

/**
 * Returns true if the problem's inequality constraints are locally infeasible.
 *
 * @param A_i The problem's inequality constraint Jacobian Aᵢ(x) evaluated at
 *   the current iterate.
 * @param c_i The problem's inequality constraints cᵢ(x) evaluated at the
 *   current iterate.
 * @param s The problem's inequality constraint slack variables.
 * @param z The problem's inequality constraint duals.
 */
inline bool IsInequalityLocallyInfeasible(
    const Eigen::SparseMatrix<double>& A_i, const Eigen::VectorXd& c_i,
    const Eigen::VectorXd& s, const Eigen::VectorXd& z) {
  // The inequality constraints are locally infeasible if
  //
  //   cᵢᵀz > 0
  //   Γ_far ≤ ε_far
  //   Γ_inf ≤ ε_inf
  //
  // where
  //
  //           |Aᵢᵀz|₁
  //   Γ_far = -------
  //            cᵢᵀz
  //
  //           |Aᵢᵀz|₁ + sᵀz
  //   Γ_inf = -------------
  //               |z|₁
  //
  // See equations (18a-c) of [5].

  if (A_i.rows() == 0) {
    return false;
  }

  constexpr double ε_far = 1e-3;
  constexpr double ε_inf = 1e-6;

  double c_i_z = (c_i.transpose() * z)(0);
  if (c_i_z == 0.0) {
    return false;
  }

  double A_i_z_lp1 = (A_i.transpose() * z).lpNorm<1>();
  double Γ_far = A_i_z_lp1 / c_i_z;
  double Γ_inf = (A_i_z_lp1 + s.transpose() * z) / z.lpNorm<1>();
  return Γ_far <= ε_far && Γ_inf <= ε_inf;
}

}  // namespace sleipnir
