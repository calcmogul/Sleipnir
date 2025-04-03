// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "optimization/inertia.hpp"

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

/**
 * Solves systems of linear equations using a regularized LDLT factorization.
 */
class RegularizedLDLT {
 public:
  /**
   * Constructs a RegularizedLDLT instance.
   *
   * @param num_decision_variables The number of decision variables in the
   *   system.
   * @param num_equality_constraints The number of equality constraints in the
   *   system.
   */
  RegularizedLDLT(size_t num_decision_variables,
                  size_t num_equality_constraints)
      : m_num_decision_variables{num_decision_variables},
        m_num_equality_constraints{num_equality_constraints} {}

  /**
   * Reports whether previous computation was successful.
   *
   * @return Whether previous computation was successful.
   */
  Eigen::ComputationInfo info() const { return m_info; }

  /**
   * Computes the regularized LDLT factorization of a matrix.
   *
   * @param lhs Left-hand side of the system.
   * @return The factorization.
   */
  RegularizedLDLT& compute(const Eigen::SparseMatrix<double>& lhs) {
    // The regularization procedure is based on algorithm B.1 of [1]

    // Max density is 50% due to the caller only providing the lower triangle.
    // We consider less than 25% to be sparse.
    m_is_sparse = lhs.nonZeros() < 0.25 * lhs.size();

    m_D_inv = scaling(lhs);

    Eigen::SparseMatrix<double> scaled_lhs = m_D_inv * lhs * m_D_inv;

    m_info = m_is_sparse ? compute_sparse(scaled_lhs).info()
                         : m_dense_solver.compute(scaled_lhs).info();

    Inertia inertia;

    if (m_info == Eigen::Success) {
      inertia = m_is_sparse ? Inertia{m_sparse_solver.vectorD()}
                            : Inertia{m_dense_solver.vectorD()};

      // If the inertia is ideal, don't regularize the system
      if (inertia == ideal_inertia) {
        m_prev_δ = 0.0;
        return *this;
      }
    }

    // Also regularize the Hessian. If the Hessian wasn't regularized in a
    // previous run of compute(), start at small values of δ and γ. Otherwise,
    // attempt a δ and γ half as big as the previous run so δ and γ can trend
    // downwards over time.
    double δ = m_prev_δ == 0.0 ? 1e-10 : m_prev_δ / 2.0;
    double γ = 1e-10;

    while (true) {
      // Regularize lhs by adding a multiple of the identity matrix
      //
      // lhs = [H + AᵢᵀΣAᵢ + δI  Aₑᵀ]
      //       [      Aₑ         −γI]
      Eigen::SparseMatrix<double> reg_lhs = scaled_lhs + regularization(δ, γ);

      if (m_is_sparse) {
        m_info = compute_sparse(reg_lhs).info();
        if (m_info == Eigen::Success) {
          inertia = Inertia{m_sparse_solver.vectorD()};
        }
      } else {
        m_info = m_dense_solver.compute(reg_lhs).info();
        if (m_info == Eigen::Success) {
          inertia = Inertia{m_dense_solver.vectorD()};
        }
      }

      if (m_info == Eigen::Success) {
        if (inertia == ideal_inertia) {
          // If the inertia is ideal, store δ and return
          m_prev_δ = δ;
          return *this;
        } else if (inertia.zero > 0) {
          // If there's zero eigenvalues, increase δ and γ by an order of
          // magnitude and try again
          δ *= 10.0;
          γ *= 10.0;
        } else if (inertia.negative > ideal_inertia.negative) {
          // If there's too many negative eigenvalues, increase δ by an order of
          // magnitude and try again
          δ *= 10.0;
        } else if (inertia.positive > ideal_inertia.positive) {
          // If there's too many positive eigenvalues, increase γ by an order of
          // magnitude and try again
          γ *= 10.0;
        }
      } else {
        // If the decomposition failed, increase δ and γ by an order of
        // magnitude and try again
        δ *= 10.0;
        γ *= 10.0;
      }

      // If the Hessian perturbation is too high, report failure. This can be
      // caused by ill-conditioning.
      if (δ > 1e20 || γ > 1e20) {
        m_info = Eigen::NumericalIssue;
        return *this;
      }
    }
  }

  /**
   * Solves the system of equations using a regularized LDLT factorization.
   *
   * @param rhs Right-hand side of the system.
   * @return The solution.
   */
  template <typename Rhs>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Rhs>& rhs) {
    // x = D⁻¹ solve(D⁻¹AD⁻¹, D⁻¹b)
    if (m_is_sparse) {
      return m_D_inv * m_sparse_solver.solve(m_D_inv * rhs);
    } else {
      return m_D_inv * m_dense_solver.solve((m_D_inv * rhs).eval());
    }
  }

  /**
   * Solves the system of equations using a regularized LDLT factorization.
   *
   * @param rhs Right-hand side of the system.
   * @return The solution.
   */
  template <typename Rhs>
  Eigen::VectorXd solve(const Eigen::SparseMatrixBase<Rhs>& rhs) {
    // x = D⁻¹ solve(D⁻¹AD⁻¹, D⁻¹b)
    if (m_is_sparse) {
      return m_D_inv * m_sparse_solver.solve(m_D_inv * rhs);
    } else {
      return m_D_inv * m_dense_solver.solve((m_D_inv * rhs).toDense());
    }
  }

  /**
   * Returns the Hessian regularization factor.
   *
   * @return Hessian regularization factor.
   */
  double hessian_regularization() const { return m_prev_δ; }

 private:
  using SparseSolver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
  using DenseSolver = Eigen::LDLT<Eigen::MatrixXd>;

  SparseSolver m_sparse_solver;
  DenseSolver m_dense_solver;
  bool m_is_sparse = true;

  /// The scaling matrix D⁻¹ in D⁻¹AD⁻¹x = D⁻¹b. This improves numerical
  /// conditioning.
  Eigen::SparseMatrix<double> m_D_inv;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  size_t m_num_decision_variables = 0;

  /// The number of equality constraints in the system.
  size_t m_num_equality_constraints = 0;

  /// The ideal system inertia.
  Inertia ideal_inertia{m_num_decision_variables, m_num_equality_constraints,
                        0};

  /// The value of δ from the previous run of compute().
  double m_prev_δ = 0.0;

  // Number of non-zeros in LHS.
  int m_non_zeros = -1;

  /**
   * Computes factorization of a sparse matrix.
   *
   * @param lhs Matrix to factorize.
   * @return The factorization.
   */
  SparseSolver& compute_sparse(const Eigen::SparseMatrix<double>& lhs) {
    // Reanalize lhs's sparsity pattern if it changed
    int non_zeros = lhs.nonZeros();
    if (m_non_zeros != non_zeros) {
      m_sparse_solver.analyzePattern(lhs);
      m_non_zeros = non_zeros;
    }

    m_sparse_solver.factorize(lhs);

    return m_sparse_solver;
  }

  /**
   * Returns scaling matrix D⁻¹.
   *
   * @param lhs Matrix to scale.
   */
  Eigen::SparseMatrix<double> scaling(const Eigen::SparseMatrix<double>& lhs) {
    //   Ax = b
    //   D⁻¹Ax = D⁻¹b
    //   D⁻¹AD⁻¹Dx = D⁻¹b
    //   (D⁻¹AD⁻¹)Dx = D⁻¹b
    //   Dx = solve(D⁻¹AD⁻¹, D⁻¹b)
    //   x = D⁻¹ solve(D⁻¹AD⁻¹, D⁻¹b)

    // Find ∞-norm of each row
    Eigen::VectorXd d = Eigen::VectorXd::Constant(lhs.rows(), 0.0);
    for (int col = 0; col < lhs.cols(); ++col) {
      for (Eigen::SparseMatrix<double>::InnerIterator it{lhs, col}; it; ++it) {
        // The extra column assignment is needed for full coverage of the
        // symmetric matrix when only given its lower triangle
        d[it.row()] = std::max(d[it.row()], std::abs(it.value()));
        d[it.col()] = std::max(d[it.col()], std::abs(it.value()));
      }
    }

    // If a row had zero norm, set norm to 1
    for (int row = 0; row < lhs.rows(); ++row) {
      if (d[row] == 0.0) {
        d[row] = 1.0;
      }
    }

    return Eigen::SparseMatrix<double>{d.cwiseInverse().asDiagonal()};
  }

  /**
   * Returns regularization matrix.
   *
   * @param δ The Hessian regularization factor.
   * @param γ The equality constraint Jacobian regularization factor.
   * @return Regularization matrix.
   */
  Eigen::SparseMatrix<double> regularization(double δ, double γ) {
    Eigen::VectorXd vec{m_num_decision_variables + m_num_equality_constraints};
    vec.segment(0, m_num_decision_variables).setConstant(δ);
    vec.segment(m_num_decision_variables, m_num_equality_constraints)
        .setConstant(-γ);

    return Eigen::SparseMatrix<double>{vec.asDiagonal()};
  }
};

}  // namespace slp
