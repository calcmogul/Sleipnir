// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <cstddef>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "optimization/Eigen/SparseCholesky"

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
    m_info = compute_sparse(lhs).info();

    return *this;
  }

  /**
   * Solves the system of equations using a regularized LDLT factorization.
   *
   * @param rhs Right-hand side of the system.
   * @return The solution.
   */
  template <typename Rhs>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Rhs>& rhs) {
    return m_sparse_solver.solve(rhs);
  }

  /**
   * Solves the system of equations using a regularized LDLT factorization.
   *
   * @param rhs Right-hand side of the system.
   * @return The solution.
   */
  template <typename Rhs>
  Eigen::VectorXd solve(const Eigen::SparseMatrixBase<Rhs>& rhs) {
    return m_sparse_solver.solve(rhs);
  }

  /**
   * Returns the Hessian regularization factor.
   *
   * @return Hessian regularization factor.
   */
  double hessian_regularization() const {
    return m_sparse_solver.getMaxRegularization();
  }

 private:
  using SparseSolver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;

  SparseSolver m_sparse_solver;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  size_t m_num_decision_variables = 0;

  /// The number of equality constraints in the system.
  [[maybe_unused]]
  size_t m_num_equality_constraints = 0;

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

    m_sparse_solver.factorize(lhs, m_num_decision_variables);

    return m_sparse_solver;
  }
};

}  // namespace slp
