// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <cstddef>

#include <Eigen/Cholesky>
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
   */
  explicit RegularizedLDLT(size_t num_decision_variables)
      : m_num_decision_variables{num_decision_variables} {}

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

    m_info = m_is_sparse ? compute_sparse(lhs).info()
                         : m_dense_solver.compute(lhs).info();

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
    if (m_is_sparse) {
      return m_sparse_solver.solve(rhs);
    } else {
      return m_dense_solver.solve(rhs);
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
    if (m_is_sparse) {
      return m_sparse_solver.solve(rhs);
    } else {
      return m_dense_solver.solve(rhs.toDense());
    }
  }

  /**
   * Returns the Hessian regularization factor.
   *
   * @return Hessian regularization factor.
   */
  double hessian_regularization() const {
    if (m_is_sparse) {
      return m_sparse_solver.getMaxRegularization();
    } else {
      return 0.0;
    }
  }

 private:
  using SparseSolver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
  using DenseSolver = Eigen::LDLT<Eigen::MatrixXd>;

  SparseSolver m_sparse_solver;
  DenseSolver m_dense_solver;
  bool m_is_sparse = true;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  [[maybe_unused]]
  size_t m_num_decision_variables = 0;

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
