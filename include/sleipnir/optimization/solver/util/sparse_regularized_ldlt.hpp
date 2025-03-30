// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/util/inertia.hpp"

namespace slp {

/// Solves sparse systems of linear equations using a regularized LDLT
/// factorization.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class SparseRegularizedLDLT {
 public:
  /// Type alias for dense vector.
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  /// Constructs a SparseRegularizedLDLT instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  SparseRegularizedLDLT(int num_decision_variables,
                        int num_equality_constraints)
      : m_num_decision_variables{num_decision_variables},
        m_num_equality_constraints{num_equality_constraints} {}

  /// Reports whether previous computation was successful.
  ///
  /// @return Whether previous computation was successful.
  Eigen::ComputationInfo info() const { return m_info; }

  /// Computes the regularized LDLT factorization of a matrix.
  ///
  /// The matrix's symbolic decomposition is reused in subsequent calls, so
  /// subsequent calls must be given a matrix with the same sparsity pattern.
  ///
  /// @param lhs Left-hand side of the system.
  /// @return The factorization.
  SparseRegularizedLDLT& compute(const SparseMatrix& lhs) {
    return compute(lhs, SparseMatrix{lhs.rows(), lhs.cols()});
  }

  /// Computes the regularized LDLT factorization of a matrix.
  ///
  /// The matrix's symbolic decomposition is reused in subsequent calls, so
  /// subsequent calls must be given a matrix with the same sparsity pattern.
  ///
  /// @param lhs Left-hand side of the system.
  /// @param reg Regularization matrix to add to lhs.
  /// @return The factorization.
  SparseRegularizedLDLT& compute(const SparseMatrix& lhs,
                                 const SparseMatrix& reg) {
    // Regularize lhs by adding a multiple of the identity matrix
    //
    // lhs = [H + AᵢᵀΣAᵢ + δI  Aₑᵀ]
    //       [      Aₑ         −γI]
    SparseMatrix lhs_reg = lhs + reg;

    if (!m_analyzed_pattern) {
      m_solver.analyzePattern(lhs_reg);
      m_analyzed_pattern = true;
    }

    m_solver.factorize(lhs_reg);
    m_info = m_solver.info();

    if (m_info == Eigen::Success &&
        Inertia{m_solver.vectorD()} == ideal_inertia) {
      return *this;
    } else {
      m_info = Eigen::NumericalIssue;
      return *this;
    }
  }

  /// Solves the system of equations using a regularized LDLT factorization.
  ///
  /// @param rhs Right-hand side of the system.
  /// @return The solution.
  template <typename Rhs>
  DenseVector solve(const Eigen::MatrixBase<Rhs>& rhs) const {
    return m_solver.solve(rhs);
  }

  /// Solves the system of equations using a regularized LDLT factorization.
  ///
  /// @param rhs Right-hand side of the system.
  /// @return The solution.
  template <typename Rhs>
  DenseVector solve(const Eigen::SparseMatrixBase<Rhs>& rhs) const {
    return m_solver.solve(rhs);
  }

  /// Returns the Hessian regularization factor.
  ///
  /// @return Hessian regularization factor.
  Scalar hessian_regularization() const { return Scalar(0); }

  /// Returns the constraint Jacobian regularization factor.
  ///
  /// @return Constraint Jacobian regularization factor.
  Scalar constraint_jacobian_regularization() const { return Scalar(0); }

 private:
  using Solver = Eigen::SimplicialLDLT<SparseMatrix>;

  Solver m_solver;
  bool m_analyzed_pattern = false;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  int m_num_decision_variables = 0;

  /// The number of equality constraints in the system.
  int m_num_equality_constraints = 0;

  /// The ideal system inertia.
  Inertia ideal_inertia{m_num_decision_variables, m_num_equality_constraints,
                        0};
};

}  // namespace slp
