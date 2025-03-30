// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/util/inertia.hpp"

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

/// Solves systems of linear equations using a regularized LDLT factorization.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class RegularizedLDLT {
 public:
  /// Type alias for dense matrix.
  using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  /// Type alias for dense vector.
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  /// Constructs a RegularizedLDLT instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  RegularizedLDLT(int num_decision_variables, int num_equality_constraints)
      : m_num_decision_variables{num_decision_variables},
        m_num_equality_constraints{num_equality_constraints} {}

  /// Reports whether previous computation was successful.
  ///
  /// @return Whether previous computation was successful.
  Eigen::ComputationInfo info() const { return m_info; }

  /// Computes the regularized LDLT factorization of a matrix.
  ///
  /// @param lhs Left-hand side of the system.
  /// @return The factorization.
  RegularizedLDLT& compute(const SparseMatrix& lhs) {
    return compute(lhs, SparseMatrix{lhs.rows(), lhs.cols()});
  }

  /// Computes the regularized LDLT factorization of a matrix.
  ///
  /// @param lhs Left-hand side of the system.
  /// @param reg Regularization matrix to add to lhs.
  /// @return The factorization.
  RegularizedLDLT& compute(const SparseMatrix& lhs, const SparseMatrix& reg) {
    // Max density is 50% due to the caller only providing the lower triangle.
    // We consider less than 25% to be sparse.
    m_is_sparse = lhs.nonZeros() < 0.25 * lhs.size();

    m_info = m_is_sparse ? compute_sparse(lhs).info()
                         : m_dense_solver.compute(lhs).info();

    Inertia inertia;

    if (m_info == Eigen::Success) {
      inertia = m_is_sparse ? Inertia{m_sparse_solver.vectorD()}
                            : Inertia{m_dense_solver.vectorD()};

      // If the inertia is ideal, don't regularize the system
      if (inertia == ideal_inertia) {
        m_prev_δ = Scalar(0);
        return *this;
      }
    }

    // Regularize lhs by adding a multiple of the identity matrix
    //
    // lhs = [H + AᵢᵀΣAᵢ + δI  Aₑᵀ]
    //       [      Aₑ         −γI]
    if (m_is_sparse) {
      m_info = compute_sparse(lhs + reg).info();
      if (m_info == Eigen::Success) {
        inertia = Inertia{m_sparse_solver.vectorD()};
      }
    } else {
      m_info = m_dense_solver.compute(lhs + reg).info();
      if (m_info == Eigen::Success) {
        inertia = Inertia{m_dense_solver.vectorD()};
      }
    }

    if (m_info == Eigen::Success && inertia == ideal_inertia) {
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
  DenseVector solve(const Eigen::MatrixBase<Rhs>& rhs) {
    if (m_is_sparse) {
      return m_sparse_solver.solve(rhs);
    } else {
      return m_dense_solver.solve(rhs);
    }
  }

  /// Solves the system of equations using a regularized LDLT factorization.
  ///
  /// @param rhs Right-hand side of the system.
  /// @return The solution.
  template <typename Rhs>
  DenseVector solve(const Eigen::SparseMatrixBase<Rhs>& rhs) {
    if (m_is_sparse) {
      return m_sparse_solver.solve(rhs);
    } else {
      return m_dense_solver.solve(rhs.toDense());
    }
  }

  /// Returns the Hessian regularization factor.
  ///
  /// @return Hessian regularization factor.
  Scalar hessian_regularization() const { return m_prev_δ; }

 private:
  using SparseSolver = Eigen::SimplicialLDLT<SparseMatrix>;
  using DenseSolver = Eigen::LDLT<DenseMatrix>;

  SparseSolver m_sparse_solver;
  DenseSolver m_dense_solver;
  bool m_is_sparse = true;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  int m_num_decision_variables = 0;

  /// The number of equality constraints in the system.
  int m_num_equality_constraints = 0;

  /// The ideal system inertia.
  Inertia ideal_inertia{m_num_decision_variables, m_num_equality_constraints,
                        0};

  /// The value of δ from the previous run of compute().
  Scalar m_prev_δ{0};

  // Number of non-zeros in LHS.
  int m_non_zeros = -1;

  /// Computes factorization of a sparse matrix.
  ///
  /// @param lhs Matrix to factorize.
  /// @return The factorization.
  SparseSolver& compute_sparse(const SparseMatrix& lhs) {
    // Reanalize lhs's sparsity pattern if it changed
    int non_zeros = lhs.nonZeros();
    if (m_non_zeros != non_zeros) {
      m_sparse_solver.analyzePattern(lhs);
      m_non_zeros = non_zeros;
    }

    m_sparse_solver.factorize(lhs);

    return m_sparse_solver;
  }
};

}  // namespace slp
