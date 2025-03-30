// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "sleipnir/optimization/solver/util/inertia.hpp"

namespace slp {

/// Solves dense systems of linear equations using a regularized LDLT
/// factorization.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class DenseRegularizedLDLT {
 public:
  /// Type alias for dense matrix.
  using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  /// Type alias for dense vector.
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  /// Constructs a DenseRegularizedLDLT instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  DenseRegularizedLDLT(int num_decision_variables, int num_equality_constraints)
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
  DenseRegularizedLDLT& compute(const DenseMatrix& lhs) {
    // Regularize lhs by adding a multiple of the identity matrix
    //
    // lhs = [H + AᵢᵀΣAᵢ + δI  Aₑᵀ]
    //       [      Aₑ         −γI]

    m_info = m_solver.compute(lhs).info();

    if (m_info == Eigen::Success &&
        Inertia{m_solver.vectorD()} == ideal_inertia) {
      return *this;
    } else {
      m_info = Eigen::NumericalIssue;
      return *this;
    }
  }

  /// Computes the regularized LDLT factorization of a matrix.
  ///
  /// @param lhs Left-hand side of the system.
  /// @param reg Regularization matrix to add to lhs.
  /// @return The factorization.
  DenseRegularizedLDLT& compute(const DenseMatrix& lhs,
                                const DenseMatrix& reg) {
    // Regularize lhs by adding a multiple of the identity matrix
    //
    // lhs = [H + AᵢᵀΣAᵢ + δI  Aₑᵀ]
    //       [      Aₑ         −γI]
    DenseMatrix lhs_reg = lhs + reg;

    m_info = m_solver.compute(lhs_reg).info();

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
    return m_solver.solve(rhs.toDense());
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
  using Solver = Eigen::LDLT<DenseMatrix>;

  Solver m_solver;

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
