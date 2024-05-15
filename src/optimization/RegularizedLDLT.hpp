// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <cstddef>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "optimization/Inertia.hpp"

// See docs/algorithms.md#Works_cited for citation definitions

namespace sleipnir {

/**
 * Solves systems of linear equations using a regularized LDLT factorization.
 */
class RegularizedLDLT {
 public:
  /**
   * Constructs a RegularizedLDLT instance.
   */
  RegularizedLDLT() = default;

  /**
   * Reports whether previous computation was successful.
   *
   * @return Whether previous computation was successful.
   */
  Eigen::ComputationInfo Info() const { return m_info; }

  /**
   * Computes the regularized LDLT factorization of a matrix.
   *
   * @param lhs Left-hand side of the system.
   * @param numEqualityConstraints The number of equality constraints in the
   *   system.
   * @param μ The barrier parameter for the current interior-point iteration.
   * @return The factorization.
   */
  RegularizedLDLT& Compute(const Eigen::SparseMatrix<double>& lhs,
                           size_t numEqualityConstraints, double μ) {
    // The regularization procedure is based on algorithm B.1 of [1]
    m_numDecisionVariables = lhs.rows() - numEqualityConstraints;
    m_numEqualityConstraints = numEqualityConstraints;

    const Inertia idealInertia{m_numDecisionVariables, m_numEqualityConstraints,
                               0};
    Inertia inertia;

    double δ = 0.0;
    double γ = 0.0;

    // Max density is 50% due to the caller only providing the lower triangle.
    // We consider less than 25% to be sparse.
    m_isSparse = lhs.nonZeros() < 0.25 * lhs.size();

    if (m_isSparse) {
      m_info = ComputeSparse(lhs).info();
    } else {
      m_info = m_denseSolver.compute(lhs).info();
    }

    if (m_info == Eigen::Success) {
      if (m_isSparse) {
        inertia = Inertia{m_sparseSolver};
      } else {
        inertia = Inertia{m_denseSolver};
      }

      // If the inertia is ideal, don't regularize the system
      if (inertia == idealInertia) {
        return *this;
      }
    }

    // If the decomposition succeeded and the inertia has some zero eigenvalues,
    // or the decomposition failed, regularize the equality constraints
    if ((m_info == Eigen::Success && inertia.zero > 0) ||
        m_info != Eigen::Success) {
      γ = 1e-8 * std::pow(μ, 0.25);
    }

    // Also regularize the Hessian. If the Hessian wasn't regularized in a
    // previous run of Compute(), start at a small value of δ. Otherwise,
    // attempt a δ half as big as the previous run so δ can trend downwards over
    // time.
    if (m_δOld == 0.0) {
      δ = 1e-4;
    } else {
      δ = m_δOld / 2.0;
    }

    while (true) {
      // Regularize lhs by adding a multiple of the identity matrix
      //
      // lhs = [H + AᵢᵀΣAᵢ + δI   Aₑᵀ]
      //       [       Aₑ        −γI ]
      if (m_isSparse) {
        m_info = ComputeSparse(lhs + Regularization(δ, γ)).info();
        if (m_info == Eigen::Success) {
          inertia = Inertia{m_sparseSolver};
        }
      } else {
        m_info = m_denseSolver.compute(lhs + Regularization(δ, γ)).info();
        if (m_info == Eigen::Success) {
          inertia = Inertia{m_denseSolver};
        }
      }

      // If the inertia is ideal, store that value of δ and return.
      // Otherwise, increase δ by an order of magnitude and try again.
      if (m_info == Eigen::Success && inertia == idealInertia) {
        m_δOld = δ;
        return *this;
      } else {
        δ *= 10.0;

        // If the Hessian perturbation is too high, report failure. This can be
        // caused by ill-conditioning.
        if (δ > 1e20) {
          m_info = Eigen::NumericalIssue;
          return *this;
        }
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
  Eigen::VectorXd Solve(const Eigen::MatrixBase<Rhs>& rhs) {
    if (m_isSparse) {
      return m_sparseSolver.solve(rhs);
    } else {
      return m_denseSolver.solve(rhs);
    }
  }

  /**
   * Solves the system of equations using a regularized LDLT factorization.
   *
   * @param rhs Right-hand side of the system.
   * @return The solution.
   */
  template <typename Rhs>
  Eigen::VectorXd Solve(const Eigen::SparseMatrixBase<Rhs>& rhs) {
    if (m_isSparse) {
      return m_sparseSolver.solve(rhs);
    } else {
      return m_denseSolver.solve(rhs.toDense());
    }
  }

  /**
   * Returns the Hessian regularization factor.
   *
   * @return Hessian regularization factor.
   */
  double HessianRegularization() const { return m_δOld; }

 private:
  using SparseSolver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;
  using DenseSolver = Eigen::LDLT<Eigen::MatrixXd>;

  SparseSolver m_sparseSolver;
  DenseSolver m_denseSolver;
  bool m_isSparse = true;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  size_t m_numDecisionVariables = 0;

  /// The number of equality constraints in the system.
  size_t m_numEqualityConstraints = 0;

  /// The value of δ from the previous run of Compute().
  double m_δOld = 0.0;

  // Number of non-zeros in LHS.
  int m_nonZeros = -1;

  /**
   * Computes factorization of a sparse matrix.
   *
   * @param lhs Matrix to factorize.
   * @return The factorization.
   */
  SparseSolver& ComputeSparse(const Eigen::SparseMatrix<double>& lhs) {
    // Reanalize lhs's sparsity pattern if it changed
    int nonZeros = lhs.nonZeros();
    if (m_nonZeros != nonZeros) {
      m_sparseSolver.analyzePattern(lhs);
      m_nonZeros = nonZeros;
    }

    m_sparseSolver.factorize(lhs);

    return m_sparseSolver;
  }

  /**
   * Returns regularization matrix.
   *
   * @param δ The Hessian regularization factor.
   * @param γ The equality constraint Jacobian regularization factor.
   * @return Regularization matrix.
   */
  Eigen::SparseMatrix<double> Regularization(double δ, double γ) {
    Eigen::VectorXd vec{m_numDecisionVariables + m_numEqualityConstraints};
    size_t row = 0;
    while (row < m_numDecisionVariables) {
      vec[row] = δ;
      ++row;
    }
    while (row < m_numDecisionVariables + m_numEqualityConstraints) {
      vec[row] = -γ;
      ++row;
    }

    return Eigen::SparseMatrix<double>{vec.asDiagonal()};
  }
};

}  // namespace sleipnir
