// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "sleipnir/optimization/solver/util/inertia.hpp"

// See docs/algorithms.md#Works_cited for citation definitions.

namespace slp {

/// Solves a dense KKT system.
///
/// Applies regularization so the solution is a descent direction.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class DenseKKTSolver {
 public:
  /// Type alias for dense matrix.
  using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  /// Type alias for dense vector.
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  /// Constructs a DenseKKTSolver instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  DenseKKTSolver(int num_decision_variables, int num_equality_constraints)
      : m_num_decision_variables{num_decision_variables},
        m_num_equality_constraints{num_equality_constraints} {}

  /// Constructs a DenseKKTSolver instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  /// @param γ_min The minimum constraint regularization.
  DenseKKTSolver(int num_decision_variables, int num_equality_constraints,
                 Scalar γ_min)
      : m_num_decision_variables{num_decision_variables},
        m_num_equality_constraints{num_equality_constraints},
        m_γ_min{γ_min} {}

  /// Reports whether previous computation was successful.
  ///
  /// @return Whether previous computation was successful.
  Eigen::ComputationInfo info() const { return m_info; }

  /// Computes the factorization of the KKT matrix.
  ///
  /// @param lhs Left-hand side of the system.
  /// @return The factorization.
  DenseKKTSolver& compute(const DenseMatrix& lhs) {
    // We'll give lhs the correct inertia by adding [δI, 0; 0, −γI] where δ and
    // γ regularize the Hessian and equality constraint Jacobian respectively.
    //
    // First, attempt the minimum δ. It's too small to distort the step, but it
    // removes the exact zero pivots that structurally zero diagonal entries
    // (e.g., decision variables with no curvature) cause in LDLT. If that
    // fails, δ jumps to a value based on the last δ that was needed, then
    // increases from there.
    //
    // See algorithm IC in section 3.1 of [2].
    Scalar δ = δ_min;

    // Start γ at the minimum to minimize equality constraint Jacobian
    // distortion.
    Scalar γ = m_γ_min;

    // Picks the next δ to try
    auto next_δ = [&] {
      // If δ is at the minimum, jump to half the last needed δ (or 1e-4 if
      // none) so δ trends downward
      if (δ == δ_min) {
        δ = m_last_needed_δ == Scalar(0)
                ? Scalar(1e-4)
                : std::max(m_last_needed_δ / Scalar(2), Scalar(10) * δ_min);
      } else {
        // Otherwise, multiply δ by 10
        δ *= Scalar(10);
      }
    };

    while (true) {
      m_info = m_solver.compute(lhs + regularization(δ, γ)).info();

      if (m_info == Eigen::Success) {
        Inertia inertia{m_solver.vectorD()};

        if (inertia == ideal_inertia) {
          // If the inertia is ideal, report success. Only remember δ if it
          // was needed.
          if (δ > δ_min) {
            m_last_needed_δ = δ;
          }
          m_δ = δ;
          m_γ = γ;
          return *this;
        } else if (inertia.zero > 0) {
          if (γ == Scalar(0)) {
            // If there's zero eigenvalues and γ = 0, increase γ to potentially
            // compensate for a rank-deficient equality constraint Jacobian
            γ = Scalar(1e-10);
          } else {
            // If there's zero eigenvalues and γ > 0, try the next δ and a
            // larger γ to drive all eigenvalues away from zero
            next_δ();
            γ *= Scalar(10);
          }
        } else if (inertia.negative > ideal_inertia.negative) {
          // If there's too many negative eigenvalues, try the next δ to add
          // more positive eigenvalues
          next_δ();
        } else if (inertia.positive > ideal_inertia.positive) {
          // If there's too many positive eigenvalues, increase γ to add more
          // negative eigenvalues
          γ = γ == Scalar(0) ? Scalar(1e-10) : γ * Scalar(10);
        }
      } else {
        // If the decomposition failed, try the next δ and a larger γ to drive
        // all eigenvalues away from zero
        next_δ();
        γ = γ == Scalar(0) ? Scalar(1e-10) : γ * Scalar(10);
      }

      // If the lhs perturbation is too high, report failure. This can be caused
      // by ill-conditioning.
      if (δ > Scalar(1e20) || γ > Scalar(1e20)) {
        m_info = Eigen::NumericalIssue;
        m_last_needed_δ = δ;
        m_δ = δ;
        m_γ = γ;
        return *this;
      }
    }
  }

  /// Solves the system of equations.
  ///
  /// @param rhs Right-hand side of the system.
  /// @return The solution.
  template <typename Rhs>
  DenseVector solve(const Eigen::MatrixBase<Rhs>& rhs) const {
    return m_solver.solve(rhs);
  }

  /// Solves the system of equations.
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
  Scalar hessian_regularization() const { return m_δ; }

  /// Returns the constraint Jacobian regularization factor.
  ///
  /// @return Constraint Jacobian regularization factor.
  Scalar constraint_jacobian_regularization() const { return m_γ; }

 private:
  using Solver = Eigen::LDLT<DenseMatrix>;

  Solver m_solver;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  int m_num_decision_variables = 0;

  /// The number of equality constraints in the system.
  int m_num_equality_constraints = 0;

  /// The minimum constraint regularization.
  Scalar m_γ_min{1e-10};

  /// The ideal system inertia.
  Inertia ideal_inertia{m_num_decision_variables, m_num_equality_constraints,
                        0};

  /// The minimum Hessian regularization factor.
  static constexpr Scalar δ_min{1e-8};

  /// The last δ larger than the minimum that was needed for ideal inertia
  /// (or that hit the failure limit). Seeds next_δ().
  Scalar m_last_needed_δ{0};

  /// The Hessian regularization factor of the current factorization.
  Scalar m_δ{0};

  /// The equality constraint Jacobian regularization factor of the current
  /// factorization.
  Scalar m_γ{0};

  /// Returns regularization matrix.
  ///
  ///   [δI    0]
  ///   [ 0  −γI]
  ///
  /// @param δ The Hessian regularization factor.
  /// @param γ The equality constraint Jacobian regularization factor.
  /// @return Regularization matrix.
  DenseMatrix regularization(Scalar δ, Scalar γ) const {
    DenseVector vec{m_num_decision_variables + m_num_equality_constraints};
    vec.segment(0, m_num_decision_variables).setConstant(δ);
    vec.segment(m_num_decision_variables, m_num_equality_constraints)
        .setConstant(-γ);

    return vec.asDiagonal().toDenseMatrix();
  }
};

}  // namespace slp
