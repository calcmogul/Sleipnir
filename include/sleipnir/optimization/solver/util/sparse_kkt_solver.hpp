// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/util/inertia.hpp"
#include "sleipnir/optimization/solver/util/kkt_ordering.hpp"
#include "sleipnir/optimization/solver/util/regularization_limits.hpp"

// See docs/algorithms.md#Works_cited for citation definitions.

namespace slp {

/// Solves a sparse KKT system.
///
/// Applies regularization so the solution is a descent direction.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class SparseKKTSolver {
 public:
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  /// Constructs a SparseKKTSolver instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  SparseKKTSolver(int num_decision_variables, int num_equality_constraints)
      : m_num_decision_variables{num_decision_variables},
        m_num_equality_constraints{num_equality_constraints} {}

  /// Constructs a SparseKKTSolver instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  /// @param γ_min The minimum constraint regularization.
  SparseKKTSolver(int num_decision_variables, int num_equality_constraints,
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
  /// The matrix's symbolic decomposition is reused in subsequent calls, so
  /// subsequent calls must be given a matrix with the same sparsity pattern.
  ///
  /// @param lhs Left-hand side of the system.
  /// @return The factorization.
  SparseKKTSolver& compute(const SparseMatrix& lhs) {
    // Regularization with zeros ensures the pattern analysis in the sparse
    // solver is reused by all factorizations
    if (!m_analyzed_pattern) {
      m_lhs = lhs + regularization(Scalar(0), Scalar(0));
      m_ordering.analyze(m_lhs, m_num_decision_variables);
      m_solver.analyzePattern(m_ordering.permute(m_lhs));
      m_analyzed_pattern = true;
    }

    // We'll give lhs the correct inertia by adding [δI, 0; 0, −γI] where δ and
    // γ regularize the Hessian and equality constraint Jacobian respectively.

    // The minimum nonzero Hessian regularization
    const Scalar δ_min = min_hessian_regularization<Scalar>();

    // The maximum constraint regularization. Larger values distort the equality
    // constraint Jacobian more than they help a rank-deficient one.
    const Scalar γ_max = max_constraint_regularization<Scalar>();

    // The initial nonzero constraint regularization
    const Scalar γ_initial = min_constraint_regularization<Scalar>();

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
      m_lhs = lhs + regularization(δ, γ);
      m_solver.factorize(m_ordering.permute(m_lhs));
      m_info = m_solver.info();

      if (m_info == Eigen::Success) {
        Inertia inertia = this->inertia();

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
            γ = γ_initial;
          } else {
            // If there's zero eigenvalues and γ > 0, try the next δ and a
            // larger γ to drive all eigenvalues away from zero
            next_δ();
            γ = std::min(γ * Scalar(10), γ_max);
          }
        } else if (inertia.negative > ideal_inertia.negative) {
          // If there's too many negative eigenvalues, try the next δ to add
          // more positive eigenvalues
          next_δ();
        } else if (inertia.positive > ideal_inertia.positive) {
          // If there's too many positive eigenvalues and the smallest pivot is
          // from the equality constraint Jacobian block, increase γ to add more
          // negative eigenvalues.
          //
          // Otherwise, the smallest pivot is from the Hessian block, or γ is
          // already large enough that increasing it further would distort the
          // equality constraint Jacobian more than it helps. Increase δ to
          // reduce element growth from Hessian pivots instead.
          if (γ < γ_max && !smallest_pivot_is_hessian()) {
            γ = γ == Scalar(0) ? γ_initial : std::min(γ * Scalar(10), γ_max);
          } else {
            next_δ();
          }
        }
      } else {
        // If the decomposition failed, try the next δ and a larger γ to drive
        // all eigenvalues away from zero
        next_δ();
        γ = γ == Scalar(0) ? γ_initial : std::min(γ * Scalar(10), γ_max);
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
    using std::pow;

    DenseVector x = m_ordering.inverse_permutation() *
                    m_solver.solve(m_ordering.permutation() * rhs);

    // Element growth in the factorization can make the solution inaccurate
    // (e.g., from small Hessian pivots in rows with no curvature), so apply
    // iterative refinement with the same factorization
    constexpr int max_refinement_steps = 3;

    // Stop once the relative residual is ε^(3/4), which is halfway (on a log
    // scale) between the solver's default tolerance of √ε and the
    // factorization's rounding error floor of a small multiple of ε
    const Scalar refinement_tolerance =
        pow(std::numeric_limits<Scalar>::epsilon(), Scalar(0.75));

    for (int i = 0; i < max_refinement_steps; ++i) {
      DenseVector r = rhs - m_lhs.template selfadjointView<Eigen::Lower>() * x;
      if (r.norm() <= refinement_tolerance * rhs.norm()) {
        break;
      }
      x += m_ordering.inverse_permutation() *
           m_solver.solve(m_ordering.permutation() * r);
    }

    return x;
  }

  /// Solves the system of equations.
  ///
  /// @param rhs Right-hand side of the system.
  /// @return The solution.
  template <typename Rhs>
  DenseVector solve(const Eigen::SparseMatrixBase<Rhs>& rhs) const {
    return solve(rhs.toDense());
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
  // KKTOrdering permutes the KKT matrix before factorization, so the solver
  // factors it as given. The upper triangle is the one it can factor without a
  // copy.
  using Solver = Eigen::SimplicialLDLT<
      SparseMatrix, Eigen::Upper,
      Eigen::NaturalOrdering<typename SparseMatrix::StorageIndex>>;

  Solver m_solver;
  KKTOrdering<Scalar> m_ordering;
  bool m_analyzed_pattern = false;

  /// The most recently factorized KKT matrix's lower triangle.
  SparseMatrix m_lhs;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  int m_num_decision_variables = 0;

  /// The number of equality constraints in the system.
  int m_num_equality_constraints = 0;

  /// The minimum constraint regularization.
  Scalar m_γ_min = min_constraint_regularization<Scalar>();

  /// The ideal system inertia.
  Inertia ideal_inertia{m_num_decision_variables, m_num_equality_constraints,
                        0};

  /// The last δ larger than the minimum that was needed for ideal inertia
  /// (or that hit the failure limit). Seeds next_δ().
  Scalar m_last_needed_δ{0};

  /// The Hessian regularization factor of the current factorization.
  Scalar m_δ{0};

  /// The equality constraint Jacobian regularization factor of the current
  /// factorization.
  Scalar m_γ{0};

  /// Returns the inertia of the most recently factorized KKT matrix.
  ///
  /// By Sylvester's law of inertia, the KKT matrix's inertia is the inertia of
  /// D from its LDLT factorization. Pivots within zero_pivot_threshold() of
  /// zero are counted as zero.
  ///
  /// @return The inertia of the most recently factorized KKT matrix.
  Inertia inertia() const {
    const Scalar threshold = zero_pivot_threshold<Scalar>();

    Inertia inertia;
    for (const auto& d : m_solver.vectorD()) {
      if (d > threshold) {
        ++inertia.positive;
      } else if (d < -threshold) {
        ++inertia.negative;
      } else {
        ++inertia.zero;
      }
    }

    return inertia;
  }

  /// Returns true if the smallest-magnitude pivot of the most recent
  /// factorization corresponds to a row of the Hessian block.
  ///
  /// @return True if the smallest pivot is from the Hessian block.
  bool smallest_pivot_is_hessian() const {
    Eigen::Index k;
    m_solver.vectorD().cwiseAbs().minCoeff(&k);

    // The factorization is of PAP⁻¹, so the original row of pivot k is P⁻¹k
    return m_ordering.inverse_permutation().indices()(k) <
           m_num_decision_variables;
  }

  /// Returns regularization matrix.
  ///
  ///   [δI    0]
  ///   [ 0  −γI]
  ///
  /// @param δ The Hessian regularization factor.
  /// @param γ The equality constraint Jacobian regularization factor.
  /// @return Regularization matrix.
  SparseMatrix regularization(Scalar δ, Scalar γ) const {
    DenseVector vec{m_num_decision_variables + m_num_equality_constraints};
    vec.segment(0, m_num_decision_variables).setConstant(δ);
    vec.segment(m_num_decision_variables, m_num_equality_constraints)
        .setConstant(-γ);

    return SparseMatrix{vec.asDiagonal()};
  }
};

}  // namespace slp
