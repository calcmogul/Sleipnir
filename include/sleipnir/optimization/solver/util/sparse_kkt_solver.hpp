// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/util/regularization_limits.hpp"

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
    m_lhs = lhs + regularization(Scalar(0), Scalar(0));

    if (!m_analyzed_pattern) {
      m_scaled_lhs = m_lhs;
      m_solver.analyzePattern(m_scaled_lhs);
      m_analyzed_pattern = true;
    }

    // Threshold pivoting with 2x2 blocks is stable for indefinite matrices in
    // practice, so the unregularized system is accepted whenever its inertia is
    // ideal
    factorize();
    if (m_info == Eigen::Success && has_ideal_inertia()) {
      m_prev_δ = Scalar(0);
      m_prev_γ = Scalar(0);
      return *this;
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

    // If the previous δ was zero, attempt a small value. Otherwise, attempt a
    // smaller value than the previous δ so δ trends downward. δ has a floor
    // because when regularization is needed, δ is the only curvature in some
    // directions, so the KKT matrix's smallest eigenvalues shrink with δ. Too
    // small a δ makes the matrix ill-conditioned, which amplifies the
    // factorization's rounding error in the solution, and eventually puts those
    // eigenvalues near zero, where the inertia check counts them as zero.
    Scalar δ = m_prev_δ == Scalar(0) ? Scalar(1e-4)
                                     : std::max(m_prev_δ / Scalar(2), δ_min);

    // Start γ at zero to avoid distorting the equality constraint Jacobian. It
    // only needs to be nonzero if the Jacobian turns out to be rank-deficient.
    Scalar γ(0);

    while (true) {
      m_lhs = lhs + regularization(δ, γ);
      factorize();

      if (m_info == Eigen::Success) {
        if (has_near_zero_eigenvalue()) {
          if (γ == Scalar(0)) {
            // If there's zero eigenvalues and γ = 0, increase γ to potentially
            // compensate for a rank-deficient equality constraint Jacobian
            γ = γ_initial;
          } else {
            // If there's zero eigenvalues and γ > 0, increase δ and γ to drive
            // all eigenvalues away from zero
            δ *= Scalar(10);
            γ = std::min(γ * Scalar(10), γ_max);
          }
        } else if (m_solver.numNegative() > m_num_equality_constraints) {
          // If there's too many negative eigenvalues, increase δ to add more
          // positive eigenvalues
          δ *= Scalar(10);
        } else if (m_solver.numPositive() > m_num_decision_variables) {
          // The KKT matrix has at least as many negative eigenvalues as
          // equality constraints if γ > 0 (from the −γI block) or if the
          // equality constraint Jacobian has full row rank. Too many positive
          // eigenvalues can then only come from rounding error, so increase δ
          // and γ to drive all eigenvalues away from zero.
          δ *= Scalar(10);
          γ = γ == Scalar(0) ? γ_initial : std::min(γ * Scalar(10), γ_max);
        } else {
          // If the inertia is ideal, report success
          m_prev_δ = δ;
          m_prev_γ = γ;
          return *this;
        }
      } else {
        // If the decomposition failed, increase δ and γ to drive all
        // eigenvalues away from zero
        δ *= Scalar(10);
        γ = γ == Scalar(0) ? γ_initial : std::min(γ * Scalar(10), γ_max);
      }

      // If the lhs perturbation is too high, report failure. This can be caused
      // by ill-conditioning.
      if (δ > Scalar(1e20) || γ > Scalar(1e20)) {
        m_info = Eigen::NumericalIssue;
        m_prev_δ = δ;
        m_prev_γ = γ;
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

    DenseVector x = solve_factorized(rhs);

    // The factorization is stable, but the matrix's conditioning amplifies its
    // rounding error in the solution, so apply iterative refinement with the
    // same factorization
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
      x += solve_factorized(r);
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
  Scalar hessian_regularization() const { return m_prev_δ; }

  /// Returns the constraint Jacobian regularization factor.
  ///
  /// @return Constraint Jacobian regularization factor.
  Scalar constraint_jacobian_regularization() const { return m_prev_γ; }

 private:
  using Solver = Eigen::MultifrontalLDLT<SparseMatrix, Eigen::Lower>;

  Solver m_solver;
  bool m_analyzed_pattern = false;

  /// The most recently factorized KKT matrix's lower triangle.
  SparseMatrix m_lhs;

  /// The lower triangle of S lhs S, where S is the equilibration.
  SparseMatrix m_scaled_lhs;

  /// The diagonal of the equilibration S.
  DenseVector m_scale;

  /// Each row's largest entry during equilibration, kept to avoid reallocating.
  DenseVector m_row_max;

  Eigen::ComputationInfo m_info = Eigen::Success;

  /// The number of decision variables in the system.
  int m_num_decision_variables = 0;

  /// The number of equality constraints in the system.
  int m_num_equality_constraints = 0;

  /// The value of δ from the previous run of compute().
  Scalar m_prev_δ{0};

  /// The value of γ from the previous run of compute().
  Scalar m_prev_γ{0};

  /// Factorizes the KKT matrix in m_lhs.
  ///
  /// The pivot tests compare a column's entries with each other, so a large
  /// entry in a row that isn't fully summed yet (e.g., a timestep variable
  /// shared by every dynamics constraint) can delay pivots all the way up the
  /// elimination tree. To avoid that, this factors S lhs S, where S is a Ruiz
  /// equilibration.
  void factorize() {
    using std::abs;

    // Each sweep moves every row's largest entry closer to 1. A few are enough
    // because the pivot tests only need rough scaling.
    constexpr int num_scaling_sweeps = 3;

    const int size = m_lhs.rows();
    const auto* outer = m_lhs.outerIndexPtr();
    const auto* inner = m_lhs.innerIndexPtr();

    m_scale.setOnes(size);
    m_row_max.resize(size);
    for (int sweep = 0; sweep < num_scaling_sweeps; ++sweep) {
      m_row_max.setZero();
      for (int col = 0; col < size; ++col) {
        const Scalar col_scale = m_scale[col];
        Scalar col_max{0};
        for (auto k = outer[col]; k < outer[col + 1]; ++k) {
          const int row = inner[k];
          const Scalar value =
              abs(m_lhs.valuePtr()[k]) * m_scale[row] * col_scale;
          m_row_max[row] = std::max(m_row_max[row], value);
          col_max = std::max(col_max, value);
        }
        m_row_max[col] = std::max(m_row_max[col], col_max);
      }

      // Empty rows keep their scale
      m_row_max = (m_row_max.array() > Scalar(0)).select(m_row_max, Scalar(1));
      m_scale.array() /= m_row_max.array().sqrt();
    }

    // m_scaled_lhs has the same sparsity pattern as m_lhs
    Scalar* scaled_values = m_scaled_lhs.valuePtr();
    for (int col = 0; col < size; ++col) {
      const Scalar col_scale = m_scale[col];
      for (auto k = outer[col]; k < outer[col + 1]; ++k) {
        scaled_values[k] =
            m_lhs.valuePtr()[k] * (m_scale[inner[k]] * col_scale);
      }
    }

    m_solver.factorize(m_scaled_lhs);
    m_info = m_solver.info();
  }

  /// Solves the system of equations with the most recent factorization.
  ///
  /// @param rhs Right-hand side of the system.
  /// @return The solution.
  DenseVector solve_factorized(const DenseVector& rhs) const {
    DenseVector scaled_rhs = rhs.cwiseProduct(m_scale);
    DenseVector x = m_solver.solve(scaled_rhs);
    x.array() *= m_scale.array();
    return x;
  }

  /// Returns whether the most recently factorized KKT matrix has the ideal
  /// inertia.
  ///
  /// The factorization's sign counts are the KKT matrix's inertia by
  /// Sylvester's law of inertia, but they only count exact zero pivots.
  /// Rounding can leave a tiny pivot in place of a zero, so near-zero
  /// eigenvalues are checked first.
  ///
  /// @return Whether the most recently factorized KKT matrix has the ideal
  ///     inertia.
  bool has_ideal_inertia() const {
    return !has_near_zero_eigenvalue() &&
           m_solver.numPositive() == m_num_decision_variables &&
           m_solver.numNegative() == m_num_equality_constraints;
  }

  /// Returns whether the most recently factorized KKT matrix has an eigenvalue
  /// that should be counted as zero.
  ///
  /// D from the LDLT factorization has 1x1 and 2x2 diagonal blocks. Eigenvalues
  /// within zero_pivot_threshold() of zero are counted as zero. If there aren't
  /// any, the factorization's sign counts are the inertia.
  ///
  /// The factorization is of S lhs S, whose D has the same inertia, but the
  /// threshold applies to the KKT matrix itself. Equilibration can scale a
  /// nearly singular matrix's small pivots past it, so this undoes the scaling
  /// first.
  ///
  /// @return Whether the most recently factorized KKT matrix has an eigenvalue
  ///     that should be counted as zero.
  bool has_near_zero_eigenvalue() const {
    using std::abs;
    using std::hypot;

    const Scalar threshold = zero_pivot_threshold<Scalar>();

    const auto& D = m_solver.vectorD();
    const auto& D_sub = m_solver.subDiagonal();
    const auto& rows = m_solver.permutationPinv().indices();

    // This treats NaN as zero
    auto is_near_zero = [&](Scalar λ) { return !(abs(λ) > threshold); };

    for (int k = 0; k < D.size(); ++k) {
      const Scalar s = m_scale[rows[k]];
      if (D_sub[k] != Scalar(0)) {
        const Scalar s_next = m_scale[rows[k + 1]];

        // The eigenvalues of the 2x2 block [a  b; b  c] are the roots of its
        // characteristic polynomial.
        //
        //   det([a − λ  b; b  c − λ]) = (a − λ)(c − λ) − b²
        //                             = λ² − (a + c)λ + (ac − b²)
        //
        //   λ = (a + c)/2 ± r where r = √(((a − c)/2)² + b²)
        //
        // Let λ₁ be the root farther from zero, so only the other root λ₂ needs
        // checking. λ₁'s two terms have the same sign, so it has no
        // cancellation. The two terms of λ₂ have opposite signs, so λ₂ loses
        // accuracy if |a| >> |c|. Instead, compute λ₂ from λ₁ and the product
        // of the roots.
        //
        //   λ₁λ₂ = ((a + c)/2 + r)((a + c)/2 − r)
        //   λ₂ = (a/λ₁)c − (b/λ₁)b
        //
        // Dividing first prevents ac and b² from overflowing. |λ₁| is the
        // block's spectral radius (its largest eigenvalue magnitude), which for
        // a symmetric matrix is the most it can stretch a unit vector. The
        // block maps [1, 0]ᵀ to [a, b]ᵀ, so |a| ≤ |λ₁| and |b| ≤ |λ₁|.
        const Scalar a = D[k] / (s * s);
        const Scalar b = D_sub[k] / (s * s_next);
        const Scalar c = D[k + 1] / (s_next * s_next);
        const Scalar mid = (a + c) / Scalar(2);
        const Scalar r = hypot((a - c) / Scalar(2), abs(b));
        const Scalar λ = mid >= Scalar(0) ? mid + r : mid - r;
        if (is_near_zero((a / λ) * c - (b / λ) * b)) {
          return true;
        }
        ++k;
      } else if (is_near_zero(D[k] / (s * s))) {
        return true;
      }
    }

    return false;
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
