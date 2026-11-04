// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "sleipnir/optimization/solver/util/regularization_limits.hpp"

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
  using DenseMatrix = Eigen::MatrixX<Scalar>;
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;

  /// Constructs a DenseKKTSolver instance.
  ///
  /// @param num_decision_variables The number of decision variables in the
  ///     system.
  /// @param num_equality_constraints The number of equality constraints in the
  ///     system.
  DenseKKTSolver(int num_decision_variables, int num_equality_constraints)
      : m_num_decision_variables{num_decision_variables},
        m_num_equality_constraints{num_equality_constraints} {}

  /// Reports whether previous computation was successful.
  ///
  /// @return Whether previous computation was successful.
  Eigen::ComputationInfo info() const { return m_info; }

  /// Computes the factorization of the KKT matrix.
  ///
  /// @param lhs Left-hand side of the system.
  /// @return The factorization.
  DenseKKTSolver& compute(const DenseMatrix& lhs) {
    // Bunch-Kaufman pivoting is backward stable for indefinite matrices, so the
    // unregularized system is accepted whenever its inertia is ideal
    m_lhs = lhs;
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

    // The factorization is backward stable, but the matrix's conditioning
    // amplifies its rounding error in the solution, so apply iterative
    // refinement with the same factorization
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
  using Solver = Eigen::BunchKaufman<DenseMatrix>;

  Solver m_solver;

  /// The most recently factorized KKT matrix's lower triangle.
  DenseMatrix m_lhs;

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
  void factorize() { m_info = m_solver.compute(m_lhs).info(); }

  /// Solves the system of equations with the most recent factorization.
  ///
  /// @param rhs Right-hand side of the system.
  /// @return The solution.
  DenseVector solve_factorized(const DenseVector& rhs) const {
    return m_solver.solve(rhs);
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
  /// D from the Bunch-Kaufman factorization has 1x1 and 2x2 diagonal blocks.
  /// Eigenvalues within zero_pivot_threshold() of zero are counted as zero. If
  /// there aren't any, the factorization's sign counts are the inertia.
  ///
  /// @return Whether the most recently factorized KKT matrix has an eigenvalue
  ///     that should be counted as zero.
  bool has_near_zero_eigenvalue() const {
    using std::abs;
    using std::hypot;

    const Scalar threshold = zero_pivot_threshold<Scalar>();

    const auto D = m_solver.vectorD();
    const auto& D_sub = m_solver.subDiagonal();

    // This treats NaN as zero
    auto is_near_zero = [&](Scalar λ) { return !(abs(λ) > threshold); };

    for (int k = 0; k < D.size(); ++k) {
      if (D_sub[k] != Scalar(0)) {
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
        const Scalar a = D[k];
        const Scalar b = D_sub[k];
        const Scalar c = D[k + 1];
        const Scalar mid = (a + c) / Scalar(2);
        const Scalar r = hypot((a - c) / Scalar(2), abs(b));
        const Scalar λ = mid >= Scalar(0) ? mid + r : mid - r;
        if (is_near_zero((a / λ) * c - (b / λ) * b)) {
          return true;
        }
        ++k;
      } else if (is_near_zero(D[k])) {
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
  DenseMatrix regularization(Scalar δ, Scalar γ) const {
    DenseVector vec{m_num_decision_variables + m_num_equality_constraints};
    vec.segment(0, m_num_decision_variables).setConstant(δ);
    vec.segment(m_num_decision_variables, m_num_equality_constraints)
        .setConstant(-γ);

    return vec.asDiagonal().toDenseMatrix();
  }
};

}  // namespace slp
