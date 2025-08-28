// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/util/append_as_triplets.hpp"
#include "sleipnir/optimization/solver/util/kkt_solver.hpp"
#include "sleipnir/optimization/solver/util/step.hpp"

namespace slp {

/// Newton's method's Newton-KKT system.
///
/// ```
/// Hpˣ = −∇f
/// ```
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class NewtonKKTSystem {
 public:
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  /// Type alias for sparse vector.
  using SparseVector = Eigen::SparseVector<Scalar>;

  /// Constructs the system for a problem with the given Hessian sparsity
  /// pattern.
  ///
  /// @param H The Hessian of the cost function.
  explicit NewtonKKTSystem(const SparseMatrix& H)
      // Use sparse solver if lower triangle fills < 25% of system
      : m_solver{H.nonZeros() < 0.25 * H.size(), static_cast<int>(H.rows()),
                 0} {}

  /// Factorizes the system's matrix at the given iterate.
  ///
  /// @param H The Hessian of the cost function.
  /// @return True if the factorization succeeded.
  bool factorize(const SparseMatrix& H) {
    return m_solver.compute(H).info() == Eigen::Success;
  }

  /// Solves the system with the last factorization.
  ///
  /// @param g The cost function gradient.
  /// @return The step.
  NewtonStep<Scalar> solve(const SparseVector& g) const {
    return NewtonStep<Scalar>{m_solver.solve(-g)};
  }

  /// Returns the Hessian regularization factor from the last factorize().
  ///
  /// @return Hessian regularization factor.
  Scalar hessian_regularization() const {
    return m_solver.hessian_regularization();
  }

  /// Returns the constraint Jacobian regularization factor from the last
  /// factorize().
  ///
  /// @return Constraint Jacobian regularization factor.
  Scalar constraint_jacobian_regularization() const {
    return m_solver.constraint_jacobian_regularization();
  }

 private:
  KKTSolver<Scalar> m_solver;
};

/// The sequential quadratic programming method's Newton-KKT system.
///
/// ```
/// [H   Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy]
/// [Aₑ   0 ][−pʸ]    [   cₑ    ]
/// ```
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class SQPKKTSystem {
 public:
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  /// Type alias for sparse vector.
  using SparseVector = Eigen::SparseVector<Scalar>;

  /// Constructs the system for a problem with the given derivative sparsity
  /// patterns.
  ///
  /// @param H The Hessian of the Lagrangian.
  /// @param A_e The equality constraint Jacobian.
  SQPKKTSystem(const SparseMatrix& H, const SparseMatrix& A_e)
      : m_solver{[&] {
                   // Use sparse solver if lower triangle fills < 25% of system
                   const auto rows = H.rows() + A_e.rows();
                   return H.nonZeros() + A_e.nonZeros() < 0.25 * rows * rows;
                 }(),
                 static_cast<int>(H.rows()), static_cast<int>(A_e.rows())} {}

  /// Builds the system's matrix at the given iterate.
  ///
  /// @param H The Hessian of the Lagrangian.
  /// @param A_e The equality constraint Jacobian.
  void build(const SparseMatrix& H, const SparseMatrix& A_e) {
    // lhs = [H   Aₑᵀ]
    //       [Aₑ   0 ]
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    m_triplets.clear();
    m_triplets.reserve(H.nonZeros() + A_e.nonZeros());
    append_as_triplets(m_triplets, 0, 0, {H, A_e});
    m_lhs.resize(H.rows() + A_e.rows(), H.rows() + A_e.rows());
    m_lhs.setFromSortedTriplets(m_triplets.begin(), m_triplets.end());
  }

  /// Factorizes the matrix from the last build().
  ///
  /// @return True if the factorization succeeded.
  bool factorize() { return m_solver.compute(m_lhs).info() == Eigen::Success; }

  /// Solves the system with the last factorization.
  ///
  /// @param g The cost function gradient.
  /// @param A_e The equality constraint Jacobian.
  /// @param y The equality constraint dual variables.
  /// @param c_e The equality constraint values.
  /// @return The step.
  SQPStep<Scalar> solve(const SparseVector& g, const SparseMatrix& A_e,
                        const DenseVector& y, const DenseVector& c_e) const {
    // rhs = −[∇f − Aₑᵀy]
    //        [   cₑ    ]
    DenseVector rhs{g.rows() + y.rows()};
    rhs.segment(0, g.rows()) = -g + A_e.transpose() * y;
    rhs.segment(g.rows(), y.rows()) = -c_e;

    // p = [ pˣ]
    //     [−pʸ]
    const DenseVector p = m_solver.solve(rhs);

    SQPStep<Scalar> step;
    step.p_x = p.segment(0, g.rows());
    step.p_y = -p.segment(g.rows(), y.rows());
    return step;
  }

  /// Returns the Hessian regularization factor from the last factorize().
  ///
  /// @return Hessian regularization factor.
  Scalar hessian_regularization() const {
    return m_solver.hessian_regularization();
  }

  /// Returns the constraint Jacobian regularization factor from the last
  /// factorize().
  ///
  /// @return Constraint Jacobian regularization factor.
  Scalar constraint_jacobian_regularization() const {
    return m_solver.constraint_jacobian_regularization();
  }

 private:
  KKTSolver<Scalar> m_solver;
  SparseMatrix m_lhs;
  gch::small_vector<Eigen::Triplet<Scalar>> m_triplets;
};

/// The log-domain interior-point method's Newton-KKT system.
///
/// ```
/// [H + Aᵢᵀdiag(eᵘ⁺ᵛ)Aᵢ  Aₑᵀ][ pˣ] =
/// [        Aₑ            0 ][−pʸ]
///     −[∇f − Aₑᵀy − Aᵢᵀ(√(μ)eᵘ + √(μ)eᵛ − eᵘ⁺ᵛ∘cᵢ)]
///      [                    cₑ                    ]
///
/// pᵘ = e − 1/√(μ) eᵘ∘(Aᵢpˣ + cᵢ)
/// pᵛ = pᵘ + eᵘ⁻ᵛ − e
/// ```
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class IPMKKTSystem {
 public:
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  /// Type alias for sparse vector.
  using SparseVector = Eigen::SparseVector<Scalar>;

  /// Constructs the system for a problem with the given derivative sparsity
  /// patterns.
  ///
  /// @param H The Hessian of the Lagrangian.
  /// @param A_e The equality constraint Jacobian.
  /// @param A_i The inequality constraint Jacobian.
  IPMKKTSystem(const SparseMatrix& H, const SparseMatrix& A_e,
               const SparseMatrix& A_i)
      : m_solver{[&] {
                   // Use sparse solver if lower triangle fills < 25% of system
                   const auto rows = H.rows() + A_e.rows();
                   return H.nonZeros() +
                              (A_i.transpose() * A_i)
                                  .template triangularView<Eigen::Lower>()
                                  .eval()
                                  .nonZeros() +
                              A_e.nonZeros() <
                          0.25 * rows * rows;
                 }(),
                 static_cast<int>(H.rows()), static_cast<int>(A_e.rows())} {}

  /// Builds the system's matrix at the given iterate.
  ///
  /// @param H The Hessian of the Lagrangian.
  /// @param A_e The equality constraint Jacobian.
  /// @param A_i The inequality constraint Jacobian.
  /// @param exp_u eᵘ, where u is the log-domain slack variables.
  /// @param exp_v eᵛ, where v is the log-domain dual variables.
  void build(const SparseMatrix& H, const SparseMatrix& A_e,
             const SparseMatrix& A_i, const DenseVector& exp_u,
             const DenseVector& exp_v) {
    m_exp_u = exp_u;
    m_exp_v = exp_v;
    m_exp_u_plus_v = exp_u.cwiseProduct(exp_v);
    m_exp_u_minus_v = exp_u.cwiseProduct(exp_v.cwiseInverse());

    // lhs = [H + Aᵢᵀdiag(eᵘ⁺ᵛ)Aᵢ  Aₑᵀ]
    //       [        Aₑ            0 ]
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    const SparseMatrix top_left =
        H + (A_i.transpose() * m_exp_u_plus_v.asDiagonal() * A_i)
                .template triangularView<Eigen::Lower>();
    m_triplets.clear();
    m_triplets.reserve(top_left.nonZeros() + A_e.nonZeros());
    append_as_triplets(m_triplets, 0, 0, {top_left, A_e});
    m_lhs.resize(H.rows() + A_e.rows(), H.rows() + A_e.rows());
    m_lhs.setFromSortedTriplets(m_triplets.begin(), m_triplets.end());
  }

  /// Factorizes the matrix from the last build().
  ///
  /// @return True if the factorization succeeded.
  bool factorize() { return m_solver.compute(m_lhs).info() == Eigen::Success; }

  /// Solves the system with the last factorization.
  ///
  /// @param g The cost function gradient.
  /// @param A_e The equality constraint Jacobian.
  /// @param A_i The inequality constraint Jacobian.
  /// @param y The equality constraint dual variables.
  /// @param sqrt_μ Square root of the barrier parameter.
  /// @param c_e The equality constraint values.
  /// @param c_i The inequality constraint values.
  /// @return The step.
  IPMStep<Scalar> solve(const SparseVector& g, const SparseMatrix& A_e,
                        const SparseMatrix& A_i, const DenseVector& y,
                        Scalar sqrt_μ, const DenseVector& c_e,
                        const DenseVector& c_i) const {
    // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(√(μ)eᵘ + √(μ)eᵛ − eᵘ⁺ᵛ∘cᵢ)]
    //        [                    cₑ                    ]
    DenseVector rhs{g.rows() + y.rows()};
    rhs.segment(0, g.rows()) =
        -g + A_e.transpose() * y +
        A_i.transpose() * (sqrt_μ * m_exp_u + sqrt_μ * m_exp_v -
                           m_exp_u_plus_v.asDiagonal() * c_i);
    rhs.segment(g.rows(), y.rows()) = -c_e;

    // p = [ pˣ]
    //     [−pʸ]
    const DenseVector p = m_solver.solve(rhs);

    IPMStep<Scalar> step;
    step.p_x = p.segment(0, g.rows());
    step.p_y = -p.segment(g.rows(), y.rows());

    // pᵘ = e − 1/√(μ) eᵘ∘(Aᵢpˣ + cᵢ)
    step.p_u = DenseVector::Ones(c_i.rows()) - Scalar(1) / sqrt_μ *
                                                   m_exp_u.asDiagonal() *
                                                   (A_i * step.p_x + c_i);

    // pᵛ = pᵘ + eᵘ⁻ᵛ − e
    step.p_v = step.p_u + m_exp_u_minus_v - DenseVector::Ones(c_i.rows());

    return step;
  }

  /// Solves the system with the last factorization for a second-order
  /// correction.
  ///
  /// Since eᵘ⁺ᵛ∘s = √(μ)eᵛ, the rhs can be written in terms of the inequality
  /// constraint residual cᵢ − s.
  ///
  /// ```
  /// rhs = −[∇f − Aₑᵀy − Aᵢᵀ(√(μ)eᵘ − eᵘ⁺ᵛ∘(cᵢ − s)ˢᵒᶜ)]
  ///        [                  cₑˢᵒᶜ                   ]
  ///
  /// pᵘ = −1/√(μ) eᵘ∘(Aᵢpˣ + (cᵢ − s)ˢᵒᶜ)
  /// pᵛ = pᵘ + eᵘ⁻ᵛ − e
  /// ```
  ///
  /// @param g The cost function gradient.
  /// @param A_e The equality constraint Jacobian.
  /// @param A_i The inequality constraint Jacobian.
  /// @param y The equality constraint dual variables.
  /// @param sqrt_μ Square root of the barrier parameter.
  /// @param c_e_soc The corrected equality constraint values.
  /// @param c_i_minus_s_soc The corrected inequality constraint values minus
  ///     the slack variables.
  /// @return The step.
  IPMStep<Scalar> solve_soc(const SparseVector& g, const SparseMatrix& A_e,
                            const SparseMatrix& A_i, const DenseVector& y,
                            Scalar sqrt_μ, const DenseVector& c_e_soc,
                            const DenseVector& c_i_minus_s_soc) const {
    DenseVector rhs{g.rows() + y.rows()};
    rhs.segment(0, g.rows()) =
        -g + A_e.transpose() * y +
        A_i.transpose() *
            (sqrt_μ * m_exp_u - m_exp_u_plus_v.asDiagonal() * c_i_minus_s_soc);
    rhs.segment(g.rows(), y.rows()) = -c_e_soc;

    // p = [ pˣ]
    //     [−pʸ]
    const DenseVector p = m_solver.solve(rhs);

    IPMStep<Scalar> step;
    step.p_x = p.segment(0, g.rows());
    step.p_y = -p.segment(g.rows(), y.rows());
    step.p_u = -(Scalar(1) / sqrt_μ) * m_exp_u.asDiagonal() *
               (A_i * step.p_x + c_i_minus_s_soc);
    step.p_v =
        step.p_u + m_exp_u_minus_v - DenseVector::Ones(c_i_minus_s_soc.rows());

    return step;
  }

  /// Returns the Hessian regularization factor from the last factorize().
  ///
  /// @return Hessian regularization factor.
  Scalar hessian_regularization() const {
    return m_solver.hessian_regularization();
  }

  /// Returns the constraint Jacobian regularization factor from the last
  /// factorize().
  ///
  /// @return Constraint Jacobian regularization factor.
  Scalar constraint_jacobian_regularization() const {
    return m_solver.constraint_jacobian_regularization();
  }

 private:
  KKTSolver<Scalar> m_solver;
  DenseVector m_exp_u;
  DenseVector m_exp_v;
  DenseVector m_exp_u_plus_v;
  DenseVector m_exp_u_minus_v;
  SparseMatrix m_lhs;
  gch::small_vector<Eigen::Triplet<Scalar>> m_triplets;
};

}  // namespace slp
