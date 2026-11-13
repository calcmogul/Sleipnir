// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/util/append_as_triplets.hpp"
#include "sleipnir/optimization/solver/util/kkt_solver.hpp"
#include "sleipnir/optimization/solver/util/manifold.hpp"

namespace slp {

/// Newton's method step direction.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct NewtonStep {
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;

  /// Decision variable primal step.
  DenseVector p_x;

  /// Returns the infinity norm of the primal step.
  ///
  /// @return The infinity norm of the primal step.
  Scalar primal_inf_norm() const {
    return p_x.template lpNorm<Eigen::Infinity>();
  }

  /// Returns the step size below which the primal step is negligible relative
  /// to the iterate.
  ///
  /// @param x The decision variables.
  /// @param manifold The manifold the decision variables lie on.
  /// @return The step size below which the primal step is negligible relative
  ///     to the iterate.
  Scalar tiny_step_size(const DenseVector& x,
                        const Manifold<Scalar>& manifold) const {
    // A primal step αpˣ is negligible if
    //
    //   ‖αJpˣ/(1 + |x|)‖_∞ < 10ε
    //
    // where J is the retraction's Jacobian (the identity in Euclidean space),
    // and the absolute value and division are elementwise. Since α > 0, it
    // factors out of the norm.
    //
    //   α‖Jpˣ/(1 + |x|)‖_∞ < 10ε
    //
    // Dividing both sides by the norm gives
    //
    //   α < 10ε / ‖Jpˣ/(1 + |x|)‖_∞
    return Scalar(10) * std::numeric_limits<Scalar>::epsilon() /
           manifold.relative_step_inf_norm(x, p_x);
  }
};

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

/// Sequential quadratic programming step direction.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SQPStep {
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;

  /// Decision variable primal step.
  DenseVector p_x;
  /// Equality constraint dual step.
  DenseVector p_y;

  /// Returns the infinity norm of the primal step.
  ///
  /// @return The infinity norm of the primal step.
  Scalar primal_inf_norm() const {
    return p_x.template lpNorm<Eigen::Infinity>();
  }

  /// Returns the infinity norm of the dual step.
  ///
  /// @return The infinity norm of the dual step.
  Scalar dual_inf_norm() const {
    return p_y.template lpNorm<Eigen::Infinity>();
  }

  /// Returns the step size below which the primal step is negligible relative
  /// to the iterate.
  ///
  /// @param x The decision variables.
  /// @param manifold The manifold the decision variables lie on.
  /// @return The step size below which the primal step is negligible relative
  ///     to the iterate.
  Scalar tiny_step_size(const DenseVector& x,
                        const Manifold<Scalar>& manifold) const {
    // A primal step αpˣ is negligible if
    //
    //   ‖αJpˣ/(1 + |x|)‖_∞ < 10ε
    //
    // where J is the retraction's Jacobian (the identity in Euclidean space),
    // and the absolute value and division are elementwise. Since α > 0, it
    // factors out of the norm.
    //
    //   α‖Jpˣ/(1 + |x|)‖_∞ < 10ε
    //
    // Dividing both sides by the norm gives
    //
    //   α < 10ε / ‖Jpˣ/(1 + |x|)‖_∞
    return Scalar(10) * std::numeric_limits<Scalar>::epsilon() /
           manifold.relative_step_inf_norm(x, p_x);
  }
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

/// Interior-point method step direction.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct IPMStep {
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;

  /// Decision variable primal step.
  DenseVector p_x;
  /// Inequality constraint slack variable primal step.
  DenseVector p_s;
  /// Equality constraint dual step.
  DenseVector p_y;
  /// Inequality constraint dual step.
  DenseVector p_z;

  /// Returns the infinity norm of the primal step.
  ///
  /// @return The infinity norm of the primal step.
  Scalar primal_inf_norm() const {
    return std::max(p_x.template lpNorm<Eigen::Infinity>(),
                    p_s.template lpNorm<Eigen::Infinity>());
  }

  /// Returns the infinity norm of the dual step.
  ///
  /// @return The infinity norm of the dual step.
  Scalar dual_inf_norm() const {
    return std::max(p_y.template lpNorm<Eigen::Infinity>(),
                    p_z.template lpNorm<Eigen::Infinity>());
  }

  /// Returns the step size below which the primal step is negligible relative
  /// to the iterate.
  ///
  /// @param x The decision variables.
  /// @param s The inequality constraint slack variables.
  /// @param manifold The manifold the decision variables lie on.
  /// @return The step size below which the primal step is negligible relative
  ///     to the iterate.
  Scalar tiny_step_size(const DenseVector& x, const DenseVector& s,
                        const Manifold<Scalar>& manifold) const {
    // A primal step α[pˣ pˢ] is negligible if
    //
    //   max(‖αJpˣ/(1 + |x|)‖_∞, ‖αpˢ/(1 + |s|)‖_∞) < 10ε
    //
    // where J is the retraction's Jacobian (the identity in Euclidean space),
    // and the absolute values and division are elementwise. Since α > 0, it
    // factors out of the norms and max.
    //
    //   α max(‖Jpˣ/(1 + |x|)‖_∞, ‖pˢ/(1 + |s|)‖_∞) < 10ε
    //
    // Dividing both sides by the max gives
    //
    //   α < 10ε / max(‖Jpˣ/(1 + |x|)‖_∞, ‖pˢ/(1 + |s|)‖_∞)
    return Scalar(10) * std::numeric_limits<Scalar>::epsilon() /
           std::max(manifold.relative_step_inf_norm(x, p_x),
                    (p_s.array().abs() / (Scalar(1) + s.array().abs()))
                        .matrix()
                        .template lpNorm<Eigen::Infinity>());
  }
};

/// The interior-point method's Newton-KKT system.
///
/// ```
/// [H + AᵢᵀΣAᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(μS⁻¹e − Σ(cᵢ − s))]
/// [    Aₑ       0 ][−pʸ]    [                cₑ                ]
///
/// pˢ = cᵢ − s + Aᵢpˣ
/// pᶻ = μS⁻¹e − z − Σpˢ
/// ```
///
/// where S = diag(s), Z = diag(z), and Σ = S⁻¹Z.
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
  /// @param s The inequality constraint slack variables.
  /// @param z The inequality constraint dual variables.
  void build(const SparseMatrix& H, const SparseMatrix& A_e,
             const SparseMatrix& A_i, const DenseVector& s,
             const DenseVector& z) {
    // Σ = S⁻¹Z
    m_Σ = SparseMatrix{s.cwiseInverse().asDiagonal() * z.asDiagonal()};

    // lhs = [H + AᵢᵀΣAᵢ  Aₑᵀ]
    //       [    Aₑ       0 ]
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    const SparseMatrix top_left =
        H +
        (A_i.transpose() * m_Σ * A_i).template triangularView<Eigen::Lower>();
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
  /// @param s The inequality constraint slack variables.
  /// @param z The inequality constraint dual variables.
  /// @param μ The barrier parameter.
  /// @param c_e The equality constraint values.
  /// @param c_i_minus_s The inequality constraint values minus the slack
  ///     variables.
  /// @return The step.
  IPMStep<Scalar> solve(const SparseVector& g, const SparseMatrix& A_e,
                        const SparseMatrix& A_i, const DenseVector& y,
                        const DenseVector& s, const DenseVector& z, Scalar μ,
                        const DenseVector& c_e,
                        const DenseVector& c_i_minus_s) const {
    // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(μS⁻¹e − Σ(cᵢ − s))]
    //        [                cₑ                ]
    DenseVector rhs{g.rows() + y.rows()};
    rhs.segment(0, g.rows()) =
        -g + A_e.transpose() * y +
        A_i.transpose() * (μ * s.cwiseInverse() - m_Σ * c_i_minus_s);
    rhs.segment(g.rows(), y.rows()) = -c_e;

    // p = [ pˣ]
    //     [−pʸ]
    const DenseVector p = m_solver.solve(rhs);

    IPMStep<Scalar> step;
    step.p_x = p.segment(0, g.rows());
    step.p_y = -p.segment(g.rows(), y.rows());

    // pˢ = cᵢ − s + Aᵢpˣ
    // pᶻ = μS⁻¹e − z − Σpˢ
    step.p_s = c_i_minus_s + A_i * step.p_x;
    step.p_z = μ * s.cwiseInverse() - z - m_Σ * step.p_s;

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
  SparseMatrix m_Σ;
  SparseMatrix m_lhs;
  gch::small_vector<Eigen::Triplet<Scalar>> m_triplets;
};

}  // namespace slp
