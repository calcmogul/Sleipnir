// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <limits>

#include <Eigen/Core>

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

}  // namespace slp
