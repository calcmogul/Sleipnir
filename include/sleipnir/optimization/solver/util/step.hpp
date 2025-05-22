// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <limits>

#include <Eigen/Core>

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
  /// @return The step size below which the primal step is negligible relative
  ///     to the iterate.
  Scalar tiny_step_size(const DenseVector& x) const {
    // A primal step αpˣ is negligible if
    //
    //   ‖αpˣ/(1 + |x|)‖_∞ < 10ε
    //
    // where the absolute value and division are elementwise. Since α > 0, it
    // factors out of the norm.
    //
    //   α‖pˣ/(1 + |x|)‖_∞ < 10ε
    //
    // Dividing both sides by the norm gives
    //
    //   α < 10ε / ‖pˣ/(1 + |x|)‖_∞
    return Scalar(10) * std::numeric_limits<Scalar>::epsilon() /
           (p_x.array().abs() / (Scalar(1) + x.array().abs()))
               .matrix()
               .template lpNorm<Eigen::Infinity>();
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
  /// @return The step size below which the primal step is negligible relative
  ///     to the iterate.
  Scalar tiny_step_size(const DenseVector& x) const {
    // A primal step αpˣ is negligible if
    //
    //   ‖αpˣ/(1 + |x|)‖_∞ < 10ε
    //
    // where the absolute value and division are elementwise. Since α > 0, it
    // factors out of the norm.
    //
    //   α‖pˣ/(1 + |x|)‖_∞ < 10ε
    //
    // Dividing both sides by the norm gives
    //
    //   α < 10ε / ‖pˣ/(1 + |x|)‖_∞
    return Scalar(10) * std::numeric_limits<Scalar>::epsilon() /
           (p_x.array().abs() / (Scalar(1) + x.array().abs()))
               .matrix()
               .template lpNorm<Eigen::Infinity>();
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
  /// Log-domain variable step.
  DenseVector p_v;

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
    return p_v.template lpNorm<Eigen::Infinity>();
  }

  /// Returns the step size below which the primal step is negligible relative
  /// to the iterate.
  ///
  /// @param x The decision variables.
  /// @param s The inequality constraint slack variables.
  /// @return The step size below which the primal step is negligible relative
  ///     to the iterate.
  Scalar tiny_step_size(const DenseVector& x, const DenseVector& s) const {
    // The slack variables take a linear step in s-space with pˢ = −s∘pᵛ. A
    // primal step α[pˣ pˢ] is negligible if
    //
    //   max(‖αpˣ/(1 + |x|)‖_∞, ‖αpˢ/(1 + |s|)‖_∞) < 10ε
    //
    // where the absolute values and division are elementwise. Since α > 0, it
    // factors out of the norms and max.
    //
    //   α max(‖pˣ/(1 + |x|)‖_∞, ‖pˢ/(1 + |s|)‖_∞) < 10ε
    //
    // Dividing both sides by the max gives
    //
    //   α < 10ε / max(‖pˣ/(1 + |x|)‖_∞, ‖pˢ/(1 + |s|)‖_∞)
    return Scalar(10) * std::numeric_limits<Scalar>::epsilon() /
           std::max(
               (p_x.array().abs() / (Scalar(1) + x.array().abs()))
                   .matrix()
                   .template lpNorm<Eigen::Infinity>(),
               ((s.array() * p_v.array()).abs() / (Scalar(1) + s.array().abs()))
                   .matrix()
                   .template lpNorm<Eigen::Infinity>());
  }
};

}  // namespace slp
