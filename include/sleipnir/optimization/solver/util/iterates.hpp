// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>

namespace slp {

/// The primal and dual variables of a Sequential Quadratic Programming
/// iterate.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SQPIterates {
  /// The decision variables.
  Eigen::VectorX<Scalar> x;

  /// The equality constraint dual variables.
  Eigen::VectorX<Scalar> y;
};

/// The primal and dual variables of an interior-point method iterate.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct IPMIterates {
  /// The decision variables.
  Eigen::VectorX<Scalar> x;

  /// The equality constraint dual variables.
  Eigen::VectorX<Scalar> y;

  /// The log-domain slack variables.
  ///
  /// The inequality constraint slack variables are s = √(μ)e⁻ᵘ.
  Eigen::VectorX<Scalar> u;

  /// The log-domain dual variables.
  ///
  /// The inequality constraint dual variables are z = √(μ)eᵛ.
  Eigen::VectorX<Scalar> v;
};

}  // namespace slp
