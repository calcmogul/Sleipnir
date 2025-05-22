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

  /// The log-domain variables.
  ///
  /// The inequality constraint slack variables are s = √(μ)e⁻ᵛ and the
  /// inequality constraint dual variables are z = √(μ)eᵛ.
  Eigen::VectorX<Scalar> v;
};

}  // namespace slp
