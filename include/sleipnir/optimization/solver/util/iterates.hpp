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

  /// The inequality constraint slack variables.
  Eigen::VectorX<Scalar> s;

  /// The equality constraint dual variables.
  Eigen::VectorX<Scalar> y;

  /// The inequality constraint dual variables.
  Eigen::VectorX<Scalar> z;
};

/// The primal and dual variables of an augmented Lagrangian method iterate.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct ALIterates {
  /// The decision variables.
  Eigen::VectorX<Scalar> x;

  /// The equality constraint dual variables.
  Eigen::VectorX<Scalar> y;

  /// The inequality constraint dual variables.
  Eigen::VectorX<Scalar> z;
};

}  // namespace slp
