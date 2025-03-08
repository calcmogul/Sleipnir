// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>  // IWYU pragma: keep

#include <Eigen/Core>

namespace slp {

/// The cost function and equality constraints evaluated at a Sequential
/// Quadratic Programming iterate's decision variables.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SQPFunctionValues {
  /// The cost function f(x).
  Scalar f;

  /// The equality constraints cₑ(x).
  Eigen::VectorX<Scalar> c_e;

  /// Returns whether every value is finite.
  ///
  /// @return Whether every value is finite.
  bool all_finite() const {
    using std::isfinite;
    return isfinite(f) && c_e.allFinite();
  }
};

/// The cost function and constraints evaluated at an interior-point method
/// iterate's decision variables.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct IPMFunctionValues {
  /// The cost function f(x).
  Scalar f;

  /// The equality constraints cₑ(x).
  Eigen::VectorX<Scalar> c_e;

  /// The inequality constraints cᵢ(x).
  Eigen::VectorX<Scalar> c_i;

  /// Returns whether every value is finite.
  ///
  /// @return Whether every value is finite.
  bool all_finite() const {
    using std::isfinite;
    return isfinite(f) && c_e.allFinite() && c_i.allFinite();
  }
};

/// The cost function and constraints evaluated at an augmented Lagrangian
/// method iterate's decision variables.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct ALFunctionValues {
  /// The cost function f(x).
  Scalar f;

  /// The equality constraints cₑ(x).
  Eigen::VectorX<Scalar> c_e;

  /// The inequality constraints cᵢ(x).
  Eigen::VectorX<Scalar> c_i;

  /// Returns whether every value is finite.
  ///
  /// @return Whether every value is finite.
  bool all_finite() const {
    using std::isfinite;
    return isfinite(f) && c_e.allFinite() && c_i.allFinite();
  }
};

}  // namespace slp
