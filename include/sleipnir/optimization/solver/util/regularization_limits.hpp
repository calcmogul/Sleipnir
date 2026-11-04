// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace slp {

/// Returns the minimum nonzero Hessian regularization.
///
/// @tparam Scalar Scalar type.
/// @return The minimum nonzero Hessian regularization.
template <typename Scalar>
Scalar min_hessian_regularization() {
  using std::sqrt;

  // This value keeps regularized pivots well above the inertia's zero-pivot
  // threshold
  return sqrt(std::numeric_limits<Scalar>::epsilon()) / Scalar(100);
}

/// Returns the minimum nonzero equality constraint Jacobian regularization.
///
/// @tparam Scalar Scalar type.
/// @return The minimum nonzero equality constraint Jacobian regularization.
template <typename Scalar>
Scalar min_constraint_regularization() {
  using std::sqrt;

  // γ is bounded on both sides:
  //
  //   * From below by rounding. A dual pivot is −γ − aᵀD⁻¹a, where a is the
  //     constraint's row of the Jacobian, so γ smaller than the rounding error
  //     in aᵀD⁻¹a has no effect. Below about 100ε in double, nearly dependent
  //     constraints produce exact zero pivots.
  //   * From above by the distortion of the equality constraint Jacobian,
  //     which degrades steps once γ is around 1e-7 to 1e-6 regardless of
  //     Scalar.
  //
  // The upper bound doesn't scale with ε, so the window is narrow for float
  // (about 0.25ε to 2.5ε) and wide for double (about 100ε to 1e-8). This value
  // falls inside both windows.
  return sqrt(std::numeric_limits<Scalar>::epsilon()) / Scalar(3000);
}

/// Returns the largest LDLT pivot magnitude the KKT matrix inertia counts as
/// zero.
///
/// @tparam Scalar Scalar type.
/// @return The largest LDLT pivot magnitude the KKT matrix inertia counts as
///     zero.
template <typename Scalar>
Scalar zero_pivot_threshold() {
  using std::min;

  // The threshold must be below the regularized pivots, or the regularization
  // can never fix the inertia. For float, ε exceeds the minimum constraint
  // regularization.
  return min(std::numeric_limits<Scalar>::epsilon(),
             min_constraint_regularization<Scalar>() / Scalar(100));
}

/// Returns the maximum equality constraint Jacobian regularization.
///
/// @tparam Scalar Scalar type.
/// @return The maximum equality constraint Jacobian regularization.
template <typename Scalar>
Scalar max_constraint_regularization() {
  using std::sqrt;

  // Larger values than this distort the equality constraint Jacobian more than
  // they help a rank-deficient one
  return sqrt(std::numeric_limits<Scalar>::epsilon());
}

}  // namespace slp
