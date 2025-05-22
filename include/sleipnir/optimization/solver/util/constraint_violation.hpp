// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>

namespace slp {

/// Returns the constraint violation for a Sequential Quadratic Programming
/// iterate.
///
/// @tparam Scalar Scalar type.
/// @param c_e The equality constraint values (nonzero means violation).
/// @return The constraint violation.
template <typename Scalar>
Scalar constraint_violation(const Eigen::Vector<Scalar, Eigen::Dynamic>& c_e) {
  return c_e.template lpNorm<1>();
}

/// Returns the constraint violation for an interior-point method iterate.
///
/// @tparam Scalar Scalar type.
/// @param c_i The inequality constraint values (negative means violation).
/// @param s The inequality constraint slack variables.
/// @return The constraint violation.
template <typename Scalar>
Scalar constraint_violation(const Eigen::Vector<Scalar, Eigen::Dynamic>& c_i,
                            const Eigen::Vector<Scalar, Eigen::Dynamic>& s) {
  return (c_i - s).template lpNorm<1>();
}

}  // namespace slp
