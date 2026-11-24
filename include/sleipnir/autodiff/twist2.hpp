// Copyright (c) Sleipnir contributors

#pragma once

#include <type_traits>
#include <utility>

#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/autodiff/real_submatrix.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

template <typename Scalar>
class Pose2;

/// A 2D twist of autodiff expressions.
///
/// A twist is a change in pose along an arc of constant curvature, expressed
/// in the frame of the pose it starts from. It's a linear part (dx, dy) and an
/// angular part dθ in radians.
///
/// exp() returns the pose reached by moving along the twist from the identity,
/// and Pose2::log() is its inverse. A body-frame velocity times a timestep is
/// a twist, so pose * (velocity * dt).exp() integrates a constant velocity for
/// dt.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class Twist2 {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a twist from its linear and angular parts.
  ///
  /// @param linear The linear part (dx, dy). Must be 2x1.
  /// @param angular The angular part dθ in radians.
  Twist2(RealMatrix<Scalar> linear, Real<Scalar> angular)
      : m_linear{std::move(linear)}, m_angular{std::move(angular)} {
    slp_assert(m_linear.rows() == 2 && m_linear.cols() == 1);
  }

  /// Returns the linear part (dx, dy).
  ///
  /// @return The linear part.
  RealSubmatrix<RealMatrix<Scalar>> linear() {
    return m_linear.block(0, 0, 2, 1);
  }

  /// Returns the linear part (dx, dy).
  ///
  /// @return The linear part.
  const RealMatrix<Scalar>& linear() const { return m_linear; }

  /// Returns the angular part dθ in radians.
  ///
  /// @return The angular part.
  Real<Scalar>& angular() { return m_angular; }

  /// Returns the angular part dθ in radians.
  ///
  /// @return The angular part.
  const Real<Scalar>& angular() const { return m_angular; }

  /// Returns the pose reached by moving along this twist from the identity.
  ///
  /// This is the exponential map of SE(2). Its value and derivatives are
  /// accurate for all twists, including those with zero angular part.
  ///
  /// @return The pose.
  Pose2<Scalar> exp() const;

 private:
  RealMatrix<Scalar> m_linear;
  Real<Scalar> m_angular;
};

/// Scales a twist.
///
/// @param lhs The twist.
/// @param rhs The factor.
/// @return The scaled twist.
template <typename Scalar>
Twist2<Scalar> operator*(const Twist2<Scalar>& lhs,
                         const std::type_identity_t<Real<Scalar>>& rhs) {
  return Twist2<Scalar>{lhs.linear() * rhs, lhs.angular() * rhs};
}

/// Scales a twist.
///
/// @param lhs The factor.
/// @param rhs The twist.
/// @return The scaled twist.
template <typename Scalar>
Twist2<Scalar> operator*(const std::type_identity_t<Real<Scalar>>& lhs,
                         const Twist2<Scalar>& rhs) {
  return rhs * lhs;
}

/// Adds two twists.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar>
Twist2<Scalar> operator+(const Twist2<Scalar>& lhs, const Twist2<Scalar>& rhs) {
  return Twist2<Scalar>{lhs.linear() + rhs.linear(),
                        lhs.angular() + rhs.angular()};
}

/// Subtracts two twists.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar>
Twist2<Scalar> operator-(const Twist2<Scalar>& lhs, const Twist2<Scalar>& rhs) {
  return Twist2<Scalar>{lhs.linear() - rhs.linear(),
                        lhs.angular() - rhs.angular()};
}

/// Negates a twist.
///
/// @param twist The twist.
/// @return The negated twist.
template <typename Scalar>
Twist2<Scalar> operator-(const Twist2<Scalar>& twist) {
  return Twist2<Scalar>{-twist.linear(), -twist.angular()};
}

/// Equality operator that returns an equality constraint for two twists.
///
/// The constraint is that the linear parts are equal and the angular parts
/// are equal.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraint.
template <typename Scalar>
EqualityConstraints<Scalar> operator==(const Twist2<Scalar>& lhs,
                                       const Twist2<Scalar>& rhs) {
  return {lhs.linear() == rhs.linear(), lhs.angular() == rhs.angular()};
}

}  // namespace slp
