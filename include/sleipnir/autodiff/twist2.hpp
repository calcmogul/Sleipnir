// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_block.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

template <typename Scalar>
class Pose2;

/// A 2D twist of autodiff variables.
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
  Twist2(VariableMatrix<Scalar> linear, Variable<Scalar> angular)
      : m_linear{std::move(linear)}, m_angular{std::move(angular)} {
    slp_assert(m_linear.rows() == 2 && m_linear.cols() == 1);
  }

  /// Returns the linear part (dx, dy).
  ///
  /// @return The linear part.
  VariableBlock<VariableMatrix<Scalar>> linear() {
    return m_linear.block(0, 0, 2, 1);
  }

  /// Returns the linear part (dx, dy).
  ///
  /// @return The linear part.
  const VariableMatrix<Scalar>& linear() const { return m_linear; }

  /// Returns the angular part dθ in radians.
  ///
  /// @return The angular part.
  Variable<Scalar>& angular() { return m_angular; }

  /// Returns the angular part dθ in radians.
  ///
  /// @return The angular part.
  const Variable<Scalar>& angular() const { return m_angular; }

  /// Returns the pose reached by moving along this twist from the identity.
  ///
  /// This is the exponential map of SE(2). Its value and derivatives are
  /// accurate for all twists, including those with zero angular part.
  ///
  /// @return The pose.
  Pose2<Scalar> exp() const;

  /// Scales a twist.
  ///
  /// @param lhs The twist.
  /// @param rhs The factor.
  /// @return The scaled twist.
  friend Twist2 operator*(const Twist2& lhs, const Variable<Scalar>& rhs) {
    return Twist2{lhs.m_linear * rhs, lhs.m_angular * rhs};
  }

  /// Scales a twist.
  ///
  /// @param lhs The factor.
  /// @param rhs The twist.
  /// @return The scaled twist.
  friend Twist2 operator*(const Variable<Scalar>& lhs, const Twist2& rhs) {
    return rhs * lhs;
  }

  /// Adds two twists.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The sum.
  friend Twist2 operator+(const Twist2& lhs, const Twist2& rhs) {
    return Twist2{lhs.m_linear + rhs.m_linear, lhs.m_angular + rhs.m_angular};
  }

  /// Subtracts two twists.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The difference.
  friend Twist2 operator-(const Twist2& lhs, const Twist2& rhs) {
    return Twist2{lhs.m_linear - rhs.m_linear, lhs.m_angular - rhs.m_angular};
  }

  /// Negates a twist.
  ///
  /// @param twist The twist.
  /// @return The negated twist.
  friend Twist2 operator-(const Twist2& twist) {
    return Twist2{-twist.m_linear, -twist.m_angular};
  }

  /// Equality operator that returns an equality constraint for two twists.
  ///
  /// The constraint is that the linear parts are equal and the angular parts
  /// are equal.
  ///
  /// @param lhs Left-hand side.
  /// @param rhs Right-hand side.
  /// @return The equality constraint.
  friend EqualityConstraints<Scalar> operator==(const Twist2& lhs,
                                                const Twist2& rhs) {
    return {lhs.m_linear == rhs.m_linear, lhs.m_angular == rhs.m_angular};
  }

 private:
  VariableMatrix<Scalar> m_linear;
  Variable<Scalar> m_angular;
};

}  // namespace slp
