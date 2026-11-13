// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_block.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

template <typename Scalar>
class Pose3;

/// A 3D twist of autodiff variables.
///
/// A twist is a change in pose along a screw motion, expressed in the frame of
/// the pose it starts from. It's a linear part (dx, dy, dz) and an angular part
/// (rx, ry, rz), a rotation vector in radians.
///
/// exp() returns the pose reached by moving along the twist from the identity,
/// and Pose3::log() is its inverse. A body-frame velocity times a timestep is
/// a twist, so pose * (velocity * dt).exp() integrates a constant velocity for
/// dt.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class Twist3 {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a twist from its linear and angular parts.
  ///
  /// @param linear The linear part (dx, dy, dz). Must be 3x1.
  /// @param angular The angular part (rx, ry, rz) in radians. Must be 3x1.
  Twist3(VariableMatrix<Scalar> linear, VariableMatrix<Scalar> angular)
      : m_linear{std::move(linear)}, m_angular{std::move(angular)} {
    slp_assert(m_linear.rows() == 3 && m_linear.cols() == 1);
    slp_assert(m_angular.rows() == 3 && m_angular.cols() == 1);
  }

  /// Returns the linear part (dx, dy, dz).
  ///
  /// @return The linear part.
  VariableBlock<VariableMatrix<Scalar>> linear() {
    return m_linear.block(0, 0, 3, 1);
  }

  /// Returns the linear part (dx, dy, dz).
  ///
  /// @return The linear part.
  const VariableMatrix<Scalar>& linear() const { return m_linear; }

  /// Returns the angular part (rx, ry, rz) in radians.
  ///
  /// @return The angular part.
  VariableBlock<VariableMatrix<Scalar>> angular() {
    return m_angular.block(0, 0, 3, 1);
  }

  /// Returns the angular part (rx, ry, rz) in radians.
  ///
  /// @return The angular part.
  const VariableMatrix<Scalar>& angular() const { return m_angular; }

  /// Returns the pose reached by moving along this twist from the identity.
  ///
  /// This is the exponential map of SE(3). Its value and derivatives are
  /// accurate for all twists, including those with zero angular part.
  ///
  /// @return The pose.
  Pose3<Scalar> exp() const;

  /// Scales a twist.
  ///
  /// @param lhs The twist.
  /// @param rhs The factor.
  /// @return The scaled twist.
  friend Twist3 operator*(const Twist3& lhs, const Variable<Scalar>& rhs) {
    return Twist3{lhs.m_linear * rhs, lhs.m_angular * rhs};
  }

  /// Scales a twist.
  ///
  /// @param lhs The factor.
  /// @param rhs The twist.
  /// @return The scaled twist.
  friend Twist3 operator*(const Variable<Scalar>& lhs, const Twist3& rhs) {
    return rhs * lhs;
  }

  /// Adds two twists.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The sum.
  friend Twist3 operator+(const Twist3& lhs, const Twist3& rhs) {
    return Twist3{lhs.m_linear + rhs.m_linear, lhs.m_angular + rhs.m_angular};
  }

  /// Subtracts two twists.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The difference.
  friend Twist3 operator-(const Twist3& lhs, const Twist3& rhs) {
    return Twist3{lhs.m_linear - rhs.m_linear, lhs.m_angular - rhs.m_angular};
  }

  /// Negates a twist.
  ///
  /// @param twist The twist.
  /// @return The negated twist.
  friend Twist3 operator-(const Twist3& twist) {
    return Twist3{-twist.m_linear, -twist.m_angular};
  }

  /// Equality operator that returns an equality constraint for two twists.
  ///
  /// The constraint is that the linear parts are equal and the angular parts
  /// are equal.
  ///
  /// @param lhs Left-hand side.
  /// @param rhs Right-hand side.
  /// @return The equality constraint.
  friend EqualityConstraints<Scalar> operator==(const Twist3& lhs,
                                                const Twist3& rhs) {
    return {lhs.m_linear == rhs.m_linear, lhs.m_angular == rhs.m_angular};
  }

 private:
  VariableMatrix<Scalar> m_linear;
  VariableMatrix<Scalar> m_angular;
};

}  // namespace slp
