// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include <Eigen/Core>

#include "sleipnir/autodiff/rotation2.hpp"
#include "sleipnir/autodiff/translation2.hpp"
#include "sleipnir/autodiff/trig_series.hpp"
#include "sleipnir/autodiff/twist2.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"

namespace slp {

/// A 2D pose of autodiff variables, which represents a rigid transform in
/// SE(2).
///
/// A pose is a translation t and a rotation R. It maps a point p in its own
/// frame to R p + t in the parent frame.
///
/// Poses are created by Problem::decision_variable(se2), from constants, and
/// from a translation and a Rotation2.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class Pose2 {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a constant pose.
  ///
  /// @param translation The translation.
  /// @param angle The rotation angle in radians.
  Pose2(const Eigen::Vector2<Scalar>& translation, Scalar angle)
      : m_translation{translation}, m_rotation{angle} {}

  /// Constructs a pose from a translation and a rotation.
  ///
  /// @param translation The translation.
  /// @param rotation The rotation.
  Pose2(Translation2<Scalar> translation, Rotation2<Scalar> rotation)
      : m_translation{std::move(translation)},
        m_rotation{std::move(rotation)} {}

  /// Returns the translation.
  ///
  /// @return The translation.
  Translation2<Scalar>& translation() { return m_translation; }

  /// Returns the translation.
  ///
  /// @return The translation.
  const Translation2<Scalar>& translation() const { return m_translation; }

  /// Returns the rotation.
  ///
  /// @return The rotation.
  Rotation2<Scalar>& rotation() { return m_rotation; }

  /// Returns the rotation.
  ///
  /// @return The rotation.
  const Rotation2<Scalar>& rotation() const { return m_rotation; }

  /// Composes two poses.
  ///
  /// lhs * rhs transforms by rhs, then by lhs. If rhs is a pose in lhs's
  /// frame, the result is that pose in lhs's parent frame.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The composed pose.
  friend Pose2 operator*(const Pose2& lhs, const Pose2& rhs) {
    return Pose2{rhs.m_translation.transform_by(lhs),
                 lhs.m_rotation * rhs.m_rotation};
  }

  /// Equality operator that returns an equality constraint for two poses.
  ///
  /// The constraint is that the translations are equal and the rotations are
  /// equal. See Rotation2's operator==.
  ///
  /// @param lhs Left-hand side.
  /// @param rhs Right-hand side.
  /// @return The equality constraint.
  friend EqualityConstraints<Scalar> operator==(const Pose2& lhs,
                                                const Pose2& rhs) {
    return {lhs.m_translation == rhs.m_translation,
            lhs.m_rotation == rhs.m_rotation};
  }

  /// Returns the inverse pose.
  ///
  /// It maps points in the parent frame to this pose's frame.
  ///
  /// @return The inverse pose.
  Pose2 inverse() const {
    Rotation2<Scalar> rotation = m_rotation.inverse();
    Translation2<Scalar> translation = -m_translation.rotate_by(rotation);
    return Pose2{std::move(translation), std::move(rotation)};
  }

  /// Returns this pose relative to another pose.
  ///
  /// The result is this pose expressed in the other pose's frame, other⁻¹ *
  /// this. It's often used as the error between a pose and a reference.
  ///
  /// @param other The pose whose frame the result is expressed in.
  /// @return This pose relative to the other pose.
  Pose2 relative_to(const Pose2& other) const {
    return other.inverse() * *this;
  }

  /// Returns the twist that moves from the identity to this pose.
  ///
  /// This is the inverse of Twist2::exp(), and the logarithm map of SE(2). The
  /// twist's angular part is in (−π, π].
  ///
  /// @return The twist.
  Twist2<Scalar> log() const {
    // With θ = the rotation's angle, the linear part is V⁻¹t, where
    //
    //   V⁻¹ = [ α   θ/2]
    //         [−θ/2  α ]
    //
    // and α = (θ/2) cot(θ/2)
    Variable<Scalar> θ = m_rotation.angle();
    Variable<Scalar> half_θ = Scalar(0.5) * θ;
    Variable<Scalar> α = detail::half_theta_cot_half_theta(θ * θ);
    const auto& x = m_translation.x();
    const auto& y = m_translation.y();

    return Twist2<Scalar>{
        VariableMatrix<Scalar>{{α * x + half_θ * y}, {α * y - half_θ * x}}, θ};
  }

  /// Returns the 3x3 homogeneous transformation matrix.
  ///
  /// @return The 3x3 homogeneous transformation matrix.
  VariableMatrix<Scalar> to_matrix() const {
    const auto& c = m_rotation.cos();
    const auto& s = m_rotation.sin();
    const auto& x = m_translation.x();
    const auto& y = m_translation.y();

    return VariableMatrix<Scalar>{{c, -s, x}, {s, c, y}, {0, 0, 1}};
  }

 private:
  Translation2<Scalar> m_translation;
  Rotation2<Scalar> m_rotation;
};

template <typename Scalar>
Pose2<Scalar> Twist2<Scalar>::exp() const {
  // With θ = the angular part, the translation is Vv, where
  //
  //   V = [A  −B]
  //       [B   A]
  //
  // A = sin θ/θ, and B = (1 − cos θ)/θ
  const auto& v = m_linear;
  const auto& θ = m_angular;
  Variable<Scalar> θ_sq = θ * θ;
  Variable<Scalar> A = detail::sin_over_theta(θ_sq);
  Variable<Scalar> B = θ * detail::one_minus_cos_over_theta_sq(θ_sq);

  return Pose2<Scalar>{
      Translation2<Scalar>{A * v[0] - B * v[1], B * v[0] + A * v[1]},
      Rotation2<Scalar>{θ}};
}

template <typename Scalar>
Translation2<Scalar> Translation2<Scalar>::transform_by(
    const Pose2<Scalar>& pose) const {
  return rotate_by(pose.rotation()) + pose.translation();
}

}  // namespace slp
