// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sleipnir/autodiff/rotation3.hpp"
#include "sleipnir/autodiff/translation3.hpp"
#include "sleipnir/autodiff/trig_series.hpp"
#include "sleipnir/autodiff/twist3.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"

namespace slp {

namespace detail {

/// Returns the cross product of two 3D vectors.
///
/// @param a The first vector.
/// @param b The second vector.
/// @return a × b.
template <typename Scalar>
VariableMatrix<Scalar> cross(const VariableMatrix<Scalar>& a,
                             const VariableMatrix<Scalar>& b) {
  return VariableMatrix<Scalar>{{a[1] * b[2] - a[2] * b[1]},
                                {a[2] * b[0] - a[0] * b[2]},
                                {a[0] * b[1] - a[1] * b[0]}};
}

}  // namespace detail

/// A 3D pose of autodiff variables, which represents a rigid transform in
/// SE(3).
///
/// A pose is a translation t and a rotation R. It maps a point p in its own
/// frame to R p + t in the parent frame.
///
/// Poses are created by Problem::decision_variable(se3), from constants, and
/// from a translation and a Rotation3.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class Pose3 {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a constant pose.
  ///
  /// @param translation The translation.
  /// @param rotation The rotation. It's normalized, so it must have nonzero
  ///     norm.
  Pose3(const Eigen::Vector3<Scalar>& translation,
        const Eigen::Quaternion<Scalar>& rotation)
      : m_translation{translation}, m_rotation{rotation} {}

  /// Constructs a pose from a translation and a rotation.
  ///
  /// @param translation The translation.
  /// @param rotation The rotation.
  Pose3(Translation3<Scalar> translation, Rotation3<Scalar> rotation)
      : m_translation{std::move(translation)},
        m_rotation{std::move(rotation)} {}

  /// Returns the translation.
  ///
  /// @return The translation.
  Translation3<Scalar>& translation() { return m_translation; }

  /// Returns the translation.
  ///
  /// @return The translation.
  const Translation3<Scalar>& translation() const { return m_translation; }

  /// Returns the rotation.
  ///
  /// @return The rotation.
  Rotation3<Scalar>& rotation() { return m_rotation; }

  /// Returns the rotation.
  ///
  /// @return The rotation.
  const Rotation3<Scalar>& rotation() const { return m_rotation; }

  /// Composes two poses.
  ///
  /// lhs * rhs transforms by rhs, then by lhs. If rhs is a pose in lhs's
  /// frame, the result is that pose in lhs's parent frame.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The composed pose.
  friend Pose3 operator*(const Pose3& lhs, const Pose3& rhs) {
    return Pose3{rhs.m_translation.transform_by(lhs),
                 lhs.m_rotation * rhs.m_rotation};
  }

  /// Equality operator that returns an equality constraint for two poses.
  ///
  /// The constraint is that the translations are equal and the rotations are
  /// equal. See Rotation3's operator==.
  ///
  /// @param lhs Left-hand side.
  /// @param rhs Right-hand side.
  /// @return The equality constraint.
  friend EqualityConstraints<Scalar> operator==(const Pose3& lhs,
                                                const Pose3& rhs) {
    return {lhs.m_translation == rhs.m_translation,
            lhs.m_rotation == rhs.m_rotation};
  }

  /// Returns the inverse pose.
  ///
  /// It maps points in the parent frame to this pose's frame.
  ///
  /// @return The inverse pose.
  Pose3 inverse() const {
    Rotation3<Scalar> rotation = m_rotation.inverse();
    Translation3<Scalar> translation = -m_translation.rotate_by(rotation);
    return Pose3{std::move(translation), std::move(rotation)};
  }

  /// Returns this pose relative to another pose.
  ///
  /// The result is this pose expressed in the other pose's frame, other⁻¹ *
  /// this. It's often used as the error between a pose and a reference.
  ///
  /// @param other The pose whose frame the result is expressed in.
  /// @return This pose relative to the other pose.
  Pose3 relative_to(const Pose3& other) const {
    return other.inverse() * *this;
  }

  /// Returns the twist that moves from the identity to this pose.
  ///
  /// This is the inverse of Twist3::exp(), and the logarithm map of SE(3). The
  /// twist's angular part is the rotation's to_vector(), whose angle is in
  /// [0, π].
  ///
  /// @return The twist.
  Twist3<Scalar> log() const {
    // With φ = the rotation vector, θ = ‖φ‖, and Φ = the cross product matrix
    // of φ, the linear part is V⁻¹t, where
    //
    //   V⁻¹ = I − Φ/2 + (1 − (θ/2) cot(θ/2))/θ² Φ²
    VariableMatrix<Scalar> φ = m_rotation.to_vector();
    Variable<Scalar> c =
        detail::one_minus_half_theta_cot_half_theta_over_theta_sq(
            Variable<Scalar>{(φ.T() * φ)[0]});
    const VariableMatrix<Scalar> t = m_translation;
    VariableMatrix<Scalar> φ_t = detail::cross(φ, t);

    VariableMatrix<Scalar> linear =
        t - Scalar(0.5) * φ_t + c * detail::cross(φ, φ_t);
    return Twist3<Scalar>{std::move(linear), std::move(φ)};
  }

  /// Returns the 4x4 homogeneous transformation matrix.
  ///
  /// @return The 4x4 homogeneous transformation matrix.
  VariableMatrix<Scalar> to_matrix() const {
    const auto R = m_rotation.to_matrix();
    const auto& t = m_translation;

    return VariableMatrix<Scalar>{{R[0, 0], R[0, 1], R[0, 2], t.x()},
                                  {R[1, 0], R[1, 1], R[1, 2], t.y()},
                                  {R[2, 0], R[2, 1], R[2, 2], t.z()},
                                  {0, 0, 0, 1}};
  }

 private:
  Translation3<Scalar> m_translation;
  Rotation3<Scalar> m_rotation;
};

template <typename Scalar>
Pose3<Scalar> Twist3<Scalar>::exp() const {
  // With θ = ‖ω‖ for the angular part ω, and Ω = the cross product matrix of
  // ω, the translation is Vv, where
  //
  //   V = I + (1 − cos θ)/θ² Ω + (θ − sin θ)/θ³ Ω²
  const auto& v = m_linear;
  const auto& ω = m_angular;
  Variable<Scalar> θ_sq = (ω.T() * ω)[0];
  VariableMatrix<Scalar> ω_v = detail::cross(ω, v);

  return Pose3<Scalar>{
      Translation3<Scalar>{v + detail::one_minus_cos_over_theta_sq(θ_sq) * ω_v +
                           detail::theta_minus_sin_over_theta_cubed(θ_sq) *
                               detail::cross(ω, ω_v)},
      Rotation3<Scalar>{ω}};
}

template <typename Scalar>
Translation3<Scalar> Translation3<Scalar>::transform_by(
    const Pose3<Scalar>& pose) const {
  return rotate_by(pose.rotation()) + pose.translation();
}

}  // namespace slp
