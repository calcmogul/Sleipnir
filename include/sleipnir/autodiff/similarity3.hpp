// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sleipnir/autodiff/pose3.hpp"
#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/autodiff/rotation3.hpp"
#include "sleipnir/autodiff/similarity_twist3.hpp"
#include "sleipnir/autodiff/translation3.hpp"
#include "sleipnir/autodiff/trig_series.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

/// A 3D similarity transform of autodiff expressions, which represents a
/// transform in Sim(3).
///
/// A similarity transform is a translation t, a rotation R, and a scale s > 0.
/// It maps a point p in its own frame to s R p + t in the parent frame.
///
/// The scale is stored as its logarithm σ = ln s, so s = exp(σ) is positive
/// for every value of σ.
///
/// Similarity transforms are created by
/// Problem::similarity3_decision_variable(), from constants, and from a
/// translation, a Rotation3, and a log scale.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class Similarity3 {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a constant similarity transform.
  ///
  /// @param translation The translation.
  /// @param rotation The rotation. It's normalized, so it must have nonzero
  ///     norm.
  /// @param scale The scale. Must be positive.
  Similarity3(const Eigen::Vector3<Scalar>& translation,
              const Eigen::Quaternion<Scalar>& rotation, Scalar scale)
      : m_translation{translation}, m_rotation{rotation} {
    slp_assert(scale > Scalar(0));
    using std::log;
    m_log_scale = Real<Scalar>{log(scale)};
  }

  /// Constructs a similarity transform from a translation, a rotation, and a
  /// log scale.
  ///
  /// @param translation The translation.
  /// @param rotation The rotation.
  /// @param log_scale The natural logarithm of the scale.
  Similarity3(Translation3<Scalar> translation, Rotation3<Scalar> rotation,
              Real<Scalar> log_scale)
      : m_translation{std::move(translation)},
        m_rotation{std::move(rotation)},
        m_log_scale{std::move(log_scale)} {}

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

  /// Returns the natural logarithm of the scale.
  ///
  /// @return The natural logarithm of the scale.
  Real<Scalar>& log_scale() { return m_log_scale; }

  /// Returns the natural logarithm of the scale.
  ///
  /// @return The natural logarithm of the scale.
  const Real<Scalar>& log_scale() const { return m_log_scale; }

  /// Returns the scale.
  ///
  /// @return The scale, exp(log_scale()).
  Real<Scalar> scale() const { return exp(m_log_scale); }

  /// Returns the inverse similarity transform.
  ///
  /// It maps points in the parent frame to this transform's frame.
  ///
  /// @return The inverse similarity transform.
  Similarity3 inverse() const {
    Rotation3<Scalar> rotation = m_rotation.inverse();
    Real<Scalar> log_scale = -m_log_scale;
    Translation3<Scalar> translation =
        -m_translation.rotate_by(rotation) * exp(log_scale);
    return Similarity3{std::move(translation), std::move(rotation),
                       std::move(log_scale)};
  }

  /// Returns this similarity transform relative to another.
  ///
  /// The result is this transform expressed in the other transform's frame,
  /// other⁻¹ * this. It's often used as the error between a similarity
  /// transform and a reference.
  ///
  /// @param other The similarity transform whose frame the result is expressed
  ///     in.
  /// @return This similarity transform relative to the other.
  Similarity3 relative_to(const Similarity3& other) const {
    return other.inverse() * *this;
  }

  /// Returns the similarity twist that moves from the identity to this
  /// similarity transform.
  ///
  /// This is the inverse of SimilarityTwist3::exp(), and the logarithm map of
  /// Sim(3). The twist's angular part is the rotation's to_vector(), whose
  /// angle is in [0, π].
  ///
  /// @return The similarity twist.
  SimilarityTwist3<Scalar> log() const {
    // The translation is Wv, where W = A I + BΦ + CΦ² and Φ is the cross
    // product matrix of the rotation vector φ. W scales the axis φ by A and
    // multiplies the plane perpendicular to it by f = A − θ²C + iθB as a
    // complex number, so
    //
    //   W⁻¹ = I/A − B/|f|² Φ + (B² − Re(f) C)/(A|f|²) Φ²
    RealMatrix<Scalar> φ = m_rotation.to_vector();
    Real<Scalar> s = (φ.T() * φ)[0];
    auto [A, B, C] = detail::similarity_coeffs(m_log_scale, s);
    Real<Scalar> re_f = A - s * C;
    Real<Scalar> f_sq = re_f * re_f + s * B * B;
    const RealMatrix<Scalar> t = m_translation;
    RealMatrix<Scalar> φ_t = detail::cross(φ, t);

    RealMatrix<Scalar> linear =
        t / A - B / f_sq * φ_t +
        (B * B - re_f * C) / (A * f_sq) * detail::cross(φ, φ_t);
    return SimilarityTwist3<Scalar>{std::move(linear), std::move(φ),
                                    m_log_scale};
  }

  /// Returns the 4x4 homogeneous transformation matrix.
  ///
  /// @return The 4x4 homogeneous transformation matrix.
  RealMatrix<Scalar> to_matrix() const {
    Real<Scalar> s = scale();
    const RealMatrix<Scalar> sR = s * m_rotation.to_matrix();
    const auto& t = m_translation;

    return RealMatrix<Scalar>{{sR[0, 0], sR[0, 1], sR[0, 2], t.x()},
                              {sR[1, 0], sR[1, 1], sR[1, 2], t.y()},
                              {sR[2, 0], sR[2, 1], sR[2, 2], t.z()},
                              {0, 0, 0, 1}};
  }

 private:
  Translation3<Scalar> m_translation;
  Rotation3<Scalar> m_rotation;
  Real<Scalar> m_log_scale;
};

/// Composes two similarity transforms.
///
/// lhs * rhs transforms by rhs, then by lhs. If rhs is a similarity
/// transform in lhs's frame, the result is that transform in lhs's parent
/// frame.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The composed similarity transform.
template <typename Scalar>
Similarity3<Scalar> operator*(const Similarity3<Scalar>& lhs,
                              const Similarity3<Scalar>& rhs) {
  return Similarity3<Scalar>{rhs.translation().transform_by(lhs),
                             lhs.rotation() * rhs.rotation(),
                             lhs.log_scale() + rhs.log_scale()};
}

/// Equality operator that returns an equality constraint for two similarity
/// transforms.
///
/// The constraint is that the translations are equal, the rotations are
/// equal, and the log scales are equal. See Rotation3's operator==.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraint.
template <typename Scalar>
EqualityConstraints<Scalar> operator==(const Similarity3<Scalar>& lhs,
                                       const Similarity3<Scalar>& rhs) {
  return {lhs.translation() == rhs.translation(),
          lhs.rotation() == rhs.rotation(), lhs.log_scale() == rhs.log_scale()};
}

template <typename Scalar>
Similarity3<Scalar> SimilarityTwist3<Scalar>::exp() const {
  // With θ = ‖ω‖ for the angular part ω, and Ω = the cross product matrix of
  // ω, the translation is Wv, where W = A I + BΩ + CΩ²
  const auto& v = m_linear;
  const auto& ω = m_angular;
  auto [A, B, C] =
      detail::similarity_coeffs(m_log_scale, Real<Scalar>{(ω.T() * ω)[0]});
  RealMatrix<Scalar> ω_v = detail::cross(ω, v);

  return Similarity3<Scalar>{
      Translation3<Scalar>{A * v + B * ω_v + C * detail::cross(ω, ω_v)},
      Rotation3<Scalar>{ω}, m_log_scale};
}

template <typename Scalar>
Translation3<Scalar> Translation3<Scalar>::transform_by(
    const Similarity3<Scalar>& similarity) const {
  return rotate_by(similarity.rotation()) * similarity.scale() +
         similarity.translation();
}

}  // namespace slp
