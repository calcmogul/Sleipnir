// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/autodiff/rotation2.hpp"
#include "sleipnir/autodiff/similarity_twist2.hpp"
#include "sleipnir/autodiff/translation2.hpp"
#include "sleipnir/autodiff/trig_series.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

namespace detail {

/// Returns the real and imaginary parts of f(λ) = (e^λ − 1)/λ, where
/// λ = σ + iθ.
///
/// @param σ The log scale.
/// @param θ The rotation angle.
/// @return Re f(λ) and Im f(λ).
template <typename Scalar>
std::pair<Real<Scalar>, Real<Scalar>> similarity2_coeffs(
    const Real<Scalar>& σ, const Real<Scalar>& θ) {
  // With A, B, and C from similarity_coeffs(), f(λ) = A − θ²C + iθB
  Real<Scalar> s = θ * θ;
  auto [A, B, C] = similarity_coeffs(σ, s);
  return {A - s * C, θ * B};
}

}  // namespace detail

/// A 2D similarity transform of autodiff expressions, which represents a
/// transform in Sim(2).
///
/// A similarity transform is a translation t, a rotation R, and a scale s > 0.
/// It maps a point p in its own frame to s R p + t in the parent frame.
///
/// The scale is stored as its logarithm σ = ln s, so s = exp(σ) is positive
/// for every value of σ.
///
/// Similarity transforms are created by
/// Problem::similarity2_decision_variable(), from constants, and from a
/// translation, a Rotation2, and a log scale.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class Similarity2 {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a constant similarity transform.
  ///
  /// @param translation The translation.
  /// @param angle The rotation angle in radians.
  /// @param scale The scale. Must be positive.
  Similarity2(const Eigen::Vector2<Scalar>& translation, Scalar angle,
              Scalar scale)
      : m_translation{translation}, m_rotation{angle} {
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
  Similarity2(Translation2<Scalar> translation, Rotation2<Scalar> rotation,
              Real<Scalar> log_scale)
      : m_translation{std::move(translation)},
        m_rotation{std::move(rotation)},
        m_log_scale{std::move(log_scale)} {}

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
  Similarity2 inverse() const {
    Rotation2<Scalar> rotation = m_rotation.inverse();
    Real<Scalar> log_scale = -m_log_scale;
    Translation2<Scalar> translation =
        -m_translation.rotate_by(rotation) * exp(log_scale);
    return Similarity2{std::move(translation), std::move(rotation),
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
  Similarity2 relative_to(const Similarity2& other) const {
    return other.inverse() * *this;
  }

  /// Returns the similarity twist that moves from the identity to this
  /// similarity transform.
  ///
  /// This is the inverse of SimilarityTwist2::exp(), and the logarithm map of
  /// Sim(2). The twist's angular part is in (−π, π].
  ///
  /// @return The similarity twist.
  SimilarityTwist2<Scalar> log() const {
    // With λ = σ + iθ, the translation is f(λ)v as complex numbers, where
    // f(λ) = (e^λ − 1)/λ = a + ib, so the linear part is v = conj(f)t/|f|²
    Real<Scalar> θ = m_rotation.angle();
    auto [a, b] = detail::similarity2_coeffs(m_log_scale, θ);
    Real<Scalar> f_sq = a * a + b * b;
    const auto& x = m_translation.x();
    const auto& y = m_translation.y();

    return SimilarityTwist2<Scalar>{
        RealMatrix<Scalar>{{(a * x + b * y) / f_sq}, {(a * y - b * x) / f_sq}},
        θ, m_log_scale};
  }

  /// Returns the 3x3 homogeneous transformation matrix.
  ///
  /// @return The 3x3 homogeneous transformation matrix.
  RealMatrix<Scalar> to_matrix() const {
    Real<Scalar> s = scale();
    Real<Scalar> sc = s * m_rotation.cos();
    Real<Scalar> ss = s * m_rotation.sin();
    const auto& x = m_translation.x();
    const auto& y = m_translation.y();

    return RealMatrix<Scalar>{{sc, -ss, x}, {ss, sc, y}, {0, 0, 1}};
  }

 private:
  Translation2<Scalar> m_translation;
  Rotation2<Scalar> m_rotation;
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
Similarity2<Scalar> operator*(const Similarity2<Scalar>& lhs,
                              const Similarity2<Scalar>& rhs) {
  return Similarity2<Scalar>{rhs.translation().transform_by(lhs),
                             lhs.rotation() * rhs.rotation(),
                             lhs.log_scale() + rhs.log_scale()};
}

/// Equality operator that returns an equality constraint for two similarity
/// transforms.
///
/// The constraint is that the translations are equal, the rotations are
/// equal, and the log scales are equal. See Rotation2's operator==.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraint.
template <typename Scalar>
EqualityConstraints<Scalar> operator==(const Similarity2<Scalar>& lhs,
                                       const Similarity2<Scalar>& rhs) {
  return {lhs.translation() == rhs.translation(),
          lhs.rotation() == rhs.rotation(), lhs.log_scale() == rhs.log_scale()};
}

template <typename Scalar>
Similarity2<Scalar> SimilarityTwist2<Scalar>::exp() const {
  // With λ = σ + iθ, the translation is f(λ)v as complex numbers, where
  // f(λ) = (e^λ − 1)/λ = a + ib
  const auto& v = m_linear;
  auto [a, b] = detail::similarity2_coeffs(m_log_scale, m_angular);

  return Similarity2<Scalar>{
      Translation2<Scalar>{a * v[0] - b * v[1], b * v[0] + a * v[1]},
      Rotation2<Scalar>{m_angular}, m_log_scale};
}

template <typename Scalar>
Translation2<Scalar> Translation2<Scalar>::transform_by(
    const Similarity2<Scalar>& similarity) const {
  return rotate_by(similarity.rotation()) * similarity.scale() +
         similarity.translation();
}

}  // namespace slp
