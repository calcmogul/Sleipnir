// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <utility>

#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/autodiff/unit_vector.hpp"

namespace slp {

template <typename Scalar>
class Problem;

/// A 2D rotation of autodiff expressions.
///
/// The rotation by θ is stored as the unit complex number cos θ + i sin θ, so
/// it has no discontinuity at ±π.
///
/// Rotations are created by Problem::rotation2_decision_variable(), which keeps
/// them unit-length during the solve, from constant angles, and by composition
/// and inversion.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class Rotation2 {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a constant rotation.
  ///
  /// @param angle The rotation angle in radians.
  explicit Rotation2(Scalar angle)
      : Rotation2{Real<Scalar>{scalar_cos(angle)},
                  Real<Scalar>{scalar_sin(angle)}} {}

  /// Constructs a rotation from an angle that depends on other variables.
  ///
  /// This is the inverse of angle() for angles in (−π, π]. It's the exponential
  /// map of SO(2).
  ///
  /// @param angle The rotation angle in radians.
  explicit Rotation2(const Real<Scalar>& angle)
      : Rotation2{slp::cos(angle), slp::sin(angle)} {}

  /// Returns the real component cos θ.
  ///
  /// @return The real component cos θ.
  const Real<Scalar>& cos() const { return m_cos; }

  /// Returns the imaginary component sin θ.
  ///
  /// @return The imaginary component sin θ.
  const Real<Scalar>& sin() const { return m_sin; }

  /// Returns the rotation angle in radians.
  ///
  /// The angle is in (−π, π]. It's discontinuous at ±π, so prefer cos() and
  /// sin() in costs and constraints whose solutions may be near ±π.
  ///
  /// @return The rotation angle in radians.
  Real<Scalar> angle() const { return atan2(m_sin, m_cos); }

  /// Returns the rotation angle's value in radians, in (−π, π].
  ///
  /// @return The rotation angle's value in radians.
  Scalar value() {
    using std::atan2;
    return atan2(m_sin.value(), m_cos.value());
  }

  /// Sets the rotation angle.
  ///
  /// This is only meaningful for rotations created by
  /// Problem::rotation2_decision_variable() or Problem::rotation2_parameter().
  ///
  /// @param angle The rotation angle in radians.
  void set_value(Scalar angle) {
    m_cos.set_value(scalar_cos(angle));
    m_sin.set_value(scalar_sin(angle));
  }

  /// Returns the inverse rotation.
  ///
  /// For a unit complex number, this is the conjugate.
  ///
  /// @return The inverse rotation.
  Rotation2 inverse() const { return Rotation2{m_cos, -m_sin}; }

  /// Returns this rotation relative to another rotation.
  ///
  /// The result is the rotation that takes the other rotation to this one,
  /// other⁻¹ * this. Its angle() is the angle from the other rotation to this
  /// one, in (−π, π], so it only wraps when the rotations are opposite.
  ///
  /// @param other The rotation this one is expressed relative to.
  /// @return This rotation relative to the other rotation.
  Rotation2 relative_to(const Rotation2& other) const {
    return other.inverse() * *this;
  }

  /// Returns the 2x2 rotation matrix.
  ///
  /// @return The 2x2 rotation matrix.
  RealMatrix<Scalar> to_matrix() const {
    return RealMatrix<Scalar>{{m_cos, -m_sin}, {m_sin, m_cos}};
  }

 private:
  // Problem::rotation2_decision_variable() wraps its decision variables without
  // normalizing them
  friend class Problem<Scalar>;

  // The product of unit complex numbers is unit-length, so it's wrapped
  // without normalizing
  template <typename S>
  friend Rotation2<S> operator*(const Rotation2<S>& lhs,
                                const Rotation2<S>& rhs);

  Real<Scalar> m_cos;
  Real<Scalar> m_sin;

  Rotation2(Real<Scalar> cos, Real<Scalar> sin)
      : m_cos{std::move(cos)}, m_sin{std::move(sin)} {}

  // The cos() and sin() members hide the free functions inside the class
  static Scalar scalar_cos(Scalar x) {
    using std::cos;
    return cos(x);
  }

  static Scalar scalar_sin(Scalar x) {
    using std::sin;
    return sin(x);
  }
};

/// Composes two rotations with complex multiplication.
///
/// lhs * rhs rotates by rhs, then by lhs, so the angles add.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The composed rotation.
template <typename Scalar>
Rotation2<Scalar> operator*(const Rotation2<Scalar>& lhs,
                            const Rotation2<Scalar>& rhs) {
  return Rotation2<Scalar>{lhs.cos() * rhs.cos() - lhs.sin() * rhs.sin(),
                           lhs.sin() * rhs.cos() + lhs.cos() * rhs.sin()};
}

/// Equality operator that returns an equality constraint for two rotations.
///
/// The constraint is that the angle of lhs relative to rhs is zero. Its only
/// solution is lhs = rhs, and it's well-conditioned for rotations up to π
/// apart.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraint.
template <typename Scalar>
EqualityConstraints<Scalar> operator==(const Rotation2<Scalar>& lhs,
                                       const Rotation2<Scalar>& rhs) {
  return lhs.relative_to(rhs).angle() == Scalar(0);
}

}  // namespace slp
