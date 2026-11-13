// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <utility>

#include "sleipnir/autodiff/unit_vector.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

/// A 2D rotation of autodiff variables.
///
/// The rotation by θ is stored as the unit complex number cos θ + i sin θ, so
/// it has no discontinuity at ±π.
///
/// Rotations are created by Problem::decision_variable(so2), which keeps them
/// unit-length during the solve, from constant angles, and by composition and
/// inversion.
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
      : Rotation2{Variable<Scalar>{scalar_cos(angle)},
                  Variable<Scalar>{scalar_sin(angle)}} {}

  /// Constructs a rotation from unit complex number components that are
  /// already unit length.
  ///
  /// This is for internal use. The variables aren't normalized.
  ///
  /// @param cos The real component cos θ.
  /// @param sin The imaginary component sin θ.
  Rotation2(detail::unit_length_t, Variable<Scalar> cos, Variable<Scalar> sin)
      : Rotation2{std::move(cos), std::move(sin)} {}

  /// Constructs a rotation from an angle that depends on other variables.
  ///
  /// This is the inverse of angle() for angles in (−π, π]. It's the exponential
  /// map of SO(2).
  ///
  /// @param angle The rotation angle in radians.
  explicit Rotation2(const Variable<Scalar>& angle)
      : Rotation2{slp::cos(angle), slp::sin(angle)} {}

  /// Returns the real component cos θ.
  ///
  /// @return The real component cos θ.
  const Variable<Scalar>& cos() const { return m_cos; }

  /// Returns the imaginary component sin θ.
  ///
  /// @return The imaginary component sin θ.
  const Variable<Scalar>& sin() const { return m_sin; }

  /// Returns the rotation angle in radians.
  ///
  /// The angle is in (−π, π]. It's discontinuous at ±π, so prefer cos() and
  /// sin() in costs and constraints whose solutions may be near ±π.
  ///
  /// @return The rotation angle in radians.
  Variable<Scalar> angle() const { return atan2(m_sin, m_cos); }

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
  /// Problem::decision_variable(so2).
  ///
  /// @param angle The rotation angle in radians.
  void set_value(Scalar angle) {
    m_cos.set_value(scalar_cos(angle));
    m_sin.set_value(scalar_sin(angle));
  }

  /// Composes two rotations with complex multiplication.
  ///
  /// lhs * rhs rotates by rhs, then by lhs, so the angles add.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The composed rotation.
  friend Rotation2 operator*(const Rotation2& lhs, const Rotation2& rhs) {
    return Rotation2{lhs.m_cos * rhs.m_cos - lhs.m_sin * rhs.m_sin,
                     lhs.m_sin * rhs.m_cos + lhs.m_cos * rhs.m_sin};
  }

  /// Returns the inverse rotation.
  ///
  /// For a unit complex number, this is the conjugate.
  ///
  /// @return The inverse rotation.
  Rotation2 inverse() const { return Rotation2{m_cos, -m_sin}; }

  /// Equality operator that returns an equality constraint for two rotations.
  ///
  /// The constraint is that the angle of lhs relative to rhs is zero. Its only
  /// solution is lhs = rhs, and it's well-conditioned for rotations up to π
  /// apart.
  ///
  /// @param lhs Left-hand side.
  /// @param rhs Right-hand side.
  /// @return The equality constraint.
  friend EqualityConstraints<Scalar> operator==(const Rotation2& lhs,
                                                const Rotation2& rhs) {
    return lhs.relative_to(rhs).angle() == Scalar(0);
  }

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
  VariableMatrix<Scalar> to_matrix() const {
    return VariableMatrix<Scalar>{{m_cos, -m_sin}, {m_sin, m_cos}};
  }

 private:
  Variable<Scalar> m_cos;
  Variable<Scalar> m_sin;

  Rotation2(Variable<Scalar> cos, Variable<Scalar> sin)
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

template <typename Scalar>
VariableMatrix<Scalar> VariableMatrix<Scalar>::rotate_by(
    const Rotation2<Scalar>& z) const {
  slp_assert(rows() == 2 && cols() == 1);

  const auto& v = *this;
  const auto& c = z.cos();
  const auto& s = z.sin();

  return VariableMatrix<Scalar>{{c * v[0] - s * v[1]}, {s * v[0] + c * v[1]}};
}

template <typename Scalar>
UnitVector<Scalar> UnitVector<Scalar>::rotate_by(
    const Rotation2<Scalar>& z) const {
  return UnitVector{m_storage.rotate_by(z)};
}

}  // namespace slp
