// Copyright (c) Sleipnir contributors

#pragma once

#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sleipnir/autodiff/trig_series.hpp"
#include "sleipnir/autodiff/unit_vector.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

/// A 3D rotation of autodiff variables.
///
/// The rotation is stored as a unit quaternion with components ordered
/// (w, x, y, z), where w is the scalar part.
///
/// Rotations are created by Problem::decision_variable(so3), which keeps them
/// unit-length during the solve, from constant Eigen quaternions, and by
/// composition and inversion.
///
/// q and −q represent the same rotation.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class Rotation3 {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a constant rotation from a quaternion, which is normalized.
  ///
  /// @param q The value. It must have nonzero norm.
  explicit Rotation3(const Eigen::Quaternion<Scalar>& q)
      : Rotation3{Variable<Scalar>{q.w() / q.norm()},
                  Variable<Scalar>{q.x() / q.norm()},
                  Variable<Scalar>{q.y() / q.norm()},
                  Variable<Scalar>{q.z() / q.norm()}} {
    slp_assert(q.norm() > Scalar(0));
  }

  /// Constructs a rotation from unit quaternion components that are already
  /// unit length.
  ///
  /// This is for internal use. The variables aren't normalized.
  ///
  /// @param w The scalar component w.
  /// @param x The vector component x.
  /// @param y The vector component y.
  /// @param z The vector component z.
  Rotation3(detail::unit_length_t, Variable<Scalar> w, Variable<Scalar> x,
            Variable<Scalar> y, Variable<Scalar> z)
      : Rotation3{std::move(w), std::move(x), std::move(y), std::move(z)} {}

  /// Constructs a rotation from a rotation vector (axis times angle in
  /// radians).
  ///
  /// This is the inverse of to_vector() for angles in [0, π]. It's the
  /// exponential map of SO(3). Its value and derivatives are accurate for all
  /// rotation vectors, including zero.
  ///
  /// @param rvec The rotation vector. Must be 3x1.
  explicit Rotation3(const VariableMatrix<Scalar>& rvec)
      : Rotation3{from_rotation_vector(rvec)} {}

  /// Returns the scalar component w.
  ///
  /// @return The scalar component w.
  const Variable<Scalar>& w() const { return m_w; }

  /// Returns the vector component x.
  ///
  /// @return The vector component x.
  const Variable<Scalar>& x() const { return m_x; }

  /// Returns the vector component y.
  ///
  /// @return The vector component y.
  const Variable<Scalar>& y() const { return m_y; }

  /// Returns the vector component z.
  ///
  /// @return The vector component z.
  const Variable<Scalar>& z() const { return m_z; }

  /// Returns the rotation's value as a unit quaternion.
  ///
  /// @return The quaternion's value.
  Eigen::Quaternion<Scalar> value() {
    return Eigen::Quaternion<Scalar>{m_w.value(), m_x.value(), m_y.value(),
                                     m_z.value()};
  }

  /// Sets the rotation's value to a quaternion, which is normalized.
  ///
  /// This is only meaningful for rotations created by
  /// Problem::decision_variable(so3).
  ///
  /// @param q The value. It must have nonzero norm.
  void set_value(const Eigen::Quaternion<Scalar>& q) {
    Scalar norm = q.norm();
    slp_assert(norm > Scalar(0));

    m_w.set_value(q.w() / norm);
    m_x.set_value(q.x() / norm);
    m_y.set_value(q.y() / norm);
    m_z.set_value(q.z() / norm);
  }

  /// Composes two rotations with the Hamilton product.
  ///
  /// lhs * rhs rotates by rhs, then by lhs.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The composed rotation.
  friend Rotation3 operator*(const Rotation3& lhs, const Rotation3& rhs) {
    const auto& [w1, x1, y1, z1] = lhs.components();
    const auto& [w2, x2, y2, z2] = rhs.components();

    return Rotation3{w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2};
  }

  /// Returns the inverse rotation.
  ///
  /// For a unit quaternion, this is the conjugate.
  ///
  /// @return The inverse rotation.
  Rotation3 inverse() const { return Rotation3{m_w, -m_x, -m_y, -m_z}; }

  /// Equality operator that returns an equality constraint for two rotations.
  ///
  /// The constraint is that the rotation vector of lhs relative to rhs is
  /// zero. Its only solution is lhs = rhs (as rotations, so q and −q are both
  /// solutions), and it's well-conditioned for rotations up to π apart.
  ///
  /// @param lhs Left-hand side.
  /// @param rhs Right-hand side.
  /// @return The equality constraint.
  friend EqualityConstraints<Scalar> operator==(const Rotation3& lhs,
                                                const Rotation3& rhs) {
    return lhs.relative_to(rhs).to_vector() == Scalar(0);
  }

  /// Returns this rotation relative to another rotation.
  ///
  /// The result is the rotation that takes the other rotation to this one in
  /// the other rotation's frame, other⁻¹ * this. Its to_vector() is the
  /// rotation vector from the other rotation to this one.
  ///
  /// @param other The rotation this one is expressed relative to.
  /// @return This rotation relative to the other rotation.
  Rotation3 relative_to(const Rotation3& other) const {
    return other.inverse() * *this;
  }

  /// Returns the 3x3 rotation matrix.
  ///
  /// @return The 3x3 rotation matrix.
  VariableMatrix<Scalar> to_matrix() const {
    const auto& [w, x, y, z] = components();

    return VariableMatrix<Scalar>{
        {Scalar(1) - Scalar(2) * (y * y + z * z), Scalar(2) * (x * y - w * z),
         Scalar(2) * (x * z + w * y)},
        {Scalar(2) * (x * y + w * z), Scalar(1) - Scalar(2) * (x * x + z * z),
         Scalar(2) * (y * z - w * x)},
        {Scalar(2) * (x * z - w * y), Scalar(2) * (y * z + w * x),
         Scalar(1) - Scalar(2) * (x * x + y * y)}};
  }

  /// Returns the rotation vector (axis times angle in radians).
  ///
  /// The angle is in [0, π]. The rotation vector is discontinuous at π, where
  /// the axis flips, so it's meant for rotations well below π, like residuals
  /// between nearby rotations. Its value, gradient, and Hessian are accurate
  /// near the identity.
  ///
  /// @return The rotation vector.
  VariableMatrix<Scalar> to_vector() const {
    // Use the hemisphere w ≥ 0 so the angle is in [0, π]
    constexpr auto nonnegative = +[](Scalar a, Scalar b) { return a >= b; };
    Variable<Scalar> sign =
        if_else(nonnegative, m_w, Variable<Scalar>{Scalar(0)},
                Variable<Scalar>{Scalar(1)}, Variable<Scalar>{Scalar(-1)});
    Variable<Scalar> w = sign * m_w;
    Variable<Scalar> x = sign * m_x;
    Variable<Scalar> y = sign * m_y;
    Variable<Scalar> z = sign * m_z;

    // For q = (w, u) with n = ‖u‖, the rotation vector is 2 atan2(n, w)/n u
    Variable<Scalar> k =
        Scalar(2) * detail::atan2_over_norm(x * x + y * y + z * z, w);

    return VariableMatrix<Scalar>{{k * x}, {k * y}, {k * z}};
  }

 private:
  Variable<Scalar> m_w;
  Variable<Scalar> m_x;
  Variable<Scalar> m_y;
  Variable<Scalar> m_z;

  Rotation3(Variable<Scalar> w, Variable<Scalar> x, Variable<Scalar> y,
            Variable<Scalar> z)
      : m_w{std::move(w)},
        m_x{std::move(x)},
        m_y{std::move(y)},
        m_z{std::move(z)} {}

  static Rotation3 from_rotation_vector(const VariableMatrix<Scalar>& rvec) {
    slp_assert(rvec.rows() == 3 && rvec.cols() == 1);

    // For θ = ‖φ‖, the quaternion is (cos(θ/2), sin(θ/2)/θ φ). These are
    // functions of (θ/2)² = φᵀφ/4.
    Variable<Scalar> half_angle_sq = (rvec.T() * rvec)[0] / Scalar(4);
    Variable<Scalar> k = Scalar(0.5) * detail::sin_over_theta(half_angle_sq);
    return Rotation3{detail::cos_of_sqrt(half_angle_sq), k * rvec[0],
                     k * rvec[1], k * rvec[2]};
  }

  std::tuple<const Variable<Scalar>&, const Variable<Scalar>&,
             const Variable<Scalar>&, const Variable<Scalar>&>
  components() const {
    return {m_w, m_x, m_y, m_z};
  }
};

template <typename Scalar>
VariableMatrix<Scalar> VariableMatrix<Scalar>::rotate_by(
    const Rotation3<Scalar>& q) const {
  slp_assert(rows() == 3 && cols() == 1);

  const auto& v = *this;
  const auto& w = q.w();
  const auto& x = q.x();
  const auto& y = q.y();
  const auto& z = q.z();

  // For a unit quaternion q = (w, u),
  //
  //   q ⊗ (0, v) ⊗ q* = v + w t + u × t
  //
  // where t = 2u × v.
  Variable<Scalar> t_x = Scalar(2) * (y * v[2] - z * v[1]);
  Variable<Scalar> t_y = Scalar(2) * (z * v[0] - x * v[2]);
  Variable<Scalar> t_z = Scalar(2) * (x * v[1] - y * v[0]);

  return VariableMatrix<Scalar>{{v[0] + w * t_x + (y * t_z - z * t_y)},
                                {v[1] + w * t_y + (z * t_x - x * t_z)},
                                {v[2] + w * t_z + (x * t_y - y * t_x)}};
}

template <typename Scalar>
UnitVector<Scalar> UnitVector<Scalar>::rotate_by(
    const Rotation3<Scalar>& q) const {
  return UnitVector{m_storage.rotate_by(q)};
}

}  // namespace slp
