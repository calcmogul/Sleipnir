// Copyright (c) Sleipnir contributors

#pragma once

#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/autodiff/trig_series.hpp"
#include "sleipnir/autodiff/unit_vector.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

template <typename Scalar>
class Problem;

template <typename Scalar>
class Translation3;

/// A 3D rotation of autodiff expressions.
///
/// The rotation is stored as a unit quaternion with components ordered
/// (w, x, y, z), where w is the scalar part.
///
/// Rotations are created by Problem::rotation3_decision_variable(), which keeps
/// them unit-length during the solve, from constant Eigen quaternions, and by
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
      : Rotation3{
            Real<Scalar>{q.w() / q.norm()}, Real<Scalar>{q.x() / q.norm()},
            Real<Scalar>{q.y() / q.norm()}, Real<Scalar>{q.z() / q.norm()}} {
    slp_assert(q.norm() > Scalar(0));
  }

  /// Constructs a rotation from a rotation vector (axis times angle in
  /// radians).
  ///
  /// This is the inverse of to_vector() for angles in [0, π]. It's the
  /// exponential map of SO(3). Its value and derivatives are accurate for all
  /// rotation vectors, including zero.
  ///
  /// @param rvec The rotation vector. Must be 3x1.
  explicit Rotation3(const RealMatrix<Scalar>& rvec)
      : Rotation3{from_rotation_vector(rvec)} {}

  /// Returns the scalar component w.
  ///
  /// @return The scalar component w.
  const Real<Scalar>& w() const { return m_w; }

  /// Returns the vector component x.
  ///
  /// @return The vector component x.
  const Real<Scalar>& x() const { return m_x; }

  /// Returns the vector component y.
  ///
  /// @return The vector component y.
  const Real<Scalar>& y() const { return m_y; }

  /// Returns the vector component z.
  ///
  /// @return The vector component z.
  const Real<Scalar>& z() const { return m_z; }

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
  /// Problem::rotation3_decision_variable() or Problem::rotation3_parameter().
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

  /// Returns the inverse rotation.
  ///
  /// For a unit quaternion, this is the conjugate.
  ///
  /// @return The inverse rotation.
  Rotation3 inverse() const { return Rotation3{m_w, -m_x, -m_y, -m_z}; }

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
  RealMatrix<Scalar> to_matrix() const {
    const auto& [w, x, y, z] = components();

    return RealMatrix<Scalar>{
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
  RealMatrix<Scalar> to_vector() const {
    // Use the hemisphere w ≥ 0 so the angle is in [0, π]
    Real<Scalar> sign = if_else(m_w >= Scalar(0), Real<Scalar>{Scalar(1)},
                                Real<Scalar>{Scalar(-1)});
    Real<Scalar> w = sign * m_w;
    Real<Scalar> x = sign * m_x;
    Real<Scalar> y = sign * m_y;
    Real<Scalar> z = sign * m_z;

    // For q = (w, u) with n = ‖u‖, the rotation vector is 2 atan2(n, w)/n u
    Real<Scalar> k =
        Scalar(2) * detail::atan2_over_norm(x * x + y * y + z * z, w);

    return RealMatrix<Scalar>{{k * x}, {k * y}, {k * z}};
  }

 private:
  // Problem::rotation3_decision_variable() wraps its decision variables without
  // normalizing them
  friend class Problem<Scalar>;

  // Translation3::rotate_by() and UnitVector::rotate_by() share rotate()
  friend class Translation3<Scalar>;
  friend class UnitVector<Scalar>;

  // The Hamilton product of unit quaternions is unit-length, so it's wrapped
  // without normalizing
  template <typename S>
  friend Rotation3<S> operator*(const Rotation3<S>& lhs,
                                const Rotation3<S>& rhs);

  Real<Scalar> m_w;
  Real<Scalar> m_x;
  Real<Scalar> m_y;
  Real<Scalar> m_z;

  Rotation3(Real<Scalar> w, Real<Scalar> x, Real<Scalar> y, Real<Scalar> z)
      : m_w{std::move(w)},
        m_x{std::move(x)},
        m_y{std::move(y)},
        m_z{std::move(z)} {}

  static Rotation3 from_rotation_vector(const RealMatrix<Scalar>& rvec) {
    slp_assert(rvec.rows() == 3 && rvec.cols() == 1);

    // For θ = ‖φ‖, the quaternion is (cos(θ/2), sin(θ/2)/θ φ). These are
    // functions of (θ/2)² = φᵀφ/4.
    Real<Scalar> half_angle_sq = (rvec.T() * rvec)[0] / Scalar(4);
    Real<Scalar> k = Scalar(0.5) * detail::sin_over_theta(half_angle_sq);
    return Rotation3{detail::cos_of_sqrt(half_angle_sq), k * rvec[0],
                     k * rvec[1], k * rvec[2]};
  }

  std::tuple<const Real<Scalar>&, const Real<Scalar>&, const Real<Scalar>&,
             const Real<Scalar>&>
  components() const {
    return {m_w, m_x, m_y, m_z};
  }

  // Returns the 3D vector v rotated by this rotation.
  RealMatrix<Scalar> rotate(const RealMatrix<Scalar>& v) const {
    slp_assert(v.rows() == 3 && v.cols() == 1);

    const auto& w = m_w;
    const auto& x = m_x;
    const auto& y = m_y;
    const auto& z = m_z;

    // For a unit quaternion q = (w, u),
    //
    //   q ⊗ (0, v) ⊗ q* = v + w t + u × t
    //
    // where t = 2u × v.
    Real<Scalar> t_x = Scalar(2) * (y * v[2] - z * v[1]);
    Real<Scalar> t_y = Scalar(2) * (z * v[0] - x * v[2]);
    Real<Scalar> t_z = Scalar(2) * (x * v[1] - y * v[0]);

    return RealMatrix<Scalar>{{v[0] + w * t_x + (y * t_z - z * t_y)},
                              {v[1] + w * t_y + (z * t_x - x * t_z)},
                              {v[2] + w * t_z + (x * t_y - y * t_x)}};
  }
};

/// Composes two rotations with the Hamilton product.
///
/// lhs * rhs rotates by rhs, then by lhs.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The composed rotation.
template <typename Scalar>
Rotation3<Scalar> operator*(const Rotation3<Scalar>& lhs,
                            const Rotation3<Scalar>& rhs) {
  const auto& [w1, x1, y1, z1] = lhs.components();
  const auto& [w2, x2, y2, z2] = rhs.components();

  return Rotation3<Scalar>{w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                           w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                           w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                           w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2};
}

/// Equality operator that returns an equality constraint for two rotations.
///
/// The constraint is that the rotation vector of lhs relative to rhs is
/// zero. Its only solution is lhs = rhs (as rotations, so q and −q are both
/// solutions), and it's well-conditioned for rotations up to π apart.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraint.
template <typename Scalar>
EqualityConstraints<Scalar> operator==(const Rotation3<Scalar>& lhs,
                                       const Rotation3<Scalar>& rhs) {
  return lhs.relative_to(rhs).to_vector() == Scalar(0);
}

}  // namespace slp
