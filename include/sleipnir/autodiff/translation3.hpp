// Copyright (c) Sleipnir contributors

#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

/// A 3D translation of autodiff expressions.
///
/// A translation is a point or a displacement (x, y, z). It converts implicitly
/// to a 3x1 RealMatrix.
///
/// Translations are created by Problem::translation3_decision_variable(),
/// from constants, and from expressions.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class Translation3 {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a translation from its components.
  ///
  /// @param x The x component.
  /// @param y The y component.
  /// @param z The z component.
  Translation3(Real<Scalar> x, Real<Scalar> y, Real<Scalar> z)
      : m_x{std::move(x)}, m_y{std::move(y)}, m_z{std::move(z)} {}

  /// Constructs a translation from a vector.
  ///
  /// @param vector The vector (x, y, z). Must be 3x1.
  explicit Translation3(const RealMatrix<Scalar>& vector)
      : m_x{vector[0]}, m_y{vector[1]}, m_z{vector[2]} {
    slp_assert(vector.rows() == 3 && vector.cols() == 1);
  }

  /// Constructs a constant translation.
  ///
  /// @param vector The vector (x, y, z). Must be 3x1.
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar>
  explicit Translation3(const Eigen::MatrixBase<Derived>& vector)
      : Translation3{Real<Scalar>{vector[0]}, Real<Scalar>{vector[1]},
                     Real<Scalar>{vector[2]}} {
    slp_assert(vector.rows() == 3 && vector.cols() == 1);
  }

  /// Returns the x component.
  ///
  /// @return The x component.
  Real<Scalar>& x() { return m_x; }

  /// Returns the x component.
  ///
  /// @return The x component.
  const Real<Scalar>& x() const { return m_x; }

  /// Returns the y component.
  ///
  /// @return The y component.
  Real<Scalar>& y() { return m_y; }

  /// Returns the y component.
  ///
  /// @return The y component.
  const Real<Scalar>& y() const { return m_y; }

  /// Returns the z component.
  ///
  /// @return The z component.
  Real<Scalar>& z() { return m_z; }

  /// Returns the z component.
  ///
  /// @return The z component.
  const Real<Scalar>& z() const { return m_z; }

  /// Returns the translation's value.
  ///
  /// @return The translation's value.
  Eigen::Vector3<Scalar> value() {
    return Eigen::Vector3<Scalar>{m_x.value(), m_y.value(), m_z.value()};
  }

  /// Sets the translation's value.
  ///
  /// This is only meaningful for translations of decision variables or
  /// parameters.
  ///
  /// @param value The value.
  void set_value(const Eigen::Vector3<Scalar>& value) {
    m_x.set_value(value[0]);
    m_y.set_value(value[1]);
    m_z.set_value(value[2]);
  }

  /// Implicit conversion to a 3x1 RealMatrix.
  ///
  /// @return The translation as a 3x1 RealMatrix.
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator RealMatrix<Scalar>() const {
    return RealMatrix<Scalar>{{m_x}, {m_y}, {m_z}};
  }

  /// Returns the distance to another translation.
  ///
  /// The distance isn't differentiable where it's zero. For constraints and
  /// costs that can reach zero distance, use squared_distance().
  ///
  /// @param other The other translation.
  /// @return The distance.
  Real<Scalar> distance(const Translation3& other) const {
    return sqrt(squared_distance(other));
  }

  /// Returns the squared distance to another translation.
  ///
  /// @param other The other translation.
  /// @return The squared distance.
  Real<Scalar> squared_distance(const Translation3& other) const {
    Real<Scalar> dx = m_x - other.m_x;
    Real<Scalar> dy = m_y - other.m_y;
    Real<Scalar> dz = m_z - other.m_z;
    return dx * dx + dy * dy + dz * dz;
  }

 private:
  Real<Scalar> m_x;
  Real<Scalar> m_y;
  Real<Scalar> m_z;
};

/// Adds two translations.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar>
Translation3<Scalar> operator+(const Translation3<Scalar>& lhs,
                               const Translation3<Scalar>& rhs) {
  return Translation3<Scalar>{lhs.x() + rhs.x(), lhs.y() + rhs.y(),
                              lhs.z() + rhs.z()};
}

/// Subtracts two translations.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar>
Translation3<Scalar> operator-(const Translation3<Scalar>& lhs,
                               const Translation3<Scalar>& rhs) {
  return Translation3<Scalar>{lhs.x() - rhs.x(), lhs.y() - rhs.y(),
                              lhs.z() - rhs.z()};
}

/// Negates a translation.
///
/// @param translation The translation.
/// @return The negated translation.
template <typename Scalar>
Translation3<Scalar> operator-(const Translation3<Scalar>& translation) {
  return Translation3<Scalar>{-translation.x(), -translation.y(),
                              -translation.z()};
}

/// Scales a translation.
///
/// @param lhs The translation.
/// @param rhs The factor.
/// @return The scaled translation.
template <typename Scalar>
Translation3<Scalar> operator*(const Translation3<Scalar>& lhs,
                               const std::type_identity_t<Real<Scalar>>& rhs) {
  return Translation3<Scalar>{lhs.x() * rhs, lhs.y() * rhs, lhs.z() * rhs};
}

/// Scales a translation.
///
/// @param lhs The factor.
/// @param rhs The translation.
/// @return The scaled translation.
template <typename Scalar>
Translation3<Scalar> operator*(const std::type_identity_t<Real<Scalar>>& lhs,
                               const Translation3<Scalar>& rhs) {
  return rhs * lhs;
}

/// Divides a translation by a scalar.
///
/// @param lhs The translation.
/// @param rhs The divisor.
/// @return The divided translation.
template <typename Scalar>
Translation3<Scalar> operator/(const Translation3<Scalar>& lhs,
                               const std::type_identity_t<Real<Scalar>>& rhs) {
  return Translation3<Scalar>{lhs.x() / rhs, lhs.y() / rhs, lhs.z() / rhs};
}

/// Equality operator that returns an equality constraint for two
/// translations.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraint.
template <typename Scalar>
EqualityConstraints<Scalar> operator==(const Translation3<Scalar>& lhs,
                                       const Translation3<Scalar>& rhs) {
  return {lhs.x() == rhs.x(), lhs.y() == rhs.y(), lhs.z() == rhs.z()};
}

}  // namespace slp
