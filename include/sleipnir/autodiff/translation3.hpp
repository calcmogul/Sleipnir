// Copyright (c) Sleipnir contributors

#pragma once

#include <concepts>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/autodiff/rotation3.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

template <typename Scalar>
class Pose3;

/// A 3D translation of autodiff variables.
///
/// A translation is a point or a displacement (x, y, z). It converts implicitly
/// to a 3x1 VariableMatrix.
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
  Translation3(Variable<Scalar> x, Variable<Scalar> y, Variable<Scalar> z)
      : m_x{std::move(x)}, m_y{std::move(y)}, m_z{std::move(z)} {}

  /// Constructs a translation from a vector.
  ///
  /// @param vector The vector (x, y, z). Must be 3x1.
  explicit Translation3(const VariableMatrix<Scalar>& vector)
      : m_x{vector[0]}, m_y{vector[1]}, m_z{vector[2]} {
    slp_assert(vector.rows() == 3 && vector.cols() == 1);
  }

  /// Constructs a constant translation.
  ///
  /// @param vector The vector (x, y, z). Must be 3x1.
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar>
  explicit Translation3(const Eigen::MatrixBase<Derived>& vector)
      : Translation3{Variable<Scalar>{vector[0]}, Variable<Scalar>{vector[1]},
                     Variable<Scalar>{vector[2]}} {
    slp_assert(vector.rows() == 3 && vector.cols() == 1);
  }

  /// Returns the x component.
  ///
  /// @return The x component.
  Variable<Scalar>& x() { return m_x; }

  /// Returns the x component.
  ///
  /// @return The x component.
  const Variable<Scalar>& x() const { return m_x; }

  /// Returns the y component.
  ///
  /// @return The y component.
  Variable<Scalar>& y() { return m_y; }

  /// Returns the y component.
  ///
  /// @return The y component.
  const Variable<Scalar>& y() const { return m_y; }

  /// Returns the z component.
  ///
  /// @return The z component.
  Variable<Scalar>& z() { return m_z; }

  /// Returns the z component.
  ///
  /// @return The z component.
  const Variable<Scalar>& z() const { return m_z; }

  /// Returns the translation's value.
  ///
  /// @return The translation's value.
  Eigen::Vector3<Scalar> value() {
    return Eigen::Vector3<Scalar>{m_x.value(), m_y.value(), m_z.value()};
  }

  /// Sets the translation's value.
  ///
  /// This is only meaningful for translations of decision variables.
  ///
  /// @param value The value.
  void set_value(const Eigen::Vector3<Scalar>& value) {
    m_x.set_value(value[0]);
    m_y.set_value(value[1]);
    m_z.set_value(value[2]);
  }

  /// Implicit conversion to a 3x1 VariableMatrix.
  ///
  /// @return The translation as a 3x1 VariableMatrix.
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator VariableMatrix<Scalar>() const {
    return VariableMatrix<Scalar>{{m_x}, {m_y}, {m_z}};
  }

  /// Returns the distance to another translation.
  ///
  /// The distance isn't differentiable where it's zero. For constraints and
  /// costs that can reach zero distance, use squared_distance().
  ///
  /// @param other The other translation.
  /// @return The distance.
  Variable<Scalar> distance(const Translation3& other) const {
    return sqrt(squared_distance(other));
  }

  /// Returns the squared distance to another translation.
  ///
  /// @param other The other translation.
  /// @return The squared distance.
  Variable<Scalar> squared_distance(const Translation3& other) const {
    Variable<Scalar> dx = m_x - other.m_x;
    Variable<Scalar> dy = m_y - other.m_y;
    Variable<Scalar> dz = m_z - other.m_z;
    return dx * dx + dy * dy + dz * dz;
  }

  /// Returns this translation rotated by a rotation.
  ///
  /// @param rotation The rotation.
  /// @return The rotated translation.
  Translation3 rotate_by(const Rotation3<Scalar>& rotation) const {
    return Translation3{VariableMatrix<Scalar>{*this}.rotate_by(rotation)};
  }

  /// Returns this point transformed by a pose.
  ///
  /// The result is R p + t, where p is this point, and R and t are the pose's
  /// rotation and translation.
  ///
  /// @param pose The pose.
  /// @return The transformed point.
  Translation3 transform_by(const Pose3<Scalar>& pose) const;

  /// Adds two translations.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The sum.
  friend Translation3 operator+(const Translation3& lhs,
                                const Translation3& rhs) {
    return Translation3{lhs.m_x + rhs.m_x, lhs.m_y + rhs.m_y,
                        lhs.m_z + rhs.m_z};
  }

  /// Subtracts two translations.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The difference.
  friend Translation3 operator-(const Translation3& lhs,
                                const Translation3& rhs) {
    return Translation3{lhs.m_x - rhs.m_x, lhs.m_y - rhs.m_y,
                        lhs.m_z - rhs.m_z};
  }

  /// Negates a translation.
  ///
  /// @param translation The translation.
  /// @return The negated translation.
  friend Translation3 operator-(const Translation3& translation) {
    return Translation3{-translation.m_x, -translation.m_y, -translation.m_z};
  }

  /// Scales a translation.
  ///
  /// @param lhs The translation.
  /// @param rhs The factor.
  /// @return The scaled translation.
  friend Translation3 operator*(const Translation3& lhs,
                                const Variable<Scalar>& rhs) {
    return Translation3{lhs.m_x * rhs, lhs.m_y * rhs, lhs.m_z * rhs};
  }

  /// Scales a translation.
  ///
  /// @param lhs The factor.
  /// @param rhs The translation.
  /// @return The scaled translation.
  friend Translation3 operator*(const Variable<Scalar>& lhs,
                                const Translation3& rhs) {
    return rhs * lhs;
  }

  /// Divides a translation by a scalar.
  ///
  /// @param lhs The translation.
  /// @param rhs The divisor.
  /// @return The divided translation.
  friend Translation3 operator/(const Translation3& lhs,
                                const Variable<Scalar>& rhs) {
    return Translation3{lhs.m_x / rhs, lhs.m_y / rhs, lhs.m_z / rhs};
  }

  /// Equality operator that returns an equality constraint for two
  /// translations.
  ///
  /// @param lhs Left-hand side.
  /// @param rhs Right-hand side.
  /// @return The equality constraint.
  friend EqualityConstraints<Scalar> operator==(const Translation3& lhs,
                                                const Translation3& rhs) {
    return {lhs.m_x == rhs.m_x, lhs.m_y == rhs.m_y, lhs.m_z == rhs.m_z};
  }

 private:
  Variable<Scalar> m_x;
  Variable<Scalar> m_y;
  Variable<Scalar> m_z;
};

}  // namespace slp
