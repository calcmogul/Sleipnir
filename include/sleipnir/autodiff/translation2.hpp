// Copyright (c) Sleipnir contributors

#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/autodiff/rotation2.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

template <typename Scalar>
class Pose2;

template <typename Scalar>
class Similarity2;

/// A 2D translation of autodiff expressions.
///
/// A translation is a point or a displacement (x, y). It converts implicitly
/// to a 2x1 RealMatrix.
///
/// Translations are created by Problem::translation2_decision_variable(),
/// from constants, and from expressions.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class Translation2 {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a translation from its components.
  ///
  /// @param x The x component.
  /// @param y The y component.
  Translation2(Real<Scalar> x, Real<Scalar> y)
      : m_x{std::move(x)}, m_y{std::move(y)} {}

  /// Constructs a translation from a vector.
  ///
  /// @param vector The vector (x, y). Must be 2x1.
  explicit Translation2(const RealMatrix<Scalar>& vector)
      : m_x{vector[0]}, m_y{vector[1]} {
    slp_assert(vector.rows() == 2 && vector.cols() == 1);
  }

  /// Constructs a constant translation.
  ///
  /// @param vector The vector (x, y). Must be 2x1.
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar>
  explicit Translation2(const Eigen::MatrixBase<Derived>& vector)
      : Translation2{Real<Scalar>{vector[0]}, Real<Scalar>{vector[1]}} {
    slp_assert(vector.rows() == 2 && vector.cols() == 1);
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

  /// Returns the translation's value.
  ///
  /// @return The translation's value.
  Eigen::Vector2<Scalar> value() {
    return Eigen::Vector2<Scalar>{m_x.value(), m_y.value()};
  }

  /// Sets the translation's value.
  ///
  /// This is only meaningful for translations of decision variables or
  /// parameters.
  ///
  /// @param value The value.
  void set_value(const Eigen::Vector2<Scalar>& value) {
    m_x.set_value(value[0]);
    m_y.set_value(value[1]);
  }

  /// Implicit conversion to a 2x1 RealMatrix.
  ///
  /// @return The translation as a 2x1 RealMatrix.
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator RealMatrix<Scalar>() const {
    return RealMatrix<Scalar>{{m_x}, {m_y}};
  }

  /// Returns the distance to another translation.
  ///
  /// The distance isn't differentiable where it's zero. For constraints and
  /// costs that can reach zero distance, use squared_distance().
  ///
  /// @param other The other translation.
  /// @return The distance.
  Real<Scalar> distance(const Translation2& other) const {
    return sqrt(squared_distance(other));
  }

  /// Returns the squared distance to another translation.
  ///
  /// @param other The other translation.
  /// @return The squared distance.
  Real<Scalar> squared_distance(const Translation2& other) const {
    Real<Scalar> dx = m_x - other.m_x;
    Real<Scalar> dy = m_y - other.m_y;
    return dx * dx + dy * dy;
  }

  /// Returns this translation rotated by a rotation.
  ///
  /// @param rotation The rotation.
  /// @return The rotated translation.
  Translation2 rotate_by(const Rotation2<Scalar>& rotation) const {
    const auto& c = rotation.cos();
    const auto& s = rotation.sin();
    return Translation2{c * m_x - s * m_y, s * m_x + c * m_y};
  }

  /// Returns this point transformed by a pose.
  ///
  /// The result is R p + t, where p is this point, and R and t are the pose's
  /// rotation and translation.
  ///
  /// @param pose The pose.
  /// @return The transformed point.
  Translation2 transform_by(const Pose2<Scalar>& pose) const;

  /// Returns this point transformed by a similarity transform.
  ///
  /// The result is s R p + t, where p is this point, and s, R, and t are the
  /// similarity transform's scale, rotation, and translation.
  ///
  /// @param similarity The similarity transform.
  /// @return The transformed point.
  Translation2 transform_by(const Similarity2<Scalar>& similarity) const;

 private:
  Real<Scalar> m_x;
  Real<Scalar> m_y;
};

/// Adds two translations.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar>
Translation2<Scalar> operator+(const Translation2<Scalar>& lhs,
                               const Translation2<Scalar>& rhs) {
  return Translation2<Scalar>{lhs.x() + rhs.x(), lhs.y() + rhs.y()};
}

/// Subtracts two translations.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar>
Translation2<Scalar> operator-(const Translation2<Scalar>& lhs,
                               const Translation2<Scalar>& rhs) {
  return Translation2<Scalar>{lhs.x() - rhs.x(), lhs.y() - rhs.y()};
}

/// Negates a translation.
///
/// @param translation The translation.
/// @return The negated translation.
template <typename Scalar>
Translation2<Scalar> operator-(const Translation2<Scalar>& translation) {
  return Translation2<Scalar>{-translation.x(), -translation.y()};
}

/// Scales a translation.
///
/// @param lhs The translation.
/// @param rhs The factor.
/// @return The scaled translation.
template <typename Scalar>
Translation2<Scalar> operator*(const Translation2<Scalar>& lhs,
                               const std::type_identity_t<Real<Scalar>>& rhs) {
  return Translation2<Scalar>{lhs.x() * rhs, lhs.y() * rhs};
}

/// Scales a translation.
///
/// @param lhs The factor.
/// @param rhs The translation.
/// @return The scaled translation.
template <typename Scalar>
Translation2<Scalar> operator*(const std::type_identity_t<Real<Scalar>>& lhs,
                               const Translation2<Scalar>& rhs) {
  return rhs * lhs;
}

/// Divides a translation by a scalar.
///
/// @param lhs The translation.
/// @param rhs The divisor.
/// @return The divided translation.
template <typename Scalar>
Translation2<Scalar> operator/(const Translation2<Scalar>& lhs,
                               const std::type_identity_t<Real<Scalar>>& rhs) {
  return Translation2<Scalar>{lhs.x() / rhs, lhs.y() / rhs};
}

/// Equality operator that returns an equality constraint for two
/// translations.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraint.
template <typename Scalar>
EqualityConstraints<Scalar> operator==(const Translation2<Scalar>& lhs,
                                       const Translation2<Scalar>& rhs) {
  return {lhs.x() == rhs.x(), lhs.y() == rhs.y()};
}

}  // namespace slp
