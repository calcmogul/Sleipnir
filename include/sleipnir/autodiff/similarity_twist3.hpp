// Copyright (c) Sleipnir contributors

#pragma once

#include <type_traits>
#include <utility>

#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/autodiff/real_submatrix.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

template <typename Scalar>
class Similarity3;

/// A 3D similarity twist of autodiff expressions.
///
/// A similarity twist is a change in similarity transform, expressed in the
/// frame of the transform it starts from. It's a linear part (dx, dy, dz),
/// an angular part (rx, ry, rz), a rotation vector in radians, and a log-scale
/// part dσ.
///
/// exp() returns the similarity transform reached by moving along the twist
/// from the identity, and Similarity3::log() is its inverse. A body-frame
/// velocity and log-scale rate times a timestep is a similarity twist, so
/// similarity * (velocity * dt).exp() integrates them for dt.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class SimilarityTwist3 {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a similarity twist from its linear, angular, and log-scale
  /// parts.
  ///
  /// @param linear The linear part (dx, dy, dz). Must be 3x1.
  /// @param angular The angular part (rx, ry, rz) in radians. Must be 3x1.
  /// @param log_scale The log-scale part dσ.
  SimilarityTwist3(RealMatrix<Scalar> linear, RealMatrix<Scalar> angular,
                   Real<Scalar> log_scale)
      : m_linear{std::move(linear)},
        m_angular{std::move(angular)},
        m_log_scale{std::move(log_scale)} {
    slp_assert(m_linear.rows() == 3 && m_linear.cols() == 1);
    slp_assert(m_angular.rows() == 3 && m_angular.cols() == 1);
  }

  /// Returns the linear part (dx, dy, dz).
  ///
  /// @return The linear part.
  RealSubmatrix<RealMatrix<Scalar>> linear() {
    return m_linear.block(0, 0, 3, 1);
  }

  /// Returns the linear part (dx, dy, dz).
  ///
  /// @return The linear part.
  const RealMatrix<Scalar>& linear() const { return m_linear; }

  /// Returns the angular part (rx, ry, rz) in radians.
  ///
  /// @return The angular part.
  RealSubmatrix<RealMatrix<Scalar>> angular() {
    return m_angular.block(0, 0, 3, 1);
  }

  /// Returns the angular part (rx, ry, rz) in radians.
  ///
  /// @return The angular part.
  const RealMatrix<Scalar>& angular() const { return m_angular; }

  /// Returns the log-scale part dσ.
  ///
  /// @return The log-scale part.
  Real<Scalar>& log_scale() { return m_log_scale; }

  /// Returns the log-scale part dσ.
  ///
  /// @return The log-scale part.
  const Real<Scalar>& log_scale() const { return m_log_scale; }

  /// Returns the similarity transform reached by moving along this twist from
  /// the identity.
  ///
  /// This is the exponential map of Sim(3). Its value and derivatives are
  /// accurate for all twists, including those with zero angular or log-scale
  /// parts.
  ///
  /// @return The similarity transform.
  Similarity3<Scalar> exp() const;

 private:
  RealMatrix<Scalar> m_linear;
  RealMatrix<Scalar> m_angular;
  Real<Scalar> m_log_scale;
};

/// Scales a similarity twist.
///
/// @param lhs The similarity twist.
/// @param rhs The factor.
/// @return The scaled similarity twist.
template <typename Scalar>
SimilarityTwist3<Scalar> operator*(
    const SimilarityTwist3<Scalar>& lhs,
    const std::type_identity_t<Real<Scalar>>& rhs) {
  return SimilarityTwist3<Scalar>{lhs.linear() * rhs, lhs.angular() * rhs,
                                  lhs.log_scale() * rhs};
}

/// Scales a similarity twist.
///
/// @param lhs The factor.
/// @param rhs The similarity twist.
/// @return The scaled similarity twist.
template <typename Scalar>
SimilarityTwist3<Scalar> operator*(
    const std::type_identity_t<Real<Scalar>>& lhs,
    const SimilarityTwist3<Scalar>& rhs) {
  return rhs * lhs;
}

/// Adds two similarity twists.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar>
SimilarityTwist3<Scalar> operator+(const SimilarityTwist3<Scalar>& lhs,
                                   const SimilarityTwist3<Scalar>& rhs) {
  return SimilarityTwist3<Scalar>{lhs.linear() + rhs.linear(),
                                  lhs.angular() + rhs.angular(),
                                  lhs.log_scale() + rhs.log_scale()};
}

/// Subtracts two similarity twists.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar>
SimilarityTwist3<Scalar> operator-(const SimilarityTwist3<Scalar>& lhs,
                                   const SimilarityTwist3<Scalar>& rhs) {
  return SimilarityTwist3<Scalar>{lhs.linear() - rhs.linear(),
                                  lhs.angular() - rhs.angular(),
                                  lhs.log_scale() - rhs.log_scale()};
}

/// Negates a similarity twist.
///
/// @param twist The similarity twist.
/// @return The negated similarity twist.
template <typename Scalar>
SimilarityTwist3<Scalar> operator-(const SimilarityTwist3<Scalar>& twist) {
  return SimilarityTwist3<Scalar>{-twist.linear(), -twist.angular(),
                                  -twist.log_scale()};
}

/// Equality operator that returns an equality constraint for two
/// similarity twists.
///
/// The constraint is that the linear parts are equal, the angular parts are
/// equal, and the log-scale parts are equal.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraint.
template <typename Scalar>
EqualityConstraints<Scalar> operator==(const SimilarityTwist3<Scalar>& lhs,
                                       const SimilarityTwist3<Scalar>& rhs) {
  return {lhs.linear() == rhs.linear(), lhs.angular() == rhs.angular(),
          lhs.log_scale() == rhs.log_scale()};
}

}  // namespace slp
