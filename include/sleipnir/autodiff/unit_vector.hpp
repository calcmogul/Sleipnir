// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"

namespace slp {

namespace detail {

/// Returns atan2(n, w)/n, where n = √n².
///
/// Near n = 0, atan2(n, w)/n loses precision, and its derivatives are infinite
/// at n = 0. Where w > 0 and n is small relative to w, this uses
///
///   atan2(n, w)/n = 1/w atan(√t)/√t
///                 = 1/w (1 − t/3 + t²/5 − t³/7 + …)
///
/// with t = n²/w², a series in n² that's smooth at n = 0. Truncating it after
/// t⁷ makes its Hessian error grow like n¹⁴, and the atan2 form's Hessian
/// roundoff grows like ε/n³, so they cross at t ≈ ε^(2/17).
///
/// The result is infinite at n = 0 with w < 0.
///
/// @param n_sq n².
/// @param w w.
/// @return atan2(n, w)/n.
template <typename Scalar>
Real<Scalar> atan2_over_norm(const Real<Scalar>& n_sq, const Real<Scalar>& w) {
  using std::pow;

  const Scalar t_max =
      pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2) / Scalar(17));

  // Both branches are evaluated, and an unselected branch's infinite partial
  // derivatives would turn its zero adjoint into NaN, so each branch gets a
  // safe constant input when it's not selected. The series branch needs
  // w > 0, where t_max w|w| = t_max w².
  Real<Scalar> t_max_w_abs_w = t_max * w * abs(w);
  auto select = [&](const Real<Scalar>& series_value,
                    const Real<Scalar>& atan2_value) {
    return if_else(n_sq < t_max_w_abs_w, series_value, atan2_value);
  };

  // Series branch
  Real<Scalar> series_w = select(w, Real<Scalar>{Scalar(1)});
  Real<Scalar> t =
      select(n_sq, Real<Scalar>{Scalar(0)}) / (series_w * series_w);
  Real<Scalar> series{Scalar(-1) / Scalar(15)};
  for (int m = 6; m >= 0; --m) {
    Scalar coeff = Scalar(m % 2 == 0 ? 1 : -1) / Scalar(2 * m + 1);
    series = coeff + t * series;
  }

  // atan2 branch
  Real<Scalar> n = sqrt(select(Real<Scalar>{Scalar(1)}, n_sq));

  return select(series / series_w, atan2(n, w) / n);
}

}  // namespace detail

template <typename Scalar>
class Problem;

template <typename Scalar>
class UnitVector;

namespace detail {

/// Returns whether every element of a unit vector is constant.
///
/// @param v The unit vector.
/// @return Whether every element is constant.
template <typename Scalar>
bool is_constant(const UnitVector<Scalar>& v) {
  return std::ranges::all_of(v, [](const Real<Scalar>& elem) {
    return elem.type() == ExpressionType::CONSTANT;
  });
}

/// Returns the sphere's logarithm map log_b(a) in an orthonormal basis of b's
/// tangent space.
///
/// log_b(a) is the tangent vector at b that points toward a, with length equal
/// to the angle between a and b. Its only zero is a = b.
///
/// The basis comes from a Householder reflection built around b's current
/// value. It's valid unless b later moves to that reflection's one singular
/// point, which is more than 90° from b's current value.
///
/// @param a The unit vector the logarithm points to.
/// @param b The unit vector whose tangent space the logarithm is in.
/// @return The n − 1 components of log_b(a).
template <typename Scalar>
RealMatrix<Scalar> sphere_log(const UnitVector<Scalar>& a,
                              const UnitVector<Scalar>& b) {
  slp_assert(a.rows() == b.rows());

  const RealMatrix<Scalar> a_vec = a;
  RealMatrix<Scalar> b_vec = b;

  // With w = a·b and a's component orthogonal to b r = a − wb,
  //
  //   log_b(a) = atan2(‖r‖, w)/‖r‖ r
  Real<Scalar> w = a.dot(b_vec);
  RealMatrix<Scalar> r = a_vec - w * b_vec;
  Real<Scalar> r_sq = (r.T() * r)[0];
  RealMatrix<Scalar> log = atan2_over_norm(r_sq, w) * r;

  // The Householder reflection H = I − 2vvᵀ/vᵀv with v = b + s eₚ maps b to
  // −s eₚ, so its other columns are an orthonormal basis of b's tangent
  // space. Choosing eₚ as b's largest axis and s as that element's sign keeps
  // vᵀv ≥ 2 near b's current value.
  using std::abs;
  Eigen::VectorX<Scalar> b_value = b_vec.value();
  int p = 0;
  for (int i = 1; i < b.rows(); ++i) {
    if (abs(b_value[i]) > abs(b_value[p])) {
      p = i;
    }
  }
  const Scalar s = b_value[p] >= Scalar(0) ? Scalar(1) : Scalar(-1);
  RealMatrix<Scalar> v = b_vec;
  v[p] = v[p] + s;

  // Column j of H dotted with log_b(a) is log_j − 2vⱼ(vᵀlog)/vᵀv
  Real<Scalar> scale = Scalar(2) * (v.T() * log)[0] / (v.T() * v)[0];
  RealMatrix<Scalar> result{detail::empty, b.rows() - 1, 1};
  for (int j = 0, row = 0; j < b.rows(); ++j) {
    if (j != p) {
      result[row++] = log[j] - v[j] * scale;
    }
  }
  return result;
}

}  // namespace detail

/// A unit-length column vector of autodiff expressions.
///
/// UnitVectors are created by Problem::unit_vector_decision_variable(n), which
/// keeps them on the unit sphere during the solve, from constant Eigen
/// vectors, and by operations that preserve unit length, like negation.
/// Operations that don't preserve unit length, like addition and scaling,
/// return a RealMatrix.
///
/// Elements are read-only. A UnitVector converts implicitly to a
/// RealMatrix for use in general expressions.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class UnitVector {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a constant unit vector from the normalized values.
  ///
  /// @param values The values. They must form a column vector with nonzero
  ///     norm.
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar>
  explicit UnitVector(const Eigen::MatrixBase<Derived>& values)
      : m_storage{detail::empty, static_cast<int>(values.rows()), 1} {
    slp_assert(values.cols() == 1);

    Scalar norm = values.norm();
    slp_assert(norm > Scalar(0));
    for (int row = 0; row < values.rows(); ++row) {
      m_storage[row] = Real<Scalar>{values[row] / norm};
    }
  }

  /// Returns the number of rows.
  ///
  /// @return The number of rows.
  int rows() const { return m_storage.rows(); }

  /// Returns the number of columns, which is always 1.
  ///
  /// @return The number of columns.
  int cols() const { return 1; }

  /// Returns an element.
  ///
  /// @param row The row of the element to return.
  /// @return An element.
  const Real<Scalar>& operator[](int row) const {
    slp_assert(row >= 0 && row < rows());
    return m_storage[row];
  }

  /// Returns an element.
  ///
  /// @param row The row of the element to return.
  /// @param col The column of the element to return, which must be 0.
  /// @return An element.
  const Real<Scalar>& operator[](int row, [[maybe_unused]] int col) const {
    slp_assert(row >= 0 && row < rows() && col == 0);
    return m_storage[row];
  }

  /// Returns an element's value.
  ///
  /// @param row The row of the element whose value to return.
  /// @return An element's value.
  Scalar value(int row) {
    slp_assert(row >= 0 && row < rows());
    return m_storage.value(row);
  }

  /// Returns the vector's values.
  ///
  /// @return The vector's values.
  Eigen::VectorX<Scalar> value() { return m_storage.value(); }

  /// Sets the vector's values to the normalized values.
  ///
  /// This is only meaningful for UnitVectors created by
  /// Problem::unit_vector_decision_variable(n) or
  /// Problem::unit_vector_parameter(n).
  ///
  /// @param values The values. They must have nonzero norm.
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar>
  void set_value(const Eigen::MatrixBase<Derived>& values) {
    slp_assert(values.rows() == rows() && values.cols() == 1);

    Scalar norm = values.norm();
    slp_assert(norm > Scalar(0));
    m_storage.set_value((values / norm).eval());
  }

  /// Converts the vector to a RealMatrix.
  ///
  /// @return The vector as a RealMatrix.
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator RealMatrix<Scalar>() const { return m_storage; }

  /// Returns the transpose.
  ///
  /// @return The transpose.
  RealMatrix<Scalar> T() const { return m_storage.T(); }

  /// Returns the dot product with another vector.
  ///
  /// @param rhs The other vector.
  /// @return The dot product.
  Real<Scalar> dot(const RealMatrix<Scalar>& rhs) const {
    slp_assert(rhs.rows() == rows() && rhs.cols() == 1);

    Real<Scalar> result{Scalar(0)};
    for (int row = 0; row < rows(); ++row) {
      result += m_storage[row] * rhs[row];
    }
    return result;
  }

  /// Returns the cross product with another vector. Both vectors must be 3D.
  ///
  /// @param rhs The other vector.
  /// @return The cross product.
  RealMatrix<Scalar> cross(const RealMatrix<Scalar>& rhs) const {
    slp_assert(rows() == 3 && rhs.rows() == 3 && rhs.cols() == 1);

    return RealMatrix<Scalar>{{m_storage[1] * rhs[2] - m_storage[2] * rhs[1]},
                              {m_storage[2] * rhs[0] - m_storage[0] * rhs[2]},
                              {m_storage[0] * rhs[1] - m_storage[1] * rhs[0]}};
  }

  /// Returns an iterator to the beginning of the vector.
  ///
  /// @return An iterator to the beginning of the vector.
  auto begin() const { return m_storage.cbegin(); }

  /// Returns an iterator to the end of the vector.
  ///
  /// @return An iterator to the end of the vector.
  auto end() const { return m_storage.cend(); }

 private:
  RealMatrix<Scalar> m_storage;

  // Problem::unit_vector_decision_variable() wraps its decision variables
  // without normalizing them
  friend class Problem<Scalar>;

  // The negation of a unit vector is unit-length, so it's wrapped without
  // normalizing
  template <typename S>
  friend UnitVector<S> operator-(const UnitVector<S>& v);

  explicit UnitVector(RealMatrix<Scalar> storage)
      : m_storage{std::move(storage)} {
    slp_assert(m_storage.cols() == 1);
  }
};

/// Unary minus operator.
///
/// @param v The vector.
/// @return The negated vector, which is also unit-length.
template <typename Scalar>
UnitVector<Scalar> operator-(const UnitVector<Scalar>& v) {
  return UnitVector<Scalar>{-RealMatrix<Scalar>{v}};
}

/// Unary plus operator.
///
/// @param v The vector.
/// @return The vector.
template <typename Scalar>
UnitVector<Scalar> operator+(const UnitVector<Scalar>& v) {
  return v;
}

// RealMatrix's arithmetic operators are templates that don't consider
// conversions, so these cover UnitVector operands.

/// Multiplication operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product.
template <typename Scalar, typename LHS>
  requires(EigenMatrixLike<LHS> || SleipnirMatrixLike<LHS, Scalar> ||
           (ScalarLike<LHS> && !SleipnirType<LHS>))
RealMatrix<Scalar> operator*(const LHS& lhs, const UnitVector<Scalar>& rhs) {
  return lhs * RealMatrix<Scalar>{rhs};
}

/// Multiplication operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product.
template <typename Scalar, typename RHS>
  requires(EigenMatrixLike<RHS> || SleipnirMatrixLike<RHS, Scalar> ||
           (ScalarLike<RHS> && !SleipnirType<RHS>))
RealMatrix<Scalar> operator*(const UnitVector<Scalar>& lhs, const RHS& rhs) {
  return RealMatrix<Scalar>{lhs} * rhs;
}

/// Multiplication operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product.
template <typename Scalar>
RealMatrix<Scalar> operator*(const std::type_identity_t<Real<Scalar>>& lhs,
                             const UnitVector<Scalar>& rhs) {
  return lhs * RealMatrix<Scalar>{rhs};
}

/// Multiplication operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product.
template <typename Scalar>
RealMatrix<Scalar> operator*(const UnitVector<Scalar>& lhs,
                             const std::type_identity_t<Real<Scalar>>& rhs) {
  return RealMatrix<Scalar>{lhs} * rhs;
}

/// Division operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The quotient.
template <typename Scalar, ScalarLike RHS>
RealMatrix<Scalar> operator/(const UnitVector<Scalar>& lhs, const RHS& rhs) {
  return RealMatrix<Scalar>{lhs} / rhs;
}

/// Addition operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar>
RealMatrix<Scalar> operator+(const UnitVector<Scalar>& lhs,
                             const UnitVector<Scalar>& rhs) {
  return RealMatrix<Scalar>{lhs} + RealMatrix<Scalar>{rhs};
}

/// Addition operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar, typename LHS>
  requires(EigenMatrixLike<LHS> || SleipnirMatrixLike<LHS, Scalar>)
RealMatrix<Scalar> operator+(const LHS& lhs, const UnitVector<Scalar>& rhs) {
  return lhs + RealMatrix<Scalar>{rhs};
}

/// Addition operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar, typename RHS>
  requires(EigenMatrixLike<RHS> || SleipnirMatrixLike<RHS, Scalar>)
RealMatrix<Scalar> operator+(const UnitVector<Scalar>& lhs, const RHS& rhs) {
  return RealMatrix<Scalar>{lhs} + rhs;
}

/// Subtraction operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar>
RealMatrix<Scalar> operator-(const UnitVector<Scalar>& lhs,
                             const UnitVector<Scalar>& rhs) {
  return RealMatrix<Scalar>{lhs} - RealMatrix<Scalar>{rhs};
}

/// Subtraction operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar, typename LHS>
  requires(EigenMatrixLike<LHS> || SleipnirMatrixLike<LHS, Scalar>)
RealMatrix<Scalar> operator-(const LHS& lhs, const UnitVector<Scalar>& rhs) {
  return lhs - RealMatrix<Scalar>{rhs};
}

/// Subtraction operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar, typename RHS>
  requires(EigenMatrixLike<RHS> || SleipnirMatrixLike<RHS, Scalar>)
RealMatrix<Scalar> operator-(const UnitVector<Scalar>& lhs, const RHS& rhs) {
  return RealMatrix<Scalar>{lhs} - rhs;
}

/// Equality operator that returns an equality constraint for two unit
/// vectors.
///
/// The constraint is that the angle between the vectors is zero. Unit vectors
/// in ℝⁿ have n − 1 degrees of freedom, so it's n − 1 equations instead of
/// the n redundant equations of lhs − rhs = 0. Its only solution is lhs =
/// rhs.
///
/// If both vectors are decision variables, the constraint is built around
/// rhs's value when it's created, so set initial values first.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraint.
template <typename Scalar>
EqualityConstraints<Scalar> operator==(const UnitVector<Scalar>& lhs,
                                       const UnitVector<Scalar>& rhs) {
  // Build the basis around a constant operand, since its value can't change
  if (detail::is_constant(lhs) && !detail::is_constant(rhs)) {
    return detail::sphere_log(rhs, lhs) == Scalar(0);
  } else {
    return detail::sphere_log(lhs, rhs) == Scalar(0);
  }
}

/// Equality operator that returns an equality constraint for a unit vector
/// and a constant direction.
///
/// The direction is normalized. See operator==(const UnitVector&, const
/// UnitVector&).
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side. It must have nonzero norm.
/// @return The equality constraint.
template <typename Scalar, typename Derived>
  requires std::same_as<typename Derived::Scalar, Scalar>
EqualityConstraints<Scalar> operator==(const UnitVector<Scalar>& lhs,
                                       const Eigen::MatrixBase<Derived>& rhs) {
  return lhs == UnitVector<Scalar>{rhs};
}

/// Equality operator that returns an equality constraint for a constant
/// direction and a unit vector.
///
/// The direction is normalized. See operator==(const UnitVector&, const
/// UnitVector&).
///
/// @param lhs Left-hand side. It must have nonzero norm.
/// @param rhs Right-hand side.
/// @return The equality constraint.
template <typename Scalar, typename Derived>
  requires std::same_as<typename Derived::Scalar, Scalar>
EqualityConstraints<Scalar> operator==(const Eigen::MatrixBase<Derived>& lhs,
                                       const UnitVector<Scalar>& rhs) {
  return UnitVector<Scalar>{lhs} == rhs;
}

}  // namespace slp
