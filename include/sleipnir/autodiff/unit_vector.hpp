// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"

namespace slp {

namespace detail {

/// Tag type for constructing a unit-length type from values that are already
/// unit length.
struct unit_length_t {
  /// Constructs the tag.
  explicit unit_length_t() = default;
};

/// Tag for constructing a unit-length type from values that are already unit
/// length.
inline constexpr unit_length_t unit_length{};

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
Variable<Scalar> atan2_over_norm(const Variable<Scalar>& n_sq,
                                 const Variable<Scalar>& w) {
  using std::pow;

  const Scalar t_max =
      pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2) / Scalar(17));

  // Both branches are evaluated, and an unselected branch's infinite partial
  // derivatives would turn its zero adjoint into NaN, so each branch gets a
  // safe constant input when it's not selected. The series branch needs
  // w > 0, where t_max w|w| = t_max w².
  constexpr auto less = +[](Scalar a, Scalar b) { return a < b; };
  Variable<Scalar> t_max_w_abs_w = t_max * w * abs(w);
  auto select = [&](const Variable<Scalar>& series_value,
                    const Variable<Scalar>& atan2_value) {
    return if_else(less, n_sq, t_max_w_abs_w, series_value, atan2_value);
  };

  // Series branch
  Variable<Scalar> series_w = select(w, Variable<Scalar>{Scalar(1)});
  Variable<Scalar> t =
      select(n_sq, Variable<Scalar>{Scalar(0)}) / (series_w * series_w);
  Variable<Scalar> series{Scalar(-1) / Scalar(15)};
  for (int m = 6; m >= 0; --m) {
    Scalar coeff = Scalar(m % 2 == 0 ? 1 : -1) / Scalar(2 * m + 1);
    series = coeff + t * series;
  }

  // atan2 branch
  Variable<Scalar> n = sqrt(select(Variable<Scalar>{Scalar(1)}, n_sq));

  return select(series / series_w, atan2(n, w) / n);
}

}  // namespace detail

template <typename Scalar>
class Rotation2;

template <typename Scalar>
class Rotation3;

template <typename Scalar>
class UnitVector;

namespace detail {

/// Returns whether every element of a unit vector is constant.
///
/// @param v The unit vector.
/// @return Whether every element is constant.
template <typename Scalar>
bool is_constant(const UnitVector<Scalar>& v) {
  return std::ranges::all_of(v, [](const Variable<Scalar>& elem) {
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
VariableMatrix<Scalar> sphere_log(const UnitVector<Scalar>& a,
                                  const UnitVector<Scalar>& b) {
  slp_assert(a.rows() == b.rows());

  const VariableMatrix<Scalar> a_vec = a;
  VariableMatrix<Scalar> b_vec = b;

  // With w = a·b and a's component orthogonal to b r = a − wb,
  //
  //   log_b(a) = atan2(‖r‖, w)/‖r‖ r
  Variable<Scalar> w = a.dot(b_vec);
  VariableMatrix<Scalar> r = a_vec - w * b_vec;
  Variable<Scalar> r_sq = (r.T() * r)[0];
  VariableMatrix<Scalar> log = atan2_over_norm(r_sq, w) * r;

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
  VariableMatrix<Scalar> v = b_vec;
  v[p] = v[p] + s;

  // Column j of H dotted with log_b(a) is log_j − 2vⱼ(vᵀlog)/vᵀv
  Variable<Scalar> scale = Scalar(2) * (v.T() * log)[0] / (v.T() * v)[0];
  VariableMatrix<Scalar> result{detail::empty, b.rows() - 1, 1};
  for (int j = 0, row = 0; j < b.rows(); ++j) {
    if (j != p) {
      result[row++] = log[j] - v[j] * scale;
    }
  }
  return result;
}

}  // namespace detail

/// A unit-length column vector of autodiff variables.
///
/// UnitVectors are created by Problem::decision_variable(sphere, n), which
/// keeps them on the unit sphere during the solve, from constant Eigen
/// vectors, and by operations that preserve unit length, like negation and
/// rotate_by(). Operations that don't preserve unit length, like addition and
/// scaling, return a VariableMatrix.
///
/// Elements are read-only. A UnitVector converts implicitly to a
/// VariableMatrix for use in general expressions.
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
      m_storage[row] = Variable<Scalar>{values[row] / norm};
    }
  }

  /// Constructs a unit vector from variables that are already unit length.
  ///
  /// This is for internal use. The variables aren't normalized.
  ///
  /// @param storage The variables. They must form a column vector.
  UnitVector(detail::unit_length_t, VariableMatrix<Scalar> storage)
      : UnitVector{std::move(storage)} {}

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
  const Variable<Scalar>& operator[](int row) const {
    slp_assert(row >= 0 && row < rows());
    return m_storage[row];
  }

  /// Returns an element.
  ///
  /// @param row The row of the element to return.
  /// @param col The column of the element to return, which must be 0.
  /// @return An element.
  const Variable<Scalar>& operator[](int row, [[maybe_unused]] int col) const {
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
  /// Problem::decision_variable(sphere, n).
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

  /// Converts the vector to a VariableMatrix.
  ///
  /// @return The vector as a VariableMatrix.
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator VariableMatrix<Scalar>() const { return m_storage; }

  /// Returns the transpose.
  ///
  /// @return The transpose.
  VariableMatrix<Scalar> T() const { return m_storage.T(); }

  /// Returns the dot product with another vector.
  ///
  /// @param rhs The other vector.
  /// @return The dot product.
  Variable<Scalar> dot(const VariableMatrix<Scalar>& rhs) const {
    slp_assert(rhs.rows() == rows() && rhs.cols() == 1);

    Variable<Scalar> result{Scalar(0)};
    for (int row = 0; row < rows(); ++row) {
      result += m_storage[row] * rhs[row];
    }
    return result;
  }

  /// Returns the cross product with another vector. Both vectors must be 3D.
  ///
  /// @param rhs The other vector.
  /// @return The cross product.
  VariableMatrix<Scalar> cross(const VariableMatrix<Scalar>& rhs) const {
    slp_assert(rows() == 3 && rhs.rows() == 3 && rhs.cols() == 1);

    return VariableMatrix<Scalar>{
        {m_storage[1] * rhs[2] - m_storage[2] * rhs[1]},
        {m_storage[2] * rhs[0] - m_storage[0] * rhs[2]},
        {m_storage[0] * rhs[1] - m_storage[1] * rhs[0]}};
  }

  /// Returns this 3D unit vector rotated by a rotation.
  ///
  /// @param q The rotation.
  /// @return The rotated unit vector.
  UnitVector rotate_by(const Rotation3<Scalar>& q) const;

  /// Returns this 2D unit vector rotated by a rotation.
  ///
  /// @param z The rotation.
  /// @return The rotated unit vector.
  UnitVector rotate_by(const Rotation2<Scalar>& z) const;

  /// Unary minus operator.
  ///
  /// @param v The vector.
  /// @return The negated vector, which is also unit-length.
  friend UnitVector operator-(const UnitVector& v) {
    return UnitVector{-v.m_storage};
  }

  /// Unary plus operator.
  ///
  /// @param v The vector.
  /// @return The vector.
  friend UnitVector operator+(const UnitVector& v) { return v; }

  // VariableMatrix's arithmetic operators are templates that don't consider
  // conversions, so these cover UnitVector operands.

  /// Multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The product.
  template <typename LHS>
    requires(EigenMatrixLike<LHS> || SleipnirMatrixLike<LHS, Scalar> ||
             (ScalarLike<LHS> && !SleipnirType<LHS>))
  friend VariableMatrix<Scalar> operator*(const LHS& lhs,
                                          const UnitVector& rhs) {
    return lhs * rhs.m_storage;
  }

  /// Multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The product.
  template <typename RHS>
    requires(EigenMatrixLike<RHS> || SleipnirMatrixLike<RHS, Scalar> ||
             (ScalarLike<RHS> && !SleipnirType<RHS>))
  friend VariableMatrix<Scalar> operator*(const UnitVector& lhs,
                                          const RHS& rhs) {
    return lhs.m_storage * rhs;
  }

  /// Multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The product.
  friend VariableMatrix<Scalar> operator*(const Variable<Scalar>& lhs,
                                          const UnitVector& rhs) {
    return lhs * rhs.m_storage;
  }

  /// Multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The product.
  friend VariableMatrix<Scalar> operator*(const UnitVector& lhs,
                                          const Variable<Scalar>& rhs) {
    return lhs.m_storage * rhs;
  }

  /// Division operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The quotient.
  template <ScalarLike RHS>
  friend VariableMatrix<Scalar> operator/(const UnitVector& lhs,
                                          const RHS& rhs) {
    return lhs.m_storage / rhs;
  }

  /// Addition operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The sum.
  friend VariableMatrix<Scalar> operator+(const UnitVector& lhs,
                                          const UnitVector& rhs) {
    return lhs.m_storage + rhs.m_storage;
  }

  /// Addition operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The sum.
  template <typename LHS>
    requires(EigenMatrixLike<LHS> || SleipnirMatrixLike<LHS, Scalar>)
  friend VariableMatrix<Scalar> operator+(const LHS& lhs,
                                          const UnitVector& rhs) {
    return lhs + rhs.m_storage;
  }

  /// Addition operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The sum.
  template <typename RHS>
    requires(EigenMatrixLike<RHS> || SleipnirMatrixLike<RHS, Scalar>)
  friend VariableMatrix<Scalar> operator+(const UnitVector& lhs,
                                          const RHS& rhs) {
    return lhs.m_storage + rhs;
  }

  /// Subtraction operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The difference.
  friend VariableMatrix<Scalar> operator-(const UnitVector& lhs,
                                          const UnitVector& rhs) {
    return lhs.m_storage - rhs.m_storage;
  }

  /// Subtraction operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The difference.
  template <typename LHS>
    requires(EigenMatrixLike<LHS> || SleipnirMatrixLike<LHS, Scalar>)
  friend VariableMatrix<Scalar> operator-(const LHS& lhs,
                                          const UnitVector& rhs) {
    return lhs - rhs.m_storage;
  }

  /// Subtraction operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return The difference.
  template <typename RHS>
    requires(EigenMatrixLike<RHS> || SleipnirMatrixLike<RHS, Scalar>)
  friend VariableMatrix<Scalar> operator-(const UnitVector& lhs,
                                          const RHS& rhs) {
    return lhs.m_storage - rhs;
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
  friend EqualityConstraints<Scalar> operator==(const UnitVector& lhs,
                                                const UnitVector& rhs) {
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
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar>
  friend EqualityConstraints<Scalar> operator==(
      const UnitVector& lhs, const Eigen::MatrixBase<Derived>& rhs) {
    return lhs == UnitVector{rhs};
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
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar>
  friend EqualityConstraints<Scalar> operator==(
      const Eigen::MatrixBase<Derived>& lhs, const UnitVector& rhs) {
    return UnitVector{lhs} == rhs;
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
  VariableMatrix<Scalar> m_storage;

  explicit UnitVector(VariableMatrix<Scalar> storage)
      : m_storage{std::move(storage)} {
    slp_assert(m_storage.cols() == 1);
  }
};

}  // namespace slp
