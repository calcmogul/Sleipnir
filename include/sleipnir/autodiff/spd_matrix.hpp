// Copyright (c) Sleipnir contributors

#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>  // IWYU pragma: keep
#include <Eigen/Core>

#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"

namespace slp {

template <typename Scalar>
class Problem;

/// A symmetric positive definite matrix of autodiff expressions.
///
/// SPDMatrices are created by Problem::spd_decision_variable(n), which keeps
/// them positive definite during the solve, from constant Eigen matrices, and
/// by operations that preserve positive definiteness, like addition of two
/// SPDMatrices. Operations that don't, like subtraction and multiplication,
/// return a RealMatrix.
///
/// Elements are read-only, and elements (row, col) and (col, row) are the same
/// variable.
///
/// Equality constraints with an SPDMatrix constrain only the lower triangle,
/// so the other operand must be symmetric. Inequalities aren't supported,
/// since they're ambiguous between elementwise bounds and the semidefinite
/// ordering. Constrain individual elements instead, like `P[i, i] <= c`.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class SPDMatrix {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a constant symmetric positive definite matrix.
  ///
  /// @param values The values. They must be symmetric and positive definite.
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar>
  explicit SPDMatrix(const Eigen::MatrixBase<Derived>& values)
      : m_storage{detail::empty, static_cast<int>(values.rows()),
                  static_cast<int>(values.cols())} {
    slp_assert(values.rows() == values.cols());
    slp_assert(detail::is_symmetric_value<Scalar>(values));
    slp_assert(Eigen::LLT<Eigen::MatrixX<Scalar>>{values}.info() ==
               Eigen::Success);

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col <= row; ++col) {
        Real<Scalar> element{values(row, col)};
        m_storage[row, col] = element;
        m_storage[col, row] = element;
      }
    }
  }

  /// Returns the number of rows.
  ///
  /// @return The number of rows.
  int rows() const { return m_storage.rows(); }

  /// Returns the number of columns.
  ///
  /// @return The number of columns.
  int cols() const { return m_storage.cols(); }

  /// Returns an element.
  ///
  /// @param row The row of the element to return.
  /// @param col The column of the element to return.
  /// @return An element.
  const Real<Scalar>& operator[](int row, int col) const {
    slp_assert(row >= 0 && row < rows() && col >= 0 && col < cols());
    return m_storage[row, col];
  }

  /// Returns an element's value.
  ///
  /// @param row The row of the element whose value to return.
  /// @param col The column of the element whose value to return.
  /// @return An element's value.
  Scalar value(int row, int col) {
    slp_assert(row >= 0 && row < rows() && col >= 0 && col < cols());
    return m_storage.value(row, col);
  }

  /// Returns the matrix's values.
  ///
  /// @return The matrix's values.
  Eigen::MatrixX<Scalar> value() { return m_storage.value(); }

  /// Sets the matrix's values.
  ///
  /// This is only meaningful for SPDMatrices created by
  /// Problem::spd_decision_variable(n) or Problem::spd_parameter(n). Values of
  /// decision variables needn't be positive definite, since the solver
  /// projects them onto the positive definite matrices when the solve starts.
  /// Values of parameters are used as is, so they must be positive definite.
  ///
  /// @param values The values. They must be symmetric to within √ε times their
  ///     largest entry, and their symmetric part is used.
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar>
  void set_value(const Eigen::MatrixBase<Derived>& values) {
    slp_assert(values.rows() == rows() && values.cols() == cols());
    slp_assert(detail::is_symmetric_value<Scalar>(values));

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col <= row; ++col) {
        m_storage[row, col].set_value((values(row, col) + values(col, row)) /
                                      Scalar(2));
      }
    }
  }

  /// Converts the matrix to a RealMatrix.
  ///
  /// @return The matrix as a RealMatrix.
  explicit operator RealMatrix<Scalar>() const { return m_storage; }

  /// Returns the transpose, which is the matrix itself.
  ///
  /// @return The transpose.
  SPDMatrix T() const { return *this; }

  /// Returns the trace, the sum of the diagonal.
  ///
  /// @return The trace.
  Real<Scalar> trace() const { return m_storage.trace(); }

 private:
  // The lower triangle, mirrored so (row, col) and (col, row) alias
  RealMatrix<Scalar> m_storage;

  // Problem::spd_decision_variable() wraps its decision variables without
  // checking their values
  friend class Problem<Scalar>;

  // The sum of SPD matrices is SPD, so it's wrapped without checking
  template <typename S>
  friend SPDMatrix<S> operator+(const SPDMatrix<S>& lhs,
                                const SPDMatrix<S>& rhs);

  explicit SPDMatrix(RealMatrix<Scalar> storage)
      : m_storage{std::move(storage)} {}
};

/// Unary plus operator.
///
/// @param P The matrix.
/// @return The matrix.
template <typename Scalar>
SPDMatrix<Scalar> operator+(const SPDMatrix<Scalar>& P) {
  return P;
}

/// Unary minus operator.
///
/// @param P The matrix.
/// @return The negated matrix, which is negative definite.
template <typename Scalar>
RealMatrix<Scalar> operator-(const SPDMatrix<Scalar>& P) {
  return -RealMatrix<Scalar>{P};
}

/// Addition operator for two symmetric positive definite matrices, whose sum
/// is also symmetric positive definite.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar>
SPDMatrix<Scalar> operator+(const SPDMatrix<Scalar>& lhs,
                            const SPDMatrix<Scalar>& rhs) {
  slp_assert(lhs.rows() == rhs.rows());

  RealMatrix<Scalar> storage{detail::empty, lhs.rows(), lhs.cols()};
  for (int row = 0; row < lhs.rows(); ++row) {
    for (int col = 0; col <= row; ++col) {
      Real<Scalar> element =
          RealMatrix<Scalar>{lhs}[row, col] + RealMatrix<Scalar>{rhs}[row, col];
      storage[row, col] = element;
      storage[col, row] = element;
    }
  }
  return SPDMatrix<Scalar>{std::move(storage)};
}

// RealMatrix's arithmetic operators are templates that don't consider
// conversions, so these cover SPDMatrix operands.

/// Multiplication operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product.
template <typename Scalar, typename LHS>
  requires(EigenMatrixLike<LHS> || SleipnirMatrixLike<LHS, Scalar> ||
           (ScalarLike<LHS> && !SleipnirType<LHS>))
RealMatrix<Scalar> operator*(const LHS& lhs, const SPDMatrix<Scalar>& rhs) {
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
RealMatrix<Scalar> operator*(const SPDMatrix<Scalar>& lhs, const RHS& rhs) {
  return RealMatrix<Scalar>{lhs} * rhs;
}

/// Multiplication operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product.
template <typename Scalar>
RealMatrix<Scalar> operator*(const std::type_identity_t<Real<Scalar>>& lhs,
                             const SPDMatrix<Scalar>& rhs) {
  return lhs * RealMatrix<Scalar>{rhs};
}

/// Multiplication operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product.
template <typename Scalar>
RealMatrix<Scalar> operator*(const SPDMatrix<Scalar>& lhs,
                             const std::type_identity_t<Real<Scalar>>& rhs) {
  return RealMatrix<Scalar>{lhs} * rhs;
}

/// Multiplication operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product, which needn't be symmetric.
template <typename Scalar>
RealMatrix<Scalar> operator*(const SPDMatrix<Scalar>& lhs,
                             const SPDMatrix<Scalar>& rhs) {
  return RealMatrix<Scalar>{lhs} * RealMatrix<Scalar>{rhs};
}

/// Division operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The quotient.
template <typename Scalar, ScalarLike RHS>
RealMatrix<Scalar> operator/(const SPDMatrix<Scalar>& lhs, const RHS& rhs) {
  return RealMatrix<Scalar>{lhs} / rhs;
}

/// Addition operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar, typename LHS>
  requires(EigenMatrixLike<LHS> || SleipnirMatrixLike<LHS, Scalar>)
RealMatrix<Scalar> operator+(const LHS& lhs, const SPDMatrix<Scalar>& rhs) {
  return lhs + RealMatrix<Scalar>{rhs};
}

/// Addition operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar, typename RHS>
  requires(EigenMatrixLike<RHS> || SleipnirMatrixLike<RHS, Scalar>)
RealMatrix<Scalar> operator+(const SPDMatrix<Scalar>& lhs, const RHS& rhs) {
  return RealMatrix<Scalar>{lhs} + rhs;
}

/// Subtraction operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference, which needn't be positive definite.
template <typename Scalar>
RealMatrix<Scalar> operator-(const SPDMatrix<Scalar>& lhs,
                             const SPDMatrix<Scalar>& rhs) {
  return RealMatrix<Scalar>{lhs} - RealMatrix<Scalar>{rhs};
}

/// Subtraction operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar, typename LHS>
  requires(EigenMatrixLike<LHS> || SleipnirMatrixLike<LHS, Scalar>)
RealMatrix<Scalar> operator-(const LHS& lhs, const SPDMatrix<Scalar>& rhs) {
  return lhs - RealMatrix<Scalar>{rhs};
}

/// Subtraction operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar, typename RHS>
  requires(EigenMatrixLike<RHS> || SleipnirMatrixLike<RHS, Scalar>)
RealMatrix<Scalar> operator-(const SPDMatrix<Scalar>& lhs, const RHS& rhs) {
  return RealMatrix<Scalar>{lhs} - rhs;
}

/// Equality operator that returns equality constraints between the lower
/// triangles of a symmetric positive definite matrix and a symmetric
/// operand.
///
/// A symmetric n × n matrix has n(n + 1)/2 degrees of freedom, so it's
/// n(n + 1)/2 equations instead of the n² equations of an elementwise
/// comparison, whose duplicated off-diagonal equations make the constraint
/// Jacobian rank deficient. The upper triangle of rhs isn't read. When
/// assertions are enabled, rhs's current value must be symmetric to within
/// √ε times its largest entry.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side. It must be symmetric.
/// @return The n(n + 1)/2 equality constraints.
template <typename Scalar, typename RHS>
  requires EigenMatrixLike<RHS> || SleipnirMatrixLike<RHS, Scalar> ||
           SymmetricMatrixLike<RHS>
EqualityConstraints<Scalar> operator==(const SPDMatrix<Scalar>& lhs,
                                       const RHS& rhs) {
  slp_assert(detail::is_symmetric_value<Scalar>(rhs));
  return detail::lower_triangle_equality<Scalar>(lhs, rhs, lhs.rows());
}

/// Equality operator that returns equality constraints between the lower
/// triangles of a symmetric operand and a symmetric positive definite
/// matrix.
///
/// See operator==(const SPDMatrix&, const RHS&).
///
/// @param lhs Left-hand side. It must be symmetric.
/// @param rhs Right-hand side.
/// @return The n(n + 1)/2 equality constraints.
template <typename Scalar, typename LHS>
  requires EigenMatrixLike<LHS> || SleipnirMatrixLike<LHS, Scalar> ||
           std::same_as<std::decay_t<LHS>, SymmetricView<Scalar>>
EqualityConstraints<Scalar> operator==(const LHS& lhs,
                                       const SPDMatrix<Scalar>& rhs) {
  slp_assert(detail::is_symmetric_value<Scalar>(lhs));
  return detail::lower_triangle_equality<Scalar>(lhs, rhs, rhs.rows());
}

}  // namespace slp
