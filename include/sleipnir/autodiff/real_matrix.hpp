// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>  // IWYU pragma: keep
#include <Eigen/QR>  // IWYU pragma: keep
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_submatrix.hpp"
#include "sleipnir/autodiff/sleipnir_base.hpp"
#include "sleipnir/autodiff/slice.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/function_ref.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

namespace detail {

/// Returns entry (row, col) of a matrix operand, or a scalar operand itself.
///
/// @tparam Scalar Scalar type.
/// @param operand The operand.
/// @param row The entry's row.
/// @param col The entry's column.
/// @return The entry.
template <typename Scalar, typename T>
Real<Scalar> entry(const T& operand, int row, int col) {
  if constexpr (ScalarLike<T>) {
    return Real<Scalar>{operand};
  } else if constexpr (EigenMatrixLike<T>) {
    return Real<Scalar>{Scalar(operand(row, col))};
  } else {
    return operand[row, col];
  }
}

/// Returns true if a square matrix operand's value is symmetric to within
/// √ε times its largest entry, or a scalar operand.
///
/// Values computed from symmetric expressions are symmetric only to rounding,
/// so this catches asymmetric operands, like a transposed product, rather than
/// proving symmetry.
///
/// @tparam Scalar Scalar type.
/// @param operand The operand.
/// @return True if the operand's value is symmetric.
template <typename Scalar, typename T>
bool is_symmetric_value(const T& operand) {
  using std::sqrt;

  if constexpr (ScalarLike<T>) {
    return true;
  } else {
    if (operand.rows() != operand.cols()) {
      return false;
    }

    Eigen::MatrixX<Scalar> M{operand.rows(), operand.cols()};
    for (int row = 0; row < M.rows(); ++row) {
      for (int col = 0; col < M.cols(); ++col) {
        M(row, col) = entry<Scalar>(operand, row, col).value();
      }
    }

    Scalar scale = std::max(Scalar(1), M.cwiseAbs().maxCoeff());
    return (M - M.transpose()).cwiseAbs().maxCoeff() <=
           sqrt(std::numeric_limits<Scalar>::epsilon()) * scale;
  }
}

/// Returns equality constraints between the lower triangles of two n × n
/// operands, or between an n × n operand's lower triangle and a scalar.
///
/// @tparam Scalar Scalar type.
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @param n The operands' dimension.
/// @return The n(n + 1)/2 equality constraints.
template <typename Scalar, typename LHS, typename RHS>
EqualityConstraints<Scalar> lower_triangle_equality(const LHS& lhs,
                                                    const RHS& rhs, int n) {
  if constexpr (!ScalarLike<LHS>) {
    slp_assert(lhs.rows() == n && lhs.cols() == n);
  }
  if constexpr (!ScalarLike<RHS>) {
    slp_assert(rhs.rows() == n && rhs.cols() == n);
  }

  EqualityConstraints<Scalar> result{
      std::initializer_list<EqualityConstraints<Scalar>>{}};
  result.constraints.reserve(n * (n + 1) / 2);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col <= row; ++col) {
      result.constraints.emplace_back(entry<Scalar>(lhs, row, col) -
                                      entry<Scalar>(rhs, row, col));
    }
  }
  return result;
}

}  // namespace detail

/// A square matrix of autodiff expressions that the user asserts is symmetric.
///
/// It reads a square matrix's lower triangle and mirrors it, so its entries are
/// symmetric by construction and the upper triangle is never read. Equality
/// constraints with it constrain only the lower triangle, so a symmetric n × n
/// expression gives n(n + 1)/2 equations instead of the n² equations of an
/// elementwise comparison, whose duplicated off-diagonal equations make the
/// constraint Jacobian rank deficient.
///
/// Inequalities aren't supported, since for symmetric matrices they're
/// ambiguous between elementwise bounds and the semidefinite ordering.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class SymmetricView {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a symmetric view of a square matrix.
  ///
  /// RealMatrix::symmetric_view() is shorthand for this.
  ///
  /// When assertions are enabled, the matrix's current value must be
  /// symmetric to within √ε times its largest entry.
  ///
  /// @param matrix The matrix. Its upper triangle isn't read.
  explicit SymmetricView(const RealMatrix<Scalar>& matrix)
      : m_storage{detail::empty, matrix.rows(), matrix.cols()} {
    slp_assert(matrix.rows() == matrix.cols());
    slp_assert(detail::is_symmetric_value<Scalar>(matrix));

    for (int row = 0; row < matrix.rows(); ++row) {
      for (int col = 0; col <= row; ++col) {
        m_storage[row, col] = matrix[row, col];
        m_storage[col, row] = matrix[row, col];
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

  /// Returns an element. Elements (row, col) and (col, row) are the same
  /// element of the viewed matrix's lower triangle.
  ///
  /// @param row The row of the element to return.
  /// @param col The column of the element to return.
  /// @return An element.
  const Real<Scalar>& operator[](int row, int col) const {
    return m_storage[row, col];
  }

  /// Equality operator that returns equality constraints between the lower
  /// triangles of two symmetric views.
  ///
  /// @param lhs Left-hand side.
  /// @param rhs Right-hand side.
  /// @return The n(n + 1)/2 equality constraints.
  friend EqualityConstraints<Scalar> operator==(const SymmetricView& lhs,
                                                const SymmetricView& rhs) {
    return detail::lower_triangle_equality<Scalar>(lhs, rhs, lhs.rows());
  }

 private:
  // The lower triangle, mirrored so (row, col) and (col, row) alias
  RealMatrix<Scalar> m_storage;
};

/// Equality operator that returns equality constraints between the lower
/// triangles of a symmetric view and another operand.
///
/// The other operand must be symmetric too. If it's a scalar, every entry
/// of the lower triangle is constrained to it.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The n(n + 1)/2 equality constraints.
template <typename Scalar, typename RHS>
  requires ScalarLike<RHS> || EigenMatrixLike<RHS> ||
           SleipnirMatrixLike<RHS, Scalar>
EqualityConstraints<Scalar> operator==(const SymmetricView<Scalar>& lhs,
                                       const RHS& rhs) {
  slp_assert(detail::is_symmetric_value<Scalar>(rhs));
  return detail::lower_triangle_equality<Scalar>(lhs, rhs, lhs.rows());
}

/// Equality operator that returns equality constraints between the lower
/// triangles of another operand and a symmetric view.
///
/// The other operand must be symmetric too. If it's a scalar, every entry
/// of the lower triangle is constrained to it.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The n(n + 1)/2 equality constraints.
template <typename Scalar, typename LHS>
  requires ScalarLike<LHS> || EigenMatrixLike<LHS> ||
           SleipnirMatrixLike<LHS, Scalar>
EqualityConstraints<Scalar> operator==(const LHS& lhs,
                                       const SymmetricView<Scalar>& rhs) {
  slp_assert(detail::is_symmetric_value<Scalar>(lhs));
  return detail::lower_triangle_equality<Scalar>(lhs, rhs, rhs.rows());
}

/// A matrix of autodiff expressions.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class RealMatrix : public SleipnirBase {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs an empty RealMatrix.
  RealMatrix() = default;

  /// Constructs a zero-initialized RealMatrix column vector with the given
  /// rows.
  ///
  /// @param rows The number of matrix rows.
  explicit RealMatrix(int rows) : RealMatrix{rows, 1} {}

  /// Constructs a zero-initialized RealMatrix with the given dimensions.
  ///
  /// @param rows The number of matrix rows.
  /// @param cols The number of matrix columns.
  RealMatrix(int rows, int cols) : m_rows{rows}, m_cols{cols} {
    m_storage.reserve(rows * cols);
    for (int index = 0; index < rows * cols; ++index) {
      m_storage.emplace_back();
    }
  }

  /// Constructs an empty RealMatrix with the given dimensions.
  ///
  /// @param rows The number of matrix rows.
  /// @param cols The number of matrix columns.
  RealMatrix(detail::empty_t, int rows, int cols) : m_rows{rows}, m_cols{cols} {
    m_storage.reserve(rows * cols);
    for (int index = 0; index < rows * cols; ++index) {
      m_storage.emplace_back(nullptr);
    }
  }

  /// Constructs a RealMatrix from a nested list of Reals.
  ///
  /// @param list The nested list of Reals.
  RealMatrix(std::initializer_list<std::initializer_list<Real<Scalar>>> list) {
    // Get row and column counts for destination matrix
    m_rows = list.size();
    m_cols = 0;
    if (list.size() > 0) {
      m_cols = list.begin()->size();
    }

    // Assert all column counts are the same
    for ([[maybe_unused]]
         const auto& row : list) {
      slp_assert(static_cast<int>(row.size()) == m_cols);
    }

    m_storage.reserve(rows() * cols());
    for (const auto& row : list) {
      std::ranges::copy(row, std::back_inserter(m_storage));
    }
  }

  /// Constructs a RealMatrix from a nested list of scalars.
  ///
  /// This overload is for Python bindings only.
  ///
  /// @param list The nested list of Reals.
  // NOLINTNEXTLINE (google-explicit-constructor)
  RealMatrix(const std::vector<std::vector<Scalar>>& list) {
    // Get row and column counts for destination matrix
    m_rows = list.size();
    m_cols = 0;
    if (list.size() > 0) {
      m_cols = list.begin()->size();
    }

    // Assert all column counts are the same
    for ([[maybe_unused]]
         const auto& row : list) {
      slp_assert(static_cast<int>(row.size()) == m_cols);
    }

    m_storage.reserve(rows() * cols());
    for (const auto& row : list) {
      std::ranges::copy(row, std::back_inserter(m_storage));
    }
  }

  /// Constructs a RealMatrix from a nested list of Reals.
  ///
  /// This overload is for Python bindings only.
  ///
  /// @param list The nested list of Reals.
  // NOLINTNEXTLINE (google-explicit-constructor)
  RealMatrix(const std::vector<std::vector<Real<Scalar>>>& list) {
    // Get row and column counts for destination matrix
    m_rows = list.size();
    m_cols = 0;
    if (list.size() > 0) {
      m_cols = list.begin()->size();
    }

    // Assert all column counts are the same
    for ([[maybe_unused]]
         const auto& row : list) {
      slp_assert(static_cast<int>(row.size()) == m_cols);
    }

    m_storage.reserve(rows() * cols());
    for (const auto& row : list) {
      std::ranges::copy(row, std::back_inserter(m_storage));
    }
  }

  /// Constructs a RealMatrix from an Eigen matrix.
  ///
  /// @param values Eigen matrix of values.
  template <typename Derived>
    requires(!ComplexLike<Derived>)
  // NOLINTNEXTLINE (google-explicit-constructor)
  RealMatrix(const Eigen::MatrixBase<Derived>& values)
      : m_rows{static_cast<int>(values.rows())},
        m_cols{static_cast<int>(values.cols())} {
    m_storage.reserve(values.rows() * values.cols());
    for (int row = 0; row < values.rows(); ++row) {
      for (int col = 0; col < values.cols(); ++col) {
        m_storage.emplace_back(values[row, col]);
      }
    }
  }

  /// Constructs a RealMatrix from an Eigen diagonal matrix.
  ///
  /// @param values Diagonal matrix of values.
  template <typename Derived>
  // NOLINTNEXTLINE (google-explicit-constructor)
  RealMatrix(const Eigen::DiagonalBase<Derived>& values)
      : m_rows{static_cast<int>(values.rows())},
        m_cols{static_cast<int>(values.cols())} {
    m_storage.reserve(values.rows() * values.cols());
    for (int row = 0; row < values.rows(); ++row) {
      for (int col = 0; col < values.cols(); ++col) {
        if (row == col) {
          m_storage.emplace_back(values.diagonal()[row]);
        } else {
          m_storage.emplace_back(Scalar(0));
        }
      }
    }
  }

  /// Constructs a scalar RealMatrix from a Real.
  ///
  /// @param variable Real.
  // NOLINTNEXTLINE (google-explicit-constructor)
  RealMatrix(const Real<Scalar>& variable) : m_rows{1}, m_cols{1} {
    m_storage.emplace_back(variable);
  }

  /// Constructs a scalar RealMatrix from a Real.
  ///
  /// @param variable Real.
  // NOLINTNEXTLINE (google-explicit-constructor)
  RealMatrix(Real<Scalar>&& variable) : m_rows{1}, m_cols{1} {
    m_storage.emplace_back(std::move(variable));
  }

  /// Constructs a RealMatrix from a RealSubmatrix.
  ///
  /// @param values RealSubmatrix of values.
  // NOLINTNEXTLINE (google-explicit-constructor)
  RealMatrix(const RealSubmatrix<RealMatrix>& values)
      : m_rows{values.rows()}, m_cols{values.cols()} {
    m_storage.reserve(rows() * cols());
    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        m_storage.emplace_back(values[row, col]);
      }
    }
  }

  /// Constructs a RealMatrix from a RealSubmatrix.
  ///
  /// @param values RealSubmatrix of values.
  // NOLINTNEXTLINE (google-explicit-constructor)
  RealMatrix(const RealSubmatrix<const RealMatrix>& values)
      : m_rows{values.rows()}, m_cols{values.cols()} {
    m_storage.reserve(rows() * cols());
    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        m_storage.emplace_back(values[row, col]);
      }
    }
  }

  /// Constructs a RealMatrix column vector from a Real array.
  ///
  /// @param values Real array of values.
  explicit RealMatrix(std::span<const Real<Scalar>> values)
      : m_rows{static_cast<int>(values.size())}, m_cols{1} {
    m_storage.reserve(rows() * cols());
    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        m_storage.emplace_back(values[row * cols() + col]);
      }
    }
  }

  /// Constructs a RealMatrix from a Real array.
  ///
  /// @param values Real array of values.
  /// @param rows The number of matrix rows.
  /// @param cols The number of matrix columns.
  RealMatrix(std::span<const Real<Scalar>> values, int rows, int cols)
      : m_rows{rows}, m_cols{cols} {
    slp_assert(static_cast<int>(values.size()) == rows * cols);
    m_storage.reserve(rows * cols);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        m_storage.emplace_back(values[row * cols + col]);
      }
    }
  }

  /// Assigns an Eigen matrix to the matrix.
  ///
  /// @param values Eigen matrix of values.
  /// @return This matrix.
  template <typename Derived>
  RealMatrix& operator=(const Eigen::MatrixBase<Derived>& values) {
    slp_assert(rows() == values.rows() && cols() == values.cols());

    for (int row = 0; row < values.rows(); ++row) {
      for (int col = 0; col < values.cols(); ++col) {
        (*this)[row, col] = values[row, col];
      }
    }

    return *this;
  }

  /// Assigns a scalar to the matrix.
  ///
  /// This only works for matrices with one row and one column.
  ///
  /// @param value Value to assign.
  /// @return This matrix.
  RealMatrix& operator=(ScalarLike auto value) {
    slp_assert(rows() == 1 && cols() == 1);

    (*this)[0, 0] = value;

    return *this;
  }

  /// Sets the RealMatrix's internal values.
  ///
  /// @param values Eigen matrix of values.
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar>
  void set_value(const Eigen::MatrixBase<Derived>& values) {
    slp_assert(rows() == values.rows() && cols() == values.cols());

    for (int row = 0; row < values.rows(); ++row) {
      for (int col = 0; col < values.cols(); ++col) {
        (*this)[row, col].set_value(values[row, col]);
      }
    }
  }

  /// Returns the element at the given row and column.
  ///
  /// @param row The row.
  /// @param col The column.
  /// @return The element at the given row and column.
  Real<Scalar>& operator[](int row, int col) {
    slp_assert(row >= 0 && row < rows());
    slp_assert(col >= 0 && col < cols());
    return m_storage[row * cols() + col];
  }

  /// Returns the element at the given row and column.
  ///
  /// @param row The row.
  /// @param col The column.
  /// @return The element at the given row and column.
  const Real<Scalar>& operator[](int row, int col) const {
    slp_assert(row >= 0 && row < rows());
    slp_assert(col >= 0 && col < cols());
    return m_storage[row * cols() + col];
  }

  /// Returns the element at the given index.
  ///
  /// @param index The index.
  /// @return The element at the given index.
  Real<Scalar>& operator[](int index) {
    slp_assert(index >= 0 && index < rows() * cols());
    return m_storage[index];
  }

  /// Returns the element at the given index.
  ///
  /// @param index The index.
  /// @return The element at the given index.
  const Real<Scalar>& operator[](int index) const {
    slp_assert(index >= 0 && index < rows() * cols());
    return m_storage[index];
  }

  /// Returns a block of the matrix.
  ///
  /// @param row_offset The row offset of the block selection.
  /// @param col_offset The column offset of the block selection.
  /// @param block_rows The number of rows in the block selection.
  /// @param block_cols The number of columns in the block selection.
  /// @return A block of the matrix.
  RealSubmatrix<RealMatrix> block(int row_offset, int col_offset,
                                  int block_rows, int block_cols) {
    slp_assert(row_offset >= 0 && row_offset <= rows());
    slp_assert(col_offset >= 0 && col_offset <= cols());
    slp_assert(block_rows >= 0 && block_rows <= rows() - row_offset);
    slp_assert(block_cols >= 0 && block_cols <= cols() - col_offset);
    return RealSubmatrix{*this, row_offset, col_offset, block_rows, block_cols};
  }

  /// Returns a block of the matrix.
  ///
  /// @param row_offset The row offset of the block selection.
  /// @param col_offset The column offset of the block selection.
  /// @param block_rows The number of rows in the block selection.
  /// @param block_cols The number of columns in the block selection.
  /// @return A block of the matrix.
  const RealSubmatrix<const RealMatrix> block(int row_offset, int col_offset,
                                              int block_rows,
                                              int block_cols) const {
    slp_assert(row_offset >= 0 && row_offset <= rows());
    slp_assert(col_offset >= 0 && col_offset <= cols());
    slp_assert(block_rows >= 0 && block_rows <= rows() - row_offset);
    slp_assert(block_cols >= 0 && block_cols <= cols() - col_offset);
    return RealSubmatrix{*this, row_offset, col_offset, block_rows, block_cols};
  }

  /// Returns a slice of the matrix.
  ///
  /// @param row_slice The row slice.
  /// @param col_slice The column slice.
  /// @return A slice of the matrix.
  RealSubmatrix<RealMatrix> operator[](Slice row_slice, Slice col_slice) {
    int row_slice_length = row_slice.adjust(rows());
    int col_slice_length = col_slice.adjust(cols());
    return RealSubmatrix{*this, std::move(row_slice), row_slice_length,
                         std::move(col_slice), col_slice_length};
  }

  /// Returns a slice of the matrix.
  ///
  /// @param row_slice The row slice.
  /// @param col_slice The column slice.
  /// @return A slice of the matrix.
  const RealSubmatrix<const RealMatrix> operator[](Slice row_slice,
                                                   Slice col_slice) const {
    int row_slice_length = row_slice.adjust(rows());
    int col_slice_length = col_slice.adjust(cols());
    return RealSubmatrix{*this, std::move(row_slice), row_slice_length,
                         std::move(col_slice), col_slice_length};
  }

  /// Returns a slice of the matrix.
  ///
  /// The given slices aren't adjusted. This overload is for Python bindings
  /// only.
  ///
  /// @param row_slice The row slice.
  /// @param row_slice_length The row slice length.
  /// @param col_slice The column slice.
  /// @param col_slice_length The column slice length.
  /// @return A slice of the matrix.
  RealSubmatrix<RealMatrix> operator[](Slice row_slice, int row_slice_length,
                                       Slice col_slice, int col_slice_length) {
    return RealSubmatrix{*this, std::move(row_slice), row_slice_length,
                         std::move(col_slice), col_slice_length};
  }

  /// Returns a slice of the matrix.
  ///
  /// The given slices aren't adjusted. This overload is for Python bindings
  /// only.
  ///
  /// @param row_slice The row slice.
  /// @param row_slice_length The row slice length.
  /// @param col_slice The column slice.
  /// @param col_slice_length The column slice length.
  /// @return A slice of the matrix.
  const RealSubmatrix<const RealMatrix> operator[](Slice row_slice,
                                                   int row_slice_length,
                                                   Slice col_slice,
                                                   int col_slice_length) const {
    return RealSubmatrix{*this, std::move(row_slice), row_slice_length,
                         std::move(col_slice), col_slice_length};
  }

  /// Returns a segment of the vector.
  ///
  /// @param offset The offset of the segment.
  /// @param length The length of the segment.
  /// @return A segment of the vector.
  RealSubmatrix<RealMatrix> segment(int offset, int length) {
    slp_assert(cols() == 1);
    slp_assert(offset >= 0 && offset < rows());
    slp_assert(length >= 0 && length <= rows() - offset);
    return block(offset, 0, length, 1);
  }

  /// Returns a segment of the vector.
  ///
  /// @param offset The offset of the segment.
  /// @param length The length of the segment.
  /// @return A segment of the vector.
  const RealSubmatrix<const RealMatrix> segment(int offset, int length) const {
    slp_assert(cols() == 1);
    slp_assert(offset >= 0 && offset < rows());
    slp_assert(length >= 0 && length <= rows() - offset);
    return block(offset, 0, length, 1);
  }

  /// Returns a row slice of the matrix.
  ///
  /// @param row The row to slice.
  /// @return A row slice of the matrix.
  RealSubmatrix<RealMatrix> row(int row) {
    slp_assert(row >= 0 && row < rows());
    return block(row, 0, 1, cols());
  }

  /// Returns a row slice of the matrix.
  ///
  /// @param row The row to slice.
  /// @return A row slice of the matrix.
  const RealSubmatrix<const RealMatrix> row(int row) const {
    slp_assert(row >= 0 && row < rows());
    return block(row, 0, 1, cols());
  }

  /// Returns a column slice of the matrix.
  ///
  /// @param col The column to slice.
  /// @return A column slice of the matrix.
  RealSubmatrix<RealMatrix> col(int col) {
    slp_assert(col >= 0 && col < cols());
    return block(0, col, rows(), 1);
  }

  /// Returns a column slice of the matrix.
  ///
  /// @param col The column to slice.
  /// @return A column slice of the matrix.
  const RealSubmatrix<const RealMatrix> col(int col) const {
    slp_assert(col >= 0 && col < cols());
    return block(0, col, rows(), 1);
  }

  /// Matrix multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  template <EigenMatrixLike LHS, SleipnirMatrixLike<Scalar> RHS>
  friend RealMatrix<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
    slp_assert(lhs.cols() == rhs.rows());

    RealMatrix<Scalar> result(detail::empty, lhs.rows(), rhs.cols());

    for (int i = 0; i < lhs.rows(); ++i) {
      for (int j = 0; j < rhs.cols(); ++j) {
        Real sum{Scalar(0)};
        for (int k = 0; k < lhs.cols(); ++k) {
          sum += lhs[i, k] * rhs[k, j];
        }
        result[i, j] = sum;
      }
    }

    return result;
  }

  /// Matrix multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  template <SleipnirMatrixLike<Scalar> LHS, EigenMatrixLike RHS>
  friend RealMatrix<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
    slp_assert(lhs.cols() == rhs.rows());

    RealMatrix<Scalar> result(detail::empty, lhs.rows(), rhs.cols());

    for (int i = 0; i < lhs.rows(); ++i) {
      for (int j = 0; j < rhs.cols(); ++j) {
        Real sum{Scalar(0)};
        for (int k = 0; k < lhs.cols(); ++k) {
          sum += lhs[i, k] * rhs[k, j];
        }
        result[i, j] = sum;
      }
    }

    return result;
  }

  /// Matrix multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  template <SleipnirMatrixLike<Scalar> LHS, SleipnirMatrixLike<Scalar> RHS>
  friend RealMatrix<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
    slp_assert(lhs.cols() == rhs.rows());

    RealMatrix<Scalar> result(detail::empty, lhs.rows(), rhs.cols());

    for (int i = 0; i < lhs.rows(); ++i) {
      for (int j = 0; j < rhs.cols(); ++j) {
        Real sum{Scalar(0)};
        for (int k = 0; k < lhs.cols(); ++k) {
          sum += lhs[i, k] * rhs[k, j];
        }
        result[i, j] = sum;
      }
    }

    return result;
  }

  /// Sparse matrix multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  template <EigenSparseMatrixLike LHS, SleipnirMatrixLike<Scalar> RHS>
    requires std::same_as<typename std::decay_t<LHS>::Scalar, Scalar>
  friend RealMatrix<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
    slp_assert(lhs.cols() == rhs.rows());

    RealMatrix<Scalar> result(detail::empty, lhs.rows(), rhs.cols());
    for (auto& elem : result) {
      elem = Real{Scalar(0)};
    }

    // Binds without a copy if lhs is already a column-major SparseMatrix
    const Eigen::SparseMatrix<Scalar>& A = lhs;
    for (int outer = 0; outer < A.outerSize(); ++outer) {
      for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it{A, outer}; it;
           ++it) {
        int i = static_cast<int>(it.row());
        int k = static_cast<int>(it.col());
        for (int j = 0; j < rhs.cols(); ++j) {
          result[i, j] += it.value() * rhs[k, j];
        }
      }
    }

    return result;
  }

  /// Sparse matrix multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  template <SleipnirMatrixLike<Scalar> LHS, EigenSparseMatrixLike RHS>
    requires std::same_as<typename std::decay_t<RHS>::Scalar, Scalar>
  friend RealMatrix<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
    slp_assert(lhs.cols() == rhs.rows());

    RealMatrix<Scalar> result(detail::empty, lhs.rows(), rhs.cols());
    for (auto& elem : result) {
      elem = Real{Scalar(0)};
    }

    // Binds without a copy if rhs is already a column-major SparseMatrix
    const Eigen::SparseMatrix<Scalar>& B = rhs;
    for (int outer = 0; outer < B.outerSize(); ++outer) {
      for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it{B, outer}; it;
           ++it) {
        int k = static_cast<int>(it.row());
        int j = static_cast<int>(it.col());
        for (int i = 0; i < lhs.rows(); ++i) {
          result[i, j] += lhs[i, k] * it.value();
        }
      }
    }

    return result;
  }

  /// Matrix-scalar multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  template <SleipnirMatrixLike<Scalar> LHS, ScalarLike RHS>
  friend RealMatrix<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
    RealMatrix<Scalar> result(detail::empty, lhs.rows(), lhs.cols());

    for (int row = 0; row < result.rows(); ++row) {
      for (int col = 0; col < result.cols(); ++col) {
        result[row, col] = lhs[row, col] * rhs;
      }
    }

    return result;
  }

  /// Scalar-matrix multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  template <ScalarLike LHS, SleipnirMatrixLike<Scalar> RHS>
  friend RealMatrix<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
    RealMatrix<Scalar> result(detail::empty, rhs.rows(), rhs.cols());

    for (int row = 0; row < result.rows(); ++row) {
      for (int col = 0; col < result.cols(); ++col) {
        result[row, col] = rhs[row, col] * lhs;
      }
    }

    return result;
  }

  /// Compound matrix multiplication-assignment operator.
  ///
  /// @param rhs Real to multiply.
  /// @return Result of multiplication.
  RealMatrix& operator*=(const MatrixLike auto& rhs) {
    slp_assert(cols() == rhs.rows() && cols() == rhs.cols());

    for (int i = 0; i < rows(); ++i) {
      RealMatrix lhs_old_row = row(i);
      for (int j = 0; j < rhs.cols(); ++j) {
        Real sum{Scalar(0)};
        for (int k = 0; k < cols(); ++k) {
          sum += lhs_old_row[k] * rhs[k, j];
        }
        (*this)[i, j] = sum;
      }
    }

    return *this;
  }

  /// Compound matrix-scalar multiplication-assignment operator.
  ///
  /// @param rhs Real to multiply.
  /// @return Result of multiplication.
  RealMatrix& operator*=(const ScalarLike auto& rhs) {
    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] *= rhs;
      }
    }

    return *this;
  }

  /// Binary division operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of division.
  template <SleipnirMatrixLike<Scalar> LHS, ScalarLike RHS>
    requires(!SleipnirScalarLike<RHS, Scalar>)
  friend RealMatrix<Scalar> operator/(const LHS& lhs, const RHS& rhs) {
    RealMatrix<Scalar> result(detail::empty, lhs.rows(), lhs.cols());

    for (int row = 0; row < result.rows(); ++row) {
      for (int col = 0; col < result.cols(); ++col) {
        result[row, col] = lhs[row, col] / rhs;
      }
    }

    return result;
  }

  /// Binary division operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of division.
  template <SleipnirMatrixLike<Scalar> LHS>
  friend RealMatrix<Scalar> operator/(const LHS& lhs, const Real<Scalar>& rhs) {
    RealMatrix<Scalar> result(detail::empty, lhs.rows(), lhs.cols());

    for (int row = 0; row < result.rows(); ++row) {
      for (int col = 0; col < result.cols(); ++col) {
        result[row, col] = lhs[row, col] / rhs;
      }
    }

    return result;
  }

  /// Compound matrix division-assignment operator.
  ///
  /// @param rhs Real to divide.
  /// @return Result of division.
  RealMatrix& operator/=(const ScalarLike auto& rhs) {
    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] /= rhs;
      }
    }

    return *this;
  }

  /// Binary addition operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of addition.
  template <EigenMatrixLike LHS, SleipnirMatrixLike<Scalar> RHS>
  friend RealMatrix<Scalar> operator+(const LHS& lhs, const RHS& rhs) {
    slp_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());

    RealMatrix<Scalar> result(detail::empty, lhs.rows(), lhs.cols());

    for (int row = 0; row < result.rows(); ++row) {
      for (int col = 0; col < result.cols(); ++col) {
        result[row, col] = lhs[row, col] + rhs[row, col];
      }
    }

    return result;
  }

  /// Binary addition operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of addition.
  template <SleipnirMatrixLike<Scalar> LHS, EigenMatrixLike RHS>
  friend RealMatrix<Scalar> operator+(const LHS& lhs, const RHS& rhs) {
    slp_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());

    RealMatrix<Scalar> result(detail::empty, lhs.rows(), lhs.cols());

    for (int row = 0; row < result.rows(); ++row) {
      for (int col = 0; col < result.cols(); ++col) {
        result[row, col] = lhs[row, col] + rhs[row, col];
      }
    }

    return result;
  }

  /// Binary addition operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of addition.
  template <SleipnirMatrixLike<Scalar> LHS, SleipnirMatrixLike<Scalar> RHS>
  friend RealMatrix<Scalar> operator+(const LHS& lhs, const RHS& rhs) {
    slp_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());

    RealMatrix<Scalar> result(detail::empty, lhs.rows(), lhs.cols());

    for (int row = 0; row < result.rows(); ++row) {
      for (int col = 0; col < result.cols(); ++col) {
        result[row, col] = lhs[row, col] + rhs[row, col];
      }
    }

    return result;
  }

  /// Compound addition-assignment operator.
  ///
  /// @param rhs Real to add.
  /// @return Result of addition.
  RealMatrix& operator+=(const MatrixLike auto& rhs) {
    slp_assert(rows() == rhs.rows() && cols() == rhs.cols());

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] += rhs[row, col];
      }
    }

    return *this;
  }

  /// Compound addition-assignment operator.
  ///
  /// @param rhs Real to add.
  /// @return Result of addition.
  RealMatrix& operator+=(const ScalarLike auto& rhs) {
    slp_assert(rows() == 1 && cols() == 1);

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] += rhs;
      }
    }

    return *this;
  }

  /// Binary subtraction operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of subtraction.
  template <EigenMatrixLike LHS, SleipnirMatrixLike<Scalar> RHS>
  friend RealMatrix<Scalar> operator-(const LHS& lhs, const RHS& rhs) {
    slp_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());

    RealMatrix<Scalar> result(detail::empty, lhs.rows(), lhs.cols());

    for (int row = 0; row < result.rows(); ++row) {
      for (int col = 0; col < result.cols(); ++col) {
        result[row, col] = lhs[row, col] - rhs[row, col];
      }
    }

    return result;
  }

  /// Binary subtraction operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of subtraction.
  template <SleipnirMatrixLike<Scalar> LHS, EigenMatrixLike RHS>
  friend RealMatrix<Scalar> operator-(const LHS& lhs, const RHS& rhs) {
    slp_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());

    RealMatrix<Scalar> result(detail::empty, lhs.rows(), lhs.cols());

    for (int row = 0; row < result.rows(); ++row) {
      for (int col = 0; col < result.cols(); ++col) {
        result[row, col] = lhs[row, col] - rhs[row, col];
      }
    }

    return result;
  }

  /// Binary subtraction operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of subtraction.
  template <SleipnirMatrixLike<Scalar> LHS, SleipnirMatrixLike<Scalar> RHS>
  friend RealMatrix<Scalar> operator-(const LHS& lhs, const RHS& rhs) {
    slp_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());

    RealMatrix<Scalar> result(detail::empty, lhs.rows(), lhs.cols());

    for (int row = 0; row < result.rows(); ++row) {
      for (int col = 0; col < result.cols(); ++col) {
        result[row, col] = lhs[row, col] - rhs[row, col];
      }
    }

    return result;
  }

  /// Compound subtraction-assignment operator.
  ///
  /// @param rhs Real to subtract.
  /// @return Result of subtraction.
  RealMatrix& operator-=(const MatrixLike auto& rhs) {
    slp_assert(rows() == rhs.rows() && cols() == rhs.cols());

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] -= rhs[row, col];
      }
    }

    return *this;
  }

  /// Compound subtraction-assignment operator.
  ///
  /// @param rhs Real to subtract.
  /// @return Result of subtraction.
  RealMatrix& operator-=(const ScalarLike auto& rhs) {
    slp_assert(rows() == 1 && cols() == 1);

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] -= rhs;
      }
    }

    return *this;
  }

  /// Unary minus operator.
  ///
  /// @param lhs Operand for unary minus.
  friend RealMatrix<Scalar> operator-(
      const SleipnirMatrixLike<Scalar> auto& lhs) {
    RealMatrix<Scalar> result{detail::empty, lhs.rows(), lhs.cols()};

    for (int row = 0; row < result.rows(); ++row) {
      for (int col = 0; col < result.cols(); ++col) {
        result[row, col] = -lhs[row, col];
      }
    }

    return result;
  }

  /// Implicit conversion operator from 1x1 RealMatrix to Real.
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator Real<Scalar>() const {
    slp_assert(rows() == 1 && cols() == 1);
    return (*this)[0, 0];
  }

  /// Returns the transpose of the matrix.
  ///
  /// @return The transpose of the matrix.
  RealMatrix<Scalar> T() const {
    RealMatrix<Scalar> result{detail::empty, cols(), rows()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[col, row] = (*this)[row, col];
      }
    }

    return result;
  }

  /// Returns a view of this square matrix that asserts it's symmetric.
  ///
  /// The view reads the lower triangle and mirrors it, so the upper triangle
  /// is never read. Equality constraints with the view constrain only the
  /// lower triangle, which avoids the duplicated off-diagonal equations of an
  /// elementwise comparison. For example, a symmetric residual R is
  /// constrained to zero with `R.symmetric_view() == 0`.
  ///
  /// When assertions are enabled, the matrix's current value must be
  /// symmetric to within √ε times its largest entry.
  ///
  /// @return A symmetric view of this matrix.
  SymmetricView<Scalar> symmetric_view() const {
    return SymmetricView<Scalar>{*this};
  }

  /// Returns the trace of the matrix, the sum of its diagonal.
  ///
  /// @return The trace of the matrix.
  Real<Scalar> trace() const {
    slp_assert(rows() == cols());

    Real<Scalar> result{Scalar(0)};

    for (int row = 0; row < rows(); ++row) {
      result += (*this)[row, row];
    }

    return result;
  }

  /// Returns the number of rows in the matrix.
  ///
  /// @return The number of rows in the matrix.
  int rows() const { return m_rows; }

  /// Returns the number of columns in the matrix.
  ///
  /// @return The number of columns in the matrix.
  int cols() const { return m_cols; }

  /// Returns an element of the matrix.
  ///
  /// @param row The row of the element to return.
  /// @param col The column of the element to return.
  /// @return An element of the matrix.
  Scalar value(int row, int col) { return (*this)[row, col].value(); }

  /// Returns an element of the matrix.
  ///
  /// @param index The index of the element to return.
  /// @return An element of the matrix.
  Scalar value(int index) { return (*this)[index].value(); }

  /// Returns the contents of the matrix.
  ///
  /// @return The contents of the matrix.
  Eigen::MatrixX<Scalar> value() {
    Eigen::MatrixX<Scalar> result{rows(), cols()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[row, col] = value(row, col);
      }
    }

    return result;
  }

  /// Converts the RealMatrix to an Eigen matrix.
  ///
  /// @return Eigen matrix.
  Eigen::MatrixX<Real<Scalar>> to_eigen() const {
    Eigen::MatrixX<Real<Scalar>> result{rows(), cols()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[row, col] = (*this)[row, col];
      }
    }

    return result;
  }

  /// Transforms the matrix coefficient-wise with an unary operator.
  ///
  /// @param unary_op The unary operator to use for the transform operation.
  /// @return Result of the unary operator.
  RealMatrix<Scalar> cwise_transform(
      function_ref<Real<Scalar>(const Real<Scalar>& x)> unary_op) const {
    RealMatrix<Scalar> result{detail::empty, rows(), cols()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[row, col] = unary_op((*this)[row, col]);
      }
    }

    return result;
  }

  /// Returns the matrix exponential.
  ///
  /// @return The matrix exponential.
  RealMatrix<Scalar> exp() const {
    slp_assert(rows() == cols());

    // Coefficients for (13, 13) Padé approximant of exp(A) are from the
    // following program:
    //
    // ```python
    // #!/usr/bin/env python
    //
    // import mpmath as mp
    //
    // # https://en.wikipedia.org/wiki/IEEE_754#Basic_and_interchange_formats
    // mp.mp.prec = 113  # quad precision
    //
    // L = 13
    // M = 13
    // p, q = mp.pade(mp.taylor(mp.exp, 0, L + M), L, M)
    //
    // print("constexpr std::array p{")
    // for k, p_k in enumerate(p):
    //     print(f"Scalar({p_k}L){',' if k < len(p) - 1 else '};'}")
    // print("constexpr std::array q{")
    // for k, q_k in enumerate(q):
    //     print(f"Scalar({q_k}L){',' if k < len(q) - 1 else '};'}")
    // ```
    constexpr size_t NUM_COEFFS = 14;
    constexpr std::array p{Scalar(1.0L),
                           Scalar(0.499999999999999999999987615564159L),
                           Scalar(0.119999999999999999999993719796468L),
                           Scalar(0.0183333333333333333333318078905823L),
                           Scalar(0.00199275362318840579710121408775205L),
                           Scalar(0.00016304347826086956521736560868541L),
                           Scalar(1.03519668737060041407846479512981e-05L),
                           Scalar(5.17598343685300207039205144279119e-07L),
                           Scalar(2.04315135665250081725989398668247e-08L),
                           Scalar(6.30602270571759511499920806570486e-10L),
                           Scalar(1.48377004840414002705850442238441e-11L),
                           Scalar(2.52915349159796595521307855480881e-13L),
                           Scalar(2.81017054621996217245857568431558e-15L),
                           Scalar(1.54404975067030888596595580141692e-17L)};
    constexpr std::array q{Scalar(1.0L),
                           Scalar(-0.500000000000000000000012384435841L),
                           Scalar(0.120000000000000000000006104232309L),
                           Scalar(-0.0183333333333333333333347707904729L),
                           Scalar(0.00199275362318840579710166350137732L),
                           Scalar(-0.000163043478260869565217413851346892L),
                           Scalar(1.03519668737060041407885187519331e-05L),
                           Scalar(-5.17598343685300207039443859549262e-07L),
                           Scalar(2.04315135665250081726103781506842e-08L),
                           Scalar(-6.30602270571759511500345035969254e-10L),
                           Scalar(1.48377004840414002705969744674248e-11L),
                           Scalar(-2.52915349159796595521550648156349e-13L),
                           Scalar(2.81017054621996217246180850207342e-15L),
                           Scalar(-1.54404975067030888596810607146175e-17L)};
    static_assert(p.size() == NUM_COEFFS);
    static_assert(q.size() == NUM_COEFFS);

    //     13
    // P = Σ pₖAᵏ
    //    k=0
    //
    //     13
    // Q = Σ qₖAᵏ
    //    k=0
    RealMatrix<Scalar> P{
        Eigen::VectorX<Scalar>::Constant(rows(), p[0]).asDiagonal()};
    RealMatrix<Scalar> Q{
        Eigen::VectorX<Scalar>::Constant(rows(), q[0]).asDiagonal()};
    auto A_pow = *this;
    size_t k = 1;
    while (true) {
      P += p[k] * A_pow;
      Q += q[k] * A_pow;

      ++k;
      if (k < NUM_COEFFS) {
        A_pow *= *this;
      } else {
        break;
      }
    }

    // https://mpmath.org/doc/current/calculus/approximation.html#mpmath.pade
    // defines the Padé approximant as exp(A)Q ≈ P, so:
    //
    //   exp(A) ≈ P / Q
    //   exp(A) ≈ (Qᵀ \ Pᵀ)ᵀ
    return RealMatrix<Scalar>{
        Q.T().to_eigen().lu().solve(P.T().to_eigen()).transpose()};
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS

  class iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = Real<Scalar>;
    using difference_type = std::ptrdiff_t;
    using pointer = Real<Scalar>*;
    using reference = Real<Scalar>&;

    constexpr iterator() noexcept = default;

    explicit constexpr iterator(
        gch::small_vector<Real<Scalar>>::iterator it) noexcept
        : m_it{it} {}

    constexpr iterator& operator++() noexcept {
      ++m_it;
      return *this;
    }

    constexpr iterator operator++(int) noexcept {
      iterator retval = *this;
      ++(*this);
      return retval;
    }

    constexpr iterator& operator--() noexcept {
      --m_it;
      return *this;
    }

    constexpr iterator operator--(int) noexcept {
      iterator retval = *this;
      --(*this);
      return retval;
    }

    constexpr bool operator==(const iterator&) const noexcept = default;

    constexpr reference operator*() const noexcept { return *m_it; }

   private:
    gch::small_vector<Real<Scalar>>::iterator m_it;
  };

  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = Real<Scalar>;
    using difference_type = std::ptrdiff_t;
    using pointer = Real<Scalar>*;
    using const_reference = const Real<Scalar>&;

    constexpr const_iterator() noexcept = default;

    explicit constexpr const_iterator(
        gch::small_vector<Real<Scalar>>::const_iterator it) noexcept
        : m_it{it} {}

    constexpr const_iterator& operator++() noexcept {
      ++m_it;
      return *this;
    }

    constexpr const_iterator operator++(int) noexcept {
      const_iterator retval = *this;
      ++(*this);
      return retval;
    }

    constexpr const_iterator& operator--() noexcept {
      --m_it;
      return *this;
    }

    constexpr const_iterator operator--(int) noexcept {
      const_iterator retval = *this;
      --(*this);
      return retval;
    }

    constexpr bool operator==(const const_iterator&) const noexcept = default;

    constexpr const_reference operator*() const noexcept { return *m_it; }

   private:
    gch::small_vector<Real<Scalar>>::const_iterator m_it;
  };

  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

#endif  // DOXYGEN_SHOULD_SKIP_THIS

  /// Returns begin iterator.
  ///
  /// @return Begin iterator.
  iterator begin() { return iterator{m_storage.begin()}; }

  /// Returns end iterator.
  ///
  /// @return End iterator.
  iterator end() { return iterator{m_storage.end()}; }

  /// Returns const begin iterator.
  ///
  /// @return Const begin iterator.
  const_iterator begin() const { return const_iterator{m_storage.begin()}; }

  /// Returns const end iterator.
  ///
  /// @return Const end iterator.
  const_iterator end() const { return const_iterator{m_storage.end()}; }

  /// Returns const begin iterator.
  ///
  /// @return Const begin iterator.
  const_iterator cbegin() const { return const_iterator{m_storage.cbegin()}; }

  /// Returns const end iterator.
  ///
  /// @return Const end iterator.
  const_iterator cend() const { return const_iterator{m_storage.cend()}; }

  /// Returns reverse begin iterator.
  ///
  /// @return Reverse begin iterator.
  reverse_iterator rbegin() { return reverse_iterator{end()}; }

  /// Returns reverse end iterator.
  ///
  /// @return Reverse end iterator.
  reverse_iterator rend() { return reverse_iterator{begin()}; }

  /// Returns const reverse begin iterator.
  ///
  /// @return Const reverse begin iterator.
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator{end()};
  }

  /// Returns const reverse end iterator.
  ///
  /// @return Const reverse end iterator.
  const_reverse_iterator rend() const {
    return const_reverse_iterator{begin()};
  }

  /// Returns const reverse begin iterator.
  ///
  /// @return Const reverse begin iterator.
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator{cend()};
  }

  /// Returns const reverse end iterator.
  ///
  /// @return Const reverse end iterator.
  const_reverse_iterator crend() const {
    return const_reverse_iterator{cbegin()};
  }

  /// Returns number of elements in matrix.
  ///
  /// @return Number of elements in matrix.
  size_t size() const { return m_storage.size(); }

  /// Returns an identity matrix.
  ///
  /// @param rows The number of matrix rows.
  /// @return An identity matrix.
  static RealMatrix<Scalar> identity(int rows) {
    RealMatrix<Scalar> result{detail::empty, rows, rows};

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < rows; ++col) {
        result[row, col] = row == col ? Scalar(1) : Scalar(0);
      }
    }

    return result;
  }

  /// Returns a matrix filled with zeros.
  ///
  /// @param rows The number of matrix rows.
  /// @param cols The number of matrix columns.
  /// @return A matrix filled with zeros.
  static RealMatrix<Scalar> zero(int rows, int cols) {
    RealMatrix<Scalar> result{detail::empty, rows, cols};

    for (auto& elem : result) {
      elem = Scalar(0);
    }

    return result;
  }

  /// Returns a matrix filled with ones.
  ///
  /// @param rows The number of matrix rows.
  /// @param cols The number of matrix columns.
  /// @return A matrix filled with ones.
  static RealMatrix<Scalar> one(int rows, int cols) {
    RealMatrix<Scalar> result{detail::empty, rows, cols};

    for (auto& elem : result) {
      elem = Scalar(1);
    }

    return result;
  }

  /// Returns a matrix filled with a constant.
  ///
  /// @param rows The number of matrix rows.
  /// @param cols The number of matrix columns.
  /// @param constant The constant.
  /// @return A matrix filled with a constant.
  static RealMatrix<Scalar> constant(int rows, int cols, Scalar constant) {
    RealMatrix<Scalar> result{detail::empty, rows, cols};

    for (auto& elem : result) {
      elem = constant;
    }

    return result;
  }

 private:
  gch::small_vector<Real<Scalar>> m_storage;
  int m_rows = 0;
  int m_cols = 0;
};

template <typename Derived>
RealMatrix(const Eigen::MatrixBase<Derived>&)
    -> RealMatrix<typename Derived::Scalar>;

template <typename Derived>
RealMatrix(const Eigen::DiagonalBase<Derived>&)
    -> RealMatrix<typename Derived::Scalar>;

/// Matrix-scalar multiplication operator.
///
/// @tparam Scalar Scalar type.
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
template <typename Scalar, EigenMatrixLike LHS>
  requires std::same_as<typename std::decay_t<LHS>::Scalar, Scalar>
RealMatrix<Scalar> operator*(const LHS& lhs, const Real<Scalar>& rhs) {
  RealMatrix<Scalar> result(detail::empty, lhs.rows(), lhs.cols());

  for (int row = 0; row < result.rows(); ++row) {
    for (int col = 0; col < result.cols(); ++col) {
      result[row, col] = lhs[row, col] * rhs;
    }
  }

  return result;
}

/// Scalar-matrix multiplication operator.
///
/// @tparam Scalar Scalar type.
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
template <typename Scalar, EigenMatrixLike RHS>
  requires std::same_as<typename std::decay_t<RHS>::Scalar, Scalar>
RealMatrix<Scalar> operator*(const Real<Scalar>& lhs, const RHS& rhs) {
  RealMatrix<Scalar> result(detail::empty, rhs.rows(), rhs.cols());

  for (int row = 0; row < result.rows(); ++row) {
    for (int col = 0; col < result.cols(); ++col) {
      result[row, col] = rhs[row, col] * lhs;
    }
  }

  return result;
}

/// Binary division operator.
///
/// @tparam Scalar Scalar type.
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return Result of division.
template <typename Scalar, EigenMatrixLike LHS>
  requires std::same_as<typename std::decay_t<LHS>::Scalar, Scalar>
RealMatrix<Scalar> operator/(const LHS& lhs, const Real<Scalar>& rhs) {
  RealMatrix<Scalar> result(detail::empty, lhs.rows(), lhs.cols());

  for (int row = 0; row < result.rows(); ++row) {
    for (int col = 0; col < result.cols(); ++col) {
      result[row, col] = lhs[row, col] / rhs;
    }
  }

  return result;
}

/// Applies a coefficient-wise reduce operation to two matrices.
///
/// @tparam Scalar Scalar type.
/// @param lhs The left-hand side of the binary operator.
/// @param rhs The right-hand side of the binary operator.
/// @param binary_op The binary operator to use for the reduce operation.
template <typename Scalar>
RealMatrix<Scalar> cwise_reduce(
    const RealMatrix<Scalar>& lhs, const RealMatrix<Scalar>& rhs,
    function_ref<Real<Scalar>(const Real<Scalar>& x, const Real<Scalar>& y)>
        binary_op) {
  slp_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());

  RealMatrix<Scalar> result{detail::empty, lhs.rows(), lhs.cols()};

  for (int row = 0; row < lhs.rows(); ++row) {
    for (int col = 0; col < lhs.cols(); ++col) {
      result[row, col] = binary_op(lhs[row, col], rhs[row, col]);
    }
  }

  return result;
}

/// Assembles a RealMatrix from a nested list of blocks.
///
/// Each row's blocks must have the same height, and the assembled block rows
/// must have the same width. For example, for the block matrix [[A, B], [C]] to
/// be constructible, the number of rows in A and B must match, and the number
/// of columns in [A, B] and [C] must match.
///
/// @tparam Scalar Scalar type.
/// @param list The nested list of blocks.
template <typename Scalar>
RealMatrix<Scalar> block(
    std::initializer_list<std::initializer_list<RealMatrix<Scalar>>> list) {
  // Get row and column counts for destination matrix
  int rows = 0;
  int cols = -1;
  for (const auto& row : list) {
    if (row.size() > 0) {
      rows += row.begin()->rows();
    }

    // Get number of columns in this row
    int latest_cols = 0;
    for (const auto& elem : row) {
      // Assert the first and latest row have the same height
      slp_assert(row.begin()->rows() == elem.rows());

      latest_cols += elem.cols();
    }

    // If this is the first row, record the column count. Otherwise, assert the
    // first and latest column counts are the same.
    if (cols == -1) {
      cols = latest_cols;
    } else {
      slp_assert(cols == latest_cols);
    }
  }

  RealMatrix<Scalar> result{detail::empty, rows, cols};

  int row_offset = 0;
  for (const auto& row : list) {
    int col_offset = 0;
    for (const auto& elem : row) {
      result.block(row_offset, col_offset, elem.rows(), elem.cols()) = elem;
      col_offset += elem.cols();
    }
    if (row.size() > 0) {
      row_offset += row.begin()->rows();
    }
  }

  return result;
}

/// Assembles a RealMatrix from a nested list of blocks.
///
/// Each row's blocks must have the same height, and the assembled block rows
/// must have the same width. For example, for the block matrix [[A, B], [C]] to
/// be constructible, the number of rows in A and B must match, and the number
/// of columns in [A, B] and [C] must match.
///
/// This overload is for Python bindings only.
///
/// @tparam Scalar Scalar type.
/// @param list The nested list of blocks.
template <typename Scalar>
RealMatrix<Scalar> block(
    const std::vector<std::vector<RealMatrix<Scalar>>>& list) {
  // Get row and column counts for destination matrix
  int rows = 0;
  int cols = -1;
  for (const auto& row : list) {
    if (row.size() > 0) {
      rows += row.begin()->rows();
    }

    // Get number of columns in this row
    int latest_cols = 0;
    for (const auto& elem : row) {
      // Assert the first and latest row have the same height
      slp_assert(row.begin()->rows() == elem.rows());

      latest_cols += elem.cols();
    }

    // If this is the first row, record the column count. Otherwise, assert the
    // first and latest column counts are the same.
    if (cols == -1) {
      cols = latest_cols;
    } else {
      slp_assert(cols == latest_cols);
    }
  }

  RealMatrix<Scalar> result{detail::empty, rows, cols};

  int row_offset = 0;
  for (const auto& row : list) {
    int col_offset = 0;
    for (const auto& elem : row) {
      result.block(row_offset, col_offset, elem.rows(), elem.cols()) = elem;
      col_offset += elem.cols();
    }
    if (row.size() > 0) {
      row_offset += row.begin()->rows();
    }
  }

  return result;
}

/// Solves the matrix equation AX = B for X.
///
/// @tparam Scalar Scalar type.
/// @param A The left-hand side.
/// @param B The right-hand side.
/// @return The solution X.
template <typename Scalar>
RealMatrix<Scalar> solve(const RealMatrix<Scalar>& A,
                         const RealMatrix<Scalar>& B) {
  // m x n * n x p = m x p
  slp_assert(A.rows() == B.rows());

  if (A.rows() == 1 && A.cols() == 1) {
    // Compute optimal inverse instead of using Eigen's general solver
    return B[0, 0] / A[0, 0];
  } else if (A.rows() == 2 && A.cols() == 2) {
    // Compute optimal inverse instead of using Eigen's general solver
    //
    // [a  b]⁻¹  ___1___ [ d  −b]
    // [c  d]  = ad − bc [−c   a]

    const auto& a = A[0, 0];
    const auto& b = A[0, 1];
    const auto& c = A[1, 0];
    const auto& d = A[1, 1];

    RealMatrix adj_A{{d, -b}, {-c, a}};
    auto det_A = a * d - b * c;
    return adj_A / det_A * B;
  } else if (A.rows() == 3 && A.cols() == 3) {
    // Compute optimal inverse instead of using Eigen's general solver
    //
    // [a  b  c]⁻¹
    // [d  e  f]
    // [g  h  i]
    //                        1                   [ei − fh  ch − bi  bf − ce]
    //     = ------------------------------------ [fg − di  ai − cg  cd − af]
    //       a(ei − fh) + b(fg − di) + c(dh − eg) [dh − eg  bg − ah  ae − bd]
    //
    // https://www.wolframalpha.com/input?i=inverse+%7B%7Ba%2C+b%2C+c%7D%2C+%7Bd%2C+e%2C+f%7D%2C+%7Bg%2C+h%2C+i%7D%7D

    const auto& a = A[0, 0];
    const auto& b = A[0, 1];
    const auto& c = A[0, 2];
    const auto& d = A[1, 0];
    const auto& e = A[1, 1];
    const auto& f = A[1, 2];
    const auto& g = A[2, 0];
    const auto& h = A[2, 1];
    const auto& i = A[2, 2];

    auto ae = a * e;
    auto af = a * f;
    auto ah = a * h;
    auto ai = a * i;
    auto bd = b * d;
    auto bf = b * f;
    auto bg = b * g;
    auto bi = b * i;
    auto cd = c * d;
    auto ce = c * e;
    auto cg = c * g;
    auto ch = c * h;
    auto dh = d * h;
    auto di = d * i;
    auto eg = e * g;
    auto ei = e * i;
    auto fg = f * g;
    auto fh = f * h;

    auto adj_A00 = ei - fh;
    auto adj_A10 = fg - di;
    auto adj_A20 = dh - eg;

    RealMatrix adj_A{{adj_A00, ch - bi, bf - ce},
                     {adj_A10, ai - cg, cd - af},
                     {adj_A20, bg - ah, ae - bd}};
    auto det_A = a * adj_A00 + b * adj_A10 + c * adj_A20;
    return adj_A / det_A * B;
  } else if (A.rows() == 4 && A.cols() == 4) {
    // Compute optimal inverse instead of using Eigen's general solver
    //
    // [a  b  c  d]⁻¹
    // [e  f  g  h]
    // [i  j  k  l]
    // [m  n  o  p]
    //
    // https://www.wolframalpha.com/input?i=inverse+%7B%7Ba%2C+b%2C+c%2C+d%7D%2C+%7Be%2C+f%2C+g%2C+h%7D%2C+%7Bi%2C+j%2C+k%2C+l%7D%2C+%7Bm%2C+n%2C+o%2C+p%7D%7D

    const auto& a = A[0, 0];
    const auto& b = A[0, 1];
    const auto& c = A[0, 2];
    const auto& d = A[0, 3];
    const auto& e = A[1, 0];
    const auto& f = A[1, 1];
    const auto& g = A[1, 2];
    const auto& h = A[1, 3];
    const auto& i = A[2, 0];
    const auto& j = A[2, 1];
    const auto& k = A[2, 2];
    const auto& l = A[2, 3];
    const auto& m = A[3, 0];
    const auto& n = A[3, 1];
    const auto& o = A[3, 2];
    const auto& p = A[3, 3];

    auto afk = a * f * k;
    auto afl = a * f * l;
    auto afo = a * f * o;
    auto afp = a * f * p;
    auto agj = a * g * j;
    auto agl = a * g * l;
    auto agn = a * g * n;
    auto agp = a * g * p;
    auto ahj = a * h * j;
    auto ahk = a * h * k;
    auto ahn = a * h * n;
    auto aho = a * h * o;
    auto ajo = a * j * o;
    auto ajp = a * j * p;
    auto akn = a * k * n;
    auto akp = a * k * p;
    auto aln = a * l * n;
    auto alo = a * l * o;
    auto bek = b * e * k;
    auto bel = b * e * l;
    auto beo = b * e * o;
    auto bep = b * e * p;
    auto bgi = b * g * i;
    auto bgl = b * g * l;
    auto bgm = b * g * m;
    auto bgp = b * g * p;
    auto bhi = b * h * i;
    auto bhk = b * h * k;
    auto bhm = b * h * m;
    auto bho = b * h * o;
    auto bio = b * i * o;
    auto bip = b * i * p;
    auto bjp = b * j * p;
    auto bkm = b * k * m;
    auto bkp = b * k * p;
    auto blm = b * l * m;
    auto blo = b * l * o;
    auto cej = c * e * j;
    auto cel = c * e * l;
    auto cen = c * e * n;
    auto cep = c * e * p;
    auto cfi = c * f * i;
    auto cfl = c * f * l;
    auto cfm = c * f * m;
    auto cfp = c * f * p;
    auto chi = c * h * i;
    auto chj = c * h * j;
    auto chm = c * h * m;
    auto chn = c * h * n;
    auto cin = c * i * n;
    auto cip = c * i * p;
    auto cjm = c * j * m;
    auto cjp = c * j * p;
    auto clm = c * l * m;
    auto cln = c * l * n;
    auto dej = d * e * j;
    auto dek = d * e * k;
    auto den = d * e * n;
    auto deo = d * e * o;
    auto dfi = d * f * i;
    auto dfk = d * f * k;
    auto dfm = d * f * m;
    auto dfo = d * f * o;
    auto dgi = d * g * i;
    auto dgj = d * g * j;
    auto dgm = d * g * m;
    auto dgn = d * g * n;
    auto din = d * i * n;
    auto dio = d * i * o;
    auto djm = d * j * m;
    auto djo = d * j * o;
    auto dkm = d * k * m;
    auto dkn = d * k * n;
    auto ejo = e * j * o;
    auto ejp = e * j * p;
    auto ekn = e * k * n;
    auto ekp = e * k * p;
    auto eln = e * l * n;
    auto elo = e * l * o;
    auto fio = f * i * o;
    auto fip = f * i * p;
    auto fkm = f * k * m;
    auto fkp = f * k * p;
    auto flm = f * l * m;
    auto flo = f * l * o;
    auto gin = g * i * n;
    auto gip = g * i * p;
    auto gjm = g * j * m;
    auto gjp = g * j * p;
    auto glm = g * l * m;
    auto gln = g * l * n;
    auto hin = h * i * n;
    auto hio = h * i * o;
    auto hjm = h * j * m;
    auto hjo = h * j * o;
    auto hkm = h * k * m;
    auto hkn = h * k * n;

    auto adj_A00 = fkp - flo - gjp + gln + hjo - hkn;
    auto adj_A01 = -bkp + blo + cjp - cln - djo + dkn;
    auto adj_A02 = bgp - bho - cfp + chn + dfo - dgn;
    auto adj_A03 = -bgl + bhk + cfl - chj - dfk + dgj;
    auto adj_A10 = -ekp + elo + gip - glm - hio + hkm;
    auto adj_A11 = akp - alo - cip + clm + dio - dkm;
    auto adj_A12 = -agp + aho + cep - chm - deo + dgm;
    auto adj_A13 = agl - ahk - cel + chi + dek - dgi;
    auto adj_A20 = ejp - eln - fip + flm + hin - hjm;
    auto adj_A21 = -ajp + aln + bip - blm - din + djm;
    auto adj_A22 = afp - ahn - bep + bhm + den - dfm;
    auto adj_A23 = -afl + ahj + bel - bhi - dej + dfi;
    auto adj_A30 = -ejo + ekn + fio - fkm - gin + gjm;
    // NOLINTNEXTLINE(build/include_what_you_use)
    auto adj_A31 = ajo - akn - bio + bkm + cin - cjm;
    auto adj_A32 = -afo + agn + beo - bgm - cen + cfm;
    auto adj_A33 = afk - agj - bek + bgi + cej - cfi;

    RealMatrix adj_A{{adj_A00, adj_A01, adj_A02, adj_A03},
                     {adj_A10, adj_A11, adj_A12, adj_A13},
                     {adj_A20, adj_A21, adj_A22, adj_A23},
                     {adj_A30, adj_A31, adj_A32, adj_A33}};
    auto det_A = a * adj_A00 + b * adj_A10 + c * adj_A20 + d * adj_A30;
    return adj_A / det_A * B;
  } else {
    return RealMatrix<Scalar>{A.to_eigen().householderQr().solve(B.to_eigen())};
  }
}

namespace detail {

/// Returns the variable's gradient tree.
///
/// This function lazily allocates variables, so elements of the returned
/// RealMatrix will be empty if the corresponding element of wrt had no
/// adjoint. Ensure Real::expr isn't nullptr before calling member
/// functions.
///
/// @tparam Scalar Scalar type.
/// @param top_list Topologically sorted graph from parent to child.
/// @param wrt Variables with respect to which to compute the gradient.
/// @return The variable's gradient tree.
template <typename Scalar>
RealMatrix<Scalar> gradient_tree(const ExpressionGraph<Scalar>& top_list,
                                 const RealMatrix<Scalar>& wrt) {
  slp_assert(wrt.cols() == 1);

  // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
  // for background on reverse accumulation automatic differentiation.

  if (top_list.empty()) {
    return RealMatrix<Scalar>{detail::empty, wrt.rows(), 1};
  }

  // Set root node's adjoint to 1 since df/df is 1
  top_list[0]->adj_expr = constant_ptr(Scalar(1));

  // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
  // multiplied by dy/dx. If there are multiple "paths" from the root node to
  // variable; the variable's adjoint is the sum of each path's adjoint
  // contribution.
  for (auto& node : top_list) {
    // Leaf nodes have no children to propagate adjoints to, and a null (zero)
    // adjoint has nothing to contribute
    if (!node->is_leaf && node->adj_expr != nullptr) {
      node->accumulate_adjoints_expr();
    }
  }

  // Move gradient tree to return value
  RealMatrix<Scalar> grad{detail::empty, wrt.rows(), 1};
  for (int row = 0; row < grad.rows(); ++row) {
    grad[row] = Real{std::move(wrt[row].expr->adj_expr)};
  }

  // Unlink adjoints to avoid circular references between them and their
  // parent expressions. This ensures all expressions are returned to the free
  // list.
  for (auto& node : top_list) {
    node->adj_expr = nullptr;
  }

  return grad;
}

}  // namespace detail

extern template class EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
RealMatrix<double>;

extern template SLEIPNIR_DLLEXPORT RealMatrix<double> solve(
    const RealMatrix<double>& A, const RealMatrix<double>& B);

}  // namespace slp
