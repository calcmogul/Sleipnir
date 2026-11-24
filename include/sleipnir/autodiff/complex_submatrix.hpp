// Copyright (c) Sleipnir contributors

#pragma once

#include <complex>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/autodiff/complex.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/autodiff/slice.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/function_ref.hpp"

namespace slp {

/// A submatrix of complex autodiff expressions with reference semantics.
///
/// Assigning to a ComplexSubmatrix assigns to the entries of the matrix it
/// points to.
///
/// @tparam Mat The type of the matrix whose storage this class points to.
template <typename Mat>
class ComplexSubmatrix {
 public:
  /// Scalar type alias.
  using Scalar = typename Mat::Scalar;

  /// Copy constructor.
  ComplexSubmatrix(const ComplexSubmatrix<Mat>&) = default;

  /// Assigns a ComplexSubmatrix to the block.
  ///
  /// @param values ComplexSubmatrix of values.
  /// @return This submatrix.
  ComplexSubmatrix<Mat>& operator=(const ComplexSubmatrix<Mat>& values) {
    if (this == &values) {
      return *this;
    }

    if (m_mat == nullptr) {
      m_mat = values.m_mat;
      m_row_slice = values.m_row_slice;
      m_row_slice_length = values.m_row_slice_length;
      m_col_slice = values.m_col_slice;
      m_col_slice_length = values.m_col_slice_length;
    } else {
      slp_assert(rows() == values.rows() && cols() == values.cols());

      for (int row = 0; row < rows(); ++row) {
        for (int col = 0; col < cols(); ++col) {
          (*this)[row, col] = values[row, col];
        }
      }
    }

    return *this;
  }

  /// Move constructor.
  ComplexSubmatrix(ComplexSubmatrix<Mat>&&) = default;

  /// Assigns a ComplexSubmatrix to the block.
  ///
  /// @param values ComplexSubmatrix of values.
  /// @return This submatrix.
  ComplexSubmatrix<Mat>& operator=(ComplexSubmatrix<Mat>&& values) {
    return *this = values;
  }

  /// Constructs a ComplexSubmatrix pointing to all of the given matrix.
  ///
  /// @param mat The matrix to which to point.
  // NOLINTNEXTLINE (google-explicit-constructor)
  ComplexSubmatrix(Mat& mat)
      : ComplexSubmatrix{mat, 0, 0, mat.rows(), mat.cols()} {}

  /// Constructs a ComplexSubmatrix pointing to a subset of the given matrix.
  ///
  /// @param mat The matrix to which to point.
  /// @param row_offset The block's row offset.
  /// @param col_offset The block's column offset.
  /// @param block_rows The number of rows in the block.
  /// @param block_cols The number of columns in the block.
  ComplexSubmatrix(Mat& mat, int row_offset, int col_offset, int block_rows,
                   int block_cols)
      : m_mat{&mat},
        m_row_slice{row_offset, row_offset + block_rows, 1},
        m_row_slice_length{m_row_slice.adjust(mat.rows())},
        m_col_slice{col_offset, col_offset + block_cols, 1},
        m_col_slice_length{m_col_slice.adjust(mat.cols())} {}

  /// Constructs a ComplexSubmatrix pointing to a subset of the given matrix.
  ///
  /// Note that the slices are taken as is rather than adjusted.
  ///
  /// @param mat The matrix to which to point.
  /// @param row_slice The block's row slice.
  /// @param row_slice_length The block's row length.
  /// @param col_slice The block's column slice.
  /// @param col_slice_length The block's column length.
  ComplexSubmatrix(Mat& mat, Slice row_slice, int row_slice_length,
                   Slice col_slice, int col_slice_length)
      : m_mat{&mat},
        m_row_slice{std::move(row_slice)},
        m_row_slice_length{row_slice_length},
        m_col_slice{std::move(col_slice)},
        m_col_slice_length{col_slice_length} {}

  /// Assigns a scalar to the block.
  ///
  /// This only works for blocks with one row and one column.
  ///
  /// @param value Value to assign.
  /// @return This submatrix.
  template <detail::ComplexOperand<Scalar> T>
  ComplexSubmatrix<Mat>& operator=(const T& value) {
    slp_assert(rows() == 1 && cols() == 1);

    (*this)[0, 0] = Complex<Scalar>{value};

    return *this;
  }

  /// Sets the block's internal value.
  ///
  /// This only works for blocks with one row and one column.
  ///
  /// @param value Value to assign.
  void set_value(const std::complex<Scalar>& value) {
    slp_assert(rows() == 1 && cols() == 1);

    (*this)[0, 0].set_value(value);
  }

  /// Assigns an Eigen matrix of real or complex constants to the block.
  ///
  /// @param values Eigen matrix of values to assign.
  /// @return This submatrix.
  template <typename Derived>
  ComplexSubmatrix<Mat>& operator=(const Eigen::MatrixBase<Derived>& values) {
    slp_assert(rows() == values.rows() && cols() == values.cols());

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] = Complex<Scalar>{values[row, col]};
      }
    }

    return *this;
  }

  /// Sets the block's internal values.
  ///
  /// Real values set the imaginary parts to zero.
  ///
  /// @param values Eigen matrix of values.
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar> ||
             std::same_as<typename Derived::Scalar, std::complex<Scalar>>
  void set_value(const Eigen::MatrixBase<Derived>& values) {
    slp_assert(rows() == values.rows() && cols() == values.cols());

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col].set_value(std::complex<Scalar>{values[row, col]});
      }
    }
  }

  /// Assigns a ComplexMatrix to the block.
  ///
  /// @param values ComplexMatrix of values.
  /// @return This submatrix.
  ComplexSubmatrix<Mat>& operator=(const std::remove_cv_t<Mat>& values) {
    slp_assert(rows() == values.rows() && cols() == values.cols());

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] = values[row, col];
      }
    }

    return *this;
  }

  /// Assigns a ComplexMatrix to the block.
  ///
  /// @param values ComplexMatrix of values.
  /// @return This submatrix.
  ComplexSubmatrix<Mat>& operator=(std::remove_cv_t<Mat>&& values) {
    slp_assert(rows() == values.rows() && cols() == values.cols());

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] = std::move(values[row, col]);
      }
    }

    return *this;
  }

  /// Returns the element at the given row and column.
  ///
  /// @param row The row.
  /// @param col The column.
  /// @return The element at the given row and column.
  Complex<Scalar>& operator[](int row, int col)
    requires(!std::is_const_v<Mat>)
  {
    slp_assert(row >= 0 && row < rows());
    slp_assert(col >= 0 && col < cols());
    return (*m_mat)[m_row_slice.start + row * m_row_slice.step,
                    m_col_slice.start + col * m_col_slice.step];
  }

  /// Returns the element at the given row and column.
  ///
  /// @param row The row.
  /// @param col The column.
  /// @return The element at the given row and column.
  const Complex<Scalar>& operator[](int row, int col) const {
    slp_assert(row >= 0 && row < rows());
    slp_assert(col >= 0 && col < cols());
    return (*m_mat)[m_row_slice.start + row * m_row_slice.step,
                    m_col_slice.start + col * m_col_slice.step];
  }

  /// Returns the element at the given index.
  ///
  /// @param index The index.
  /// @return The element at the given index.
  Complex<Scalar>& operator[](int index)
    requires(!std::is_const_v<Mat>)
  {
    slp_assert(index >= 0 && index < rows() * cols());
    return (*this)[index / cols(), index % cols()];
  }

  /// Returns the element at the given index.
  ///
  /// @param index The index.
  /// @return The element at the given index.
  const Complex<Scalar>& operator[](int index) const {
    slp_assert(index >= 0 && index < rows() * cols());
    return (*this)[index / cols(), index % cols()];
  }

  /// Returns a block of the submatrix.
  ///
  /// @param row_offset The row offset of the block selection.
  /// @param col_offset The column offset of the block selection.
  /// @param block_rows The number of rows in the block selection.
  /// @param block_cols The number of columns in the block selection.
  /// @return A block of the submatrix.
  ComplexSubmatrix<Mat> block(int row_offset, int col_offset, int block_rows,
                              int block_cols) {
    slp_assert(row_offset >= 0 && row_offset <= rows());
    slp_assert(col_offset >= 0 && col_offset <= cols());
    slp_assert(block_rows >= 0 && block_rows <= rows() - row_offset);
    slp_assert(block_cols >= 0 && block_cols <= cols() - col_offset);
    return (*this)[Slice{row_offset, row_offset + block_rows, 1},
                   Slice{col_offset, col_offset + block_cols, 1}];
  }

  /// Returns a block of the submatrix.
  ///
  /// @param row_offset The row offset of the block selection.
  /// @param col_offset The column offset of the block selection.
  /// @param block_rows The number of rows in the block selection.
  /// @param block_cols The number of columns in the block selection.
  /// @return A block of the submatrix.
  ComplexSubmatrix<const Mat> block(int row_offset, int col_offset,
                                    int block_rows, int block_cols) const {
    slp_assert(row_offset >= 0 && row_offset <= rows());
    slp_assert(col_offset >= 0 && col_offset <= cols());
    slp_assert(block_rows >= 0 && block_rows <= rows() - row_offset);
    slp_assert(block_cols >= 0 && block_cols <= cols() - col_offset);
    return (*this)[Slice{row_offset, row_offset + block_rows, 1},
                   Slice{col_offset, col_offset + block_cols, 1}];
  }

  /// Returns a slice of the submatrix.
  ///
  /// @param row_slice The row slice.
  /// @param col_slice The column slice.
  /// @return A slice of the submatrix.
  ComplexSubmatrix<Mat> operator[](Slice row_slice, Slice col_slice) {
    int row_slice_length = row_slice.adjust(m_row_slice_length);
    int col_slice_length = col_slice.adjust(m_col_slice_length);
    return (*this)[row_slice, row_slice_length, col_slice, col_slice_length];
  }

  /// Returns a slice of the submatrix.
  ///
  /// @param row_slice The row slice.
  /// @param col_slice The column slice.
  /// @return A slice of the submatrix.
  ComplexSubmatrix<const Mat> operator[](Slice row_slice,
                                         Slice col_slice) const {
    int row_slice_length = row_slice.adjust(m_row_slice_length);
    int col_slice_length = col_slice.adjust(m_col_slice_length);
    return (*this)[row_slice, row_slice_length, col_slice, col_slice_length];
  }

  /// Returns a slice of the submatrix.
  ///
  /// The given slices aren't adjusted. This overload is for Python bindings
  /// only.
  ///
  /// @param row_slice The row slice.
  /// @param row_slice_length The row slice length.
  /// @param col_slice The column slice.
  /// @param col_slice_length The column slice length.
  /// @return A slice of the submatrix.
  ComplexSubmatrix<Mat> operator[](Slice row_slice, int row_slice_length,
                                   Slice col_slice, int col_slice_length) {
    return ComplexSubmatrix<Mat>{
        *m_mat, compose(m_row_slice, row_slice), row_slice_length,
        compose(m_col_slice, col_slice), col_slice_length};
  }

  /// Returns a slice of the submatrix.
  ///
  /// The given slices aren't adjusted. This overload is for Python bindings
  /// only.
  ///
  /// @param row_slice The row slice.
  /// @param row_slice_length The row slice length.
  /// @param col_slice The column slice.
  /// @param col_slice_length The column slice length.
  /// @return A slice of the submatrix.
  ComplexSubmatrix<const Mat> operator[](Slice row_slice, int row_slice_length,
                                         Slice col_slice,
                                         int col_slice_length) const {
    return ComplexSubmatrix<const Mat>{
        *m_mat, compose(m_row_slice, row_slice), row_slice_length,
        compose(m_col_slice, col_slice), col_slice_length};
  }

  /// Returns a segment of the submatrix.
  ///
  /// @param offset The offset of the segment.
  /// @param length The length of the segment.
  /// @return A segment of the submatrix.
  ComplexSubmatrix<Mat> segment(int offset, int length) {
    slp_assert(cols() == 1);
    return block(offset, 0, length, 1);
  }

  /// Returns a segment of the submatrix.
  ///
  /// @param offset The offset of the segment.
  /// @param length The length of the segment.
  /// @return A segment of the submatrix.
  ComplexSubmatrix<const Mat> segment(int offset, int length) const {
    slp_assert(cols() == 1);
    return block(offset, 0, length, 1);
  }

  /// Returns a row slice of the submatrix.
  ///
  /// @param row The row to slice.
  /// @return A row slice of the submatrix.
  ComplexSubmatrix<Mat> row(int row) { return block(row, 0, 1, cols()); }

  /// Returns a row slice of the submatrix.
  ///
  /// @param row The row to slice.
  /// @return A row slice of the submatrix.
  ComplexSubmatrix<const Mat> row(int row) const {
    return block(row, 0, 1, cols());
  }

  /// Returns a column slice of the submatrix.
  ///
  /// @param col The column to slice.
  /// @return A column slice of the submatrix.
  ComplexSubmatrix<Mat> col(int col) { return block(0, col, rows(), 1); }

  /// Returns a column slice of the submatrix.
  ///
  /// @param col The column to slice.
  /// @return A column slice of the submatrix.
  ComplexSubmatrix<const Mat> col(int col) const {
    return block(0, col, rows(), 1);
  }

  /// Compound multiplication-assignment operator.
  ///
  /// @param rhs Matrix or scalar to multiply.
  /// @return Result of multiplication.
  template <typename RHS>
    requires requires(const ComplexSubmatrix<Mat>& lhs, const RHS& rhs) {
      { lhs * rhs } -> std::same_as<std::remove_cv_t<Mat>>;
    }
  ComplexSubmatrix<Mat>& operator*=(const RHS& rhs) {
    *this = *this * rhs;
    return *this;
  }

  /// Compound division-assignment operator.
  ///
  /// @param rhs Scalar to divide.
  /// @return Result of division.
  template <typename RHS>
    requires requires(const ComplexSubmatrix<Mat>& lhs, const RHS& rhs) {
      { lhs / rhs } -> std::same_as<std::remove_cv_t<Mat>>;
    }
  ComplexSubmatrix<Mat>& operator/=(const RHS& rhs) {
    *this = *this / rhs;
    return *this;
  }

  /// Compound addition-assignment operator.
  ///
  /// @param rhs Matrix to add.
  /// @return Result of addition.
  template <typename RHS>
    requires requires(const ComplexSubmatrix<Mat>& lhs, const RHS& rhs) {
      { lhs + rhs } -> std::same_as<std::remove_cv_t<Mat>>;
    }
  ComplexSubmatrix<Mat>& operator+=(const RHS& rhs) {
    *this = *this + rhs;
    return *this;
  }

  /// Compound subtraction-assignment operator.
  ///
  /// @param rhs Matrix to subtract.
  /// @return Result of subtraction.
  template <typename RHS>
    requires requires(const ComplexSubmatrix<Mat>& lhs, const RHS& rhs) {
      { lhs - rhs } -> std::same_as<std::remove_cv_t<Mat>>;
    }
  ComplexSubmatrix<Mat>& operator-=(const RHS& rhs) {
    *this = *this - rhs;
    return *this;
  }

  /// Implicit conversion operator from 1x1 ComplexSubmatrix to Complex.
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator Complex<Scalar>() const {
    slp_assert(rows() == 1 && cols() == 1);
    return (*this)[0, 0];
  }

  /// Returns the transpose of the submatrix.
  ///
  /// The entries aren't conjugated. For the conjugate transpose, use H().
  ///
  /// @return The transpose of the submatrix.
  std::remove_cv_t<Mat> T() const {
    std::remove_cv_t<Mat> result{detail::empty, cols(), rows()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[col, row] = (*this)[row, col];
      }
    }

    return result;
  }

  /// Returns the elementwise complex conjugate of the submatrix.
  ///
  /// @return The elementwise complex conjugate of the submatrix.
  std::remove_cv_t<Mat> conj() const {
    return cwise_transform([](const Complex<Scalar>& z) { return z.conj(); });
  }

  /// Returns the conjugate transpose of the submatrix.
  ///
  /// @return The conjugate transpose of the submatrix.
  std::remove_cv_t<Mat> H() const {
    std::remove_cv_t<Mat> result{detail::empty, cols(), rows()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[col, row] = (*this)[row, col].conj();
      }
    }

    return result;
  }

  /// Returns the trace of the submatrix, the sum of its diagonal.
  ///
  /// @return The trace of the submatrix.
  Complex<Scalar> trace() const {
    slp_assert(rows() == cols());

    Complex<Scalar> result;

    for (int row = 0; row < rows(); ++row) {
      result = result + (*this)[row, row];
    }

    return result;
  }

  /// Returns the number of rows in the matrix.
  ///
  /// @return The number of rows in the matrix.
  int rows() const { return m_row_slice_length; }

  /// Returns the number of columns in the matrix.
  ///
  /// @return The number of columns in the matrix.
  int cols() const { return m_col_slice_length; }

  /// Returns an element of the submatrix.
  ///
  /// @param row The row of the element to return.
  /// @param col The column of the element to return.
  /// @return An element of the submatrix.
  std::complex<Scalar> value(int row, int col) {
    return (*this)[row, col].value();
  }

  /// Returns an element of the submatrix.
  ///
  /// @param index The index of the element to return.
  /// @return An element of the submatrix.
  std::complex<Scalar> value(int index) {
    slp_assert(index >= 0 && index < rows() * cols());
    return value(index / cols(), index % cols());
  }

  /// Returns the contents of the submatrix.
  ///
  /// @return The contents of the submatrix.
  Eigen::MatrixX<std::complex<Scalar>> value() {
    Eigen::MatrixX<std::complex<Scalar>> result{rows(), cols()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[row, col] = value(row, col);
      }
    }

    return result;
  }

  /// Returns the real part of each element.
  ///
  /// @return The real part of each element.
  RealMatrix<Scalar> real() const {
    RealMatrix<Scalar> result{detail::empty, rows(), cols()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[row, col] = (*this)[row, col].real();
      }
    }

    return result;
  }

  /// Returns the imaginary part of each element.
  ///
  /// @return The imaginary part of each element.
  RealMatrix<Scalar> imag() const {
    RealMatrix<Scalar> result{detail::empty, rows(), cols()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[row, col] = (*this)[row, col].imag();
      }
    }

    return result;
  }

  /// Returns the squared magnitude of each element.
  ///
  /// @return The squared magnitude of each element.
  RealMatrix<Scalar> abs2() const {
    RealMatrix<Scalar> result{detail::empty, rows(), cols()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[row, col] = (*this)[row, col].abs2();
      }
    }

    return result;
  }

  /// Returns the coefficient-wise product with another matrix.
  ///
  /// @param rhs The other matrix.
  /// @return The coefficient-wise product.
  template <typename RHS>
    requires requires(const std::remove_cv_t<Mat>& lhs, const RHS& rhs) {
      lhs.cwise_product(rhs);
    }
  std::remove_cv_t<Mat> cwise_product(const RHS& rhs) const {
    return std::remove_cv_t<Mat>{*this}.cwise_product(rhs);
  }

  /// Transforms the matrix coefficient-wise with an unary operator.
  ///
  /// @param unary_op The unary operator to use for the transform operation.
  /// @return Result of the unary operator.
  std::remove_cv_t<Mat> cwise_transform(
      function_ref<Complex<Scalar>(const Complex<Scalar>& z)> unary_op) const {
    std::remove_cv_t<Mat> result{detail::empty, rows(), cols()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[row, col] = unary_op((*this)[row, col]);
      }
    }

    return result;
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS

  class iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = Complex<Scalar>;
    using difference_type = std::ptrdiff_t;
    using pointer = Complex<Scalar>*;
    using reference = Complex<Scalar>&;

    constexpr iterator() noexcept = default;

    constexpr iterator(ComplexSubmatrix<Mat>* mat, int index) noexcept
        : m_mat{mat}, m_index{index} {}

    constexpr iterator& operator++() noexcept {
      ++m_index;
      return *this;
    }

    constexpr iterator operator++(int) noexcept {
      iterator retval = *this;
      ++(*this);
      return retval;
    }

    constexpr iterator& operator--() noexcept {
      --m_index;
      return *this;
    }

    constexpr iterator operator--(int) noexcept {
      iterator retval = *this;
      --(*this);
      return retval;
    }

    constexpr bool operator==(const iterator&) const noexcept = default;

    constexpr reference operator*() const noexcept { return (*m_mat)[m_index]; }

   private:
    ComplexSubmatrix<Mat>* m_mat = nullptr;
    int m_index = 0;
  };

  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = Complex<Scalar>;
    using difference_type = std::ptrdiff_t;
    using pointer = const Complex<Scalar>*;
    using reference = const Complex<Scalar>&;

    constexpr const_iterator() noexcept = default;

    constexpr const_iterator(const ComplexSubmatrix<Mat>* mat,
                             int index) noexcept
        : m_mat{mat}, m_index{index} {}

    constexpr const_iterator& operator++() noexcept {
      ++m_index;
      return *this;
    }

    constexpr const_iterator operator++(int) noexcept {
      const_iterator retval = *this;
      ++(*this);
      return retval;
    }

    constexpr const_iterator& operator--() noexcept {
      --m_index;
      return *this;
    }

    constexpr const_iterator operator--(int) noexcept {
      const_iterator retval = *this;
      --(*this);
      return retval;
    }

    constexpr bool operator==(const const_iterator&) const noexcept = default;

    constexpr reference operator*() const noexcept { return (*m_mat)[m_index]; }

   private:
    const ComplexSubmatrix<Mat>* m_mat = nullptr;
    int m_index = 0;
  };

#endif  // DOXYGEN_SHOULD_SKIP_THIS

  /// Returns begin iterator.
  ///
  /// @return Begin iterator.
  iterator begin()
    requires(!std::is_const_v<Mat>)
  {
    return iterator(this, 0);
  }

  /// Returns end iterator.
  ///
  /// @return End iterator.
  iterator end()
    requires(!std::is_const_v<Mat>)
  {
    return iterator(this, rows() * cols());
  }

  /// Returns const begin iterator.
  ///
  /// @return Const begin iterator.
  const_iterator begin() const { return const_iterator(this, 0); }

  /// Returns const end iterator.
  ///
  /// @return Const end iterator.
  const_iterator end() const { return const_iterator(this, rows() * cols()); }

  /// Returns const begin iterator.
  ///
  /// @return Const begin iterator.
  const_iterator cbegin() const { return const_iterator(this, 0); }

  /// Returns const end iterator.
  ///
  /// @return Const end iterator.
  const_iterator cend() const { return const_iterator(this, rows() * cols()); }

  /// Returns number of elements in matrix.
  ///
  /// @return Number of elements in matrix.
  size_t size() const { return rows() * cols(); }

 private:
  template <typename>
  friend class ComplexSubmatrix;

  Mat* m_mat = nullptr;

  Slice m_row_slice;
  int m_row_slice_length = 0;

  Slice m_col_slice;
  int m_col_slice_length = 0;

  /// Returns a slice of this submatrix's slice in the matrix's indices.
  ///
  /// @param outer This submatrix's slice of the matrix.
  /// @param inner The slice of this submatrix.
  /// @return The slice of the matrix.
  static Slice compose(const Slice& outer, const Slice& inner) {
    return Slice{outer.start + inner.start * outer.step,
                 outer.start + inner.stop * outer.step,
                 inner.step * outer.step};
  }
};

}  // namespace slp
