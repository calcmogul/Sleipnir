// Copyright (c) Sleipnir contributors

#pragma once

#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/sleipnir_base.hpp"
#include "sleipnir/autodiff/slice.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/function_ref.hpp"

namespace slp {

/// A submatrix of autodiff expressions with reference semantics.
///
/// @tparam Mat The type of the matrix whose storage this class points to.
template <typename Mat>
class RealSubmatrix : public SleipnirBase {
 public:
  /// Scalar type alias.
  using Scalar = typename Mat::Scalar;

  /// Copy constructor.
  RealSubmatrix(const RealSubmatrix<Mat>&) = default;

  /// Assigns a RealSubmatrix to the block.
  ///
  /// @param values RealSubmatrix of values.
  /// @return This submatrix.
  RealSubmatrix<Mat>& operator=(const RealSubmatrix<Mat>& values) {
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
  RealSubmatrix(RealSubmatrix<Mat>&&) = default;

  /// Assigns a RealSubmatrix to the block.
  ///
  /// @param values RealSubmatrix of values.
  /// @return This submatrix.
  RealSubmatrix<Mat>& operator=(RealSubmatrix<Mat>&& values) {
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

  /// Constructs a RealSubmatrix pointing to all of the given matrix.
  ///
  /// @param mat The matrix to which to point.
  // NOLINTNEXTLINE (google-explicit-constructor)
  RealSubmatrix(Mat& mat) : RealSubmatrix{mat, 0, 0, mat.rows(), mat.cols()} {}

  /// Constructs a RealSubmatrix pointing to a subset of the given matrix.
  ///
  /// @param mat The matrix to which to point.
  /// @param row_offset The block's row offset.
  /// @param col_offset The block's column offset.
  /// @param block_rows The number of rows in the block.
  /// @param block_cols The number of columns in the block.
  RealSubmatrix(Mat& mat, int row_offset, int col_offset, int block_rows,
                int block_cols)
      : m_mat{&mat},
        m_row_slice{row_offset, row_offset + block_rows, 1},
        m_row_slice_length{m_row_slice.adjust(mat.rows())},
        m_col_slice{col_offset, col_offset + block_cols, 1},
        m_col_slice_length{m_col_slice.adjust(mat.cols())} {}

  /// Constructs a RealSubmatrix pointing to a subset of the given matrix.
  ///
  /// Note that the slices are taken as is rather than adjusted.
  ///
  /// @param mat The matrix to which to point.
  /// @param row_slice The block's row slice.
  /// @param row_slice_length The block's row length.
  /// @param col_slice The block's column slice.
  /// @param col_slice_length The block's column length.
  RealSubmatrix(Mat& mat, Slice row_slice, int row_slice_length,
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
  RealSubmatrix<Mat>& operator=(ScalarLike auto value) {
    slp_assert(rows() == 1 && cols() == 1);

    (*this)[0, 0] = value;

    return *this;
  }

  /// Assigns a scalar to the block.
  ///
  /// This only works for blocks with one row and one column.
  ///
  /// @param value Value to assign.
  void set_value(Scalar value) {
    slp_assert(rows() == 1 && cols() == 1);

    (*this)[0, 0].set_value(value);
  }

  /// Assigns an Eigen matrix to the block.
  ///
  /// @param values Eigen matrix of values to assign.
  /// @return This submatrix.
  template <typename Derived>
  RealSubmatrix<Mat>& operator=(const Eigen::MatrixBase<Derived>& values) {
    slp_assert(rows() == values.rows() && cols() == values.cols());

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] = values[row, col];
      }
    }

    return *this;
  }

  /// Sets the block's internal values.
  ///
  /// @param values Eigen matrix of values.
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, Scalar>
  void set_value(const Eigen::MatrixBase<Derived>& values) {
    slp_assert(rows() == values.rows() && cols() == values.cols());

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col].set_value(values[row, col]);
      }
    }
  }

  /// Assigns a RealMatrix to the block.
  ///
  /// @param values RealMatrix of values.
  /// @return This submatrix.
  RealSubmatrix<Mat>& operator=(const Mat& values) {
    slp_assert(rows() == values.rows() && cols() == values.cols());

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] = values[row, col];
      }
    }
    return *this;
  }

  /// Assigns a RealMatrix to the block.
  ///
  /// @param values RealMatrix of values.
  /// @return This submatrix.
  RealSubmatrix<Mat>& operator=(Mat&& values) {
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
  Real<Scalar>& operator[](int row, int col)
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
  const Real<Scalar>& operator[](int row, int col) const {
    slp_assert(row >= 0 && row < rows());
    slp_assert(col >= 0 && col < cols());
    return (*m_mat)[m_row_slice.start + row * m_row_slice.step,
                    m_col_slice.start + col * m_col_slice.step];
  }

  /// Returns the element at the given index.
  ///
  /// @param index The index.
  /// @return The element at the given index.
  Real<Scalar>& operator[](int index)
    requires(!std::is_const_v<Mat>)
  {
    slp_assert(index >= 0 && index < rows() * cols());
    return (*this)[index / cols(), index % cols()];
  }

  /// Returns the element at the given index.
  ///
  /// @param index The index.
  /// @return The element at the given index.
  const Real<Scalar>& operator[](int index) const {
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
  RealSubmatrix<Mat> block(int row_offset, int col_offset, int block_rows,
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
  const RealSubmatrix<const Mat> block(int row_offset, int col_offset,
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
  RealSubmatrix<Mat> operator[](Slice row_slice, Slice col_slice) {
    int row_slice_length = row_slice.adjust(m_row_slice_length);
    int col_slice_length = col_slice.adjust(m_col_slice_length);
    return (*this)[row_slice, row_slice_length, col_slice, col_slice_length];
  }

  /// Returns a slice of the submatrix.
  ///
  /// @param row_slice The row slice.
  /// @param col_slice The column slice.
  /// @return A slice of the submatrix.
  const RealSubmatrix<const Mat> operator[](Slice row_slice,
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
  RealSubmatrix<Mat> operator[](Slice row_slice, int row_slice_length,
                                Slice col_slice, int col_slice_length) {
    return RealSubmatrix{
        *m_mat,
        {m_row_slice.start + row_slice.start * m_row_slice.step,
         m_row_slice.start + row_slice.stop * m_row_slice.step,
         row_slice.step * m_row_slice.step},
        row_slice_length,
        {m_col_slice.start + col_slice.start * m_col_slice.step,
         m_col_slice.start + col_slice.stop * m_col_slice.step,
         col_slice.step * m_col_slice.step},
        col_slice_length};
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
  const RealSubmatrix<const Mat> operator[](Slice row_slice,
                                            int row_slice_length,
                                            Slice col_slice,
                                            int col_slice_length) const {
    return RealSubmatrix<const Mat>{
        *m_mat,
        {m_row_slice.start + row_slice.start * m_row_slice.step,
         m_row_slice.start + row_slice.stop * m_row_slice.step,
         row_slice.step * m_row_slice.step},
        row_slice_length,
        {m_col_slice.start + col_slice.start * m_col_slice.step,
         m_col_slice.start + col_slice.stop * m_col_slice.step,
         col_slice.step * m_col_slice.step},
        col_slice_length};
  }

  /// Returns a segment of the submatrix.
  ///
  /// @param offset The offset of the segment.
  /// @param length The length of the segment.
  /// @return A segment of the submatrix.
  RealSubmatrix<Mat> segment(int offset, int length) {
    slp_assert(cols() == 1);
    slp_assert(offset >= 0 && offset < rows());
    slp_assert(length >= 0 && length <= rows() - offset);
    return block(offset, 0, length, 1);
  }

  /// Returns a segment of the submatrix.
  ///
  /// @param offset The offset of the segment.
  /// @param length The length of the segment.
  /// @return A segment of the submatrix.
  const RealSubmatrix<const Mat> segment(int offset, int length) const {
    slp_assert(cols() == 1);
    slp_assert(offset >= 0 && offset < rows());
    slp_assert(length >= 0 && length <= rows() - offset);
    return block(offset, 0, length, 1);
  }

  /// Returns a row slice of the submatrix.
  ///
  /// @param row The row to slice.
  /// @return A row slice of the submatrix.
  RealSubmatrix<Mat> row(int row) {
    slp_assert(row >= 0 && row < rows());
    return block(row, 0, 1, cols());
  }

  /// Returns a row slice of the submatrix.
  ///
  /// @param row The row to slice.
  /// @return A row slice of the submatrix.
  RealSubmatrix<const Mat> row(int row) const {
    slp_assert(row >= 0 && row < rows());
    return block(row, 0, 1, cols());
  }

  /// Returns a column slice of the submatrix.
  ///
  /// @param col The column to slice.
  /// @return A column slice of the submatrix.
  RealSubmatrix<Mat> col(int col) {
    slp_assert(col >= 0 && col < cols());
    return block(0, col, rows(), 1);
  }

  /// Returns a column slice of the submatrix.
  ///
  /// @param col The column to slice.
  /// @return A column slice of the submatrix.
  RealSubmatrix<const Mat> col(int col) const {
    slp_assert(col >= 0 && col < cols());
    return block(0, col, rows(), 1);
  }

  /// Compound matrix multiplication-assignment operator.
  ///
  /// @param rhs Real to multiply.
  /// @return Result of multiplication.
  RealSubmatrix<Mat>& operator*=(const MatrixLike auto& rhs) {
    slp_assert(cols() == rhs.rows() && cols() == rhs.cols());

    for (int i = 0; i < rows(); ++i) {
      Mat lhs_old_row = row(i);
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

  /// Compound matrix multiplication-assignment operator.
  ///
  /// @param rhs Real to multiply.
  /// @return Result of multiplication.
  RealSubmatrix<Mat>& operator*=(const ScalarLike auto& rhs) {
    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] *= rhs;
      }
    }

    return *this;
  }

  /// Compound matrix division-assignment operator.
  ///
  /// @param rhs Real to divide.
  /// @return Result of division.
  RealSubmatrix<Mat>& operator/=(const ScalarLike auto& rhs) {
    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] /= rhs;
      }
    }

    return *this;
  }

  /// Compound addition-assignment operator.
  ///
  /// @param rhs Real to add.
  /// @return Result of addition.
  RealSubmatrix<Mat>& operator+=(const MatrixLike auto& rhs) {
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
  RealSubmatrix<Mat>& operator+=(const ScalarLike auto& rhs) {
    slp_assert(rows() == 1 && cols() == 1);

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] += rhs;
      }
    }

    return *this;
  }

  /// Compound subtraction-assignment operator.
  ///
  /// @param rhs Real to subtract.
  /// @return Result of subtraction.
  RealSubmatrix<Mat>& operator-=(const MatrixLike auto& rhs) {
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
  RealSubmatrix<Mat>& operator-=(const ScalarLike auto& rhs) {
    slp_assert(rows() == 1 && cols() == 1);

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] -= rhs;
      }
    }

    return *this;
  }

  /// Implicit conversion operator from 1x1 RealSubmatrix to Real.
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator Real<Scalar>() const {
    slp_assert(rows() == 1 && cols() == 1);
    return (*this)[0, 0];
  }

  /// Returns the transpose of the submatrix.
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

  /// Returns the trace of the submatrix, the sum of its diagonal.
  ///
  /// @return The trace of the submatrix.
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
  Scalar value(int row, int col) { return (*this)[row, col].value(); }

  /// Returns an element of the submatrix.
  ///
  /// @param index The index of the element to return.
  /// @return An element of the submatrix.
  Scalar value(int index) {
    slp_assert(index >= 0 && index < rows() * cols());
    return value(index / cols(), index % cols());
  }

  /// Returns the contents of the submatrix.
  ///
  /// @return The contents of the submatrix.
  Eigen::MatrixX<Scalar> value() {
    Eigen::MatrixX<Scalar> result{rows(), cols()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[row, col] = value(row, col);
      }
    }

    return result;
  }

  /// Transforms the matrix coefficient-wise with an unary operator.
  ///
  /// @param unary_op The unary operator to use for the transform operation.
  /// @return Result of the unary operator.
  std::remove_cv_t<Mat> cwise_transform(
      function_ref<Real<Scalar>(const Real<Scalar>& x)> unary_op) const {
    std::remove_cv_t<Mat> result{detail::empty, rows(), cols()};

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
  std::remove_cv_t<Mat> exp() const {
    return std::remove_cv_t<Mat>{*this}.exp();
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

    constexpr iterator(RealSubmatrix<Mat>* mat, int index) noexcept
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
    RealSubmatrix<Mat>* m_mat = nullptr;
    int m_index = 0;
  };

  class const_iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = Real<Scalar>;
    using difference_type = std::ptrdiff_t;
    using pointer = Real<Scalar>*;
    using const_reference = const Real<Scalar>&;

    constexpr const_iterator() noexcept = default;

    constexpr const_iterator(const RealSubmatrix<Mat>* mat, int index) noexcept
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
      iterator retval = *this;
      --(*this);
      return retval;
    }

    constexpr bool operator==(const const_iterator&) const noexcept = default;

    constexpr const_reference operator*() const noexcept {
      return (*m_mat)[m_index];
    }

   private:
    const RealSubmatrix<Mat>* m_mat = nullptr;
    int m_index = 0;
  };

  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

#endif  // DOXYGEN_SHOULD_SKIP_THIS

  /// Returns begin iterator.
  ///
  /// @return Begin iterator.
  iterator begin() { return iterator(this, 0); }

  /// Returns end iterator.
  ///
  /// @return End iterator.
  iterator end() { return iterator(this, rows() * cols()); }

  /// Returns begin iterator.
  ///
  /// @return Begin iterator.
  const_iterator begin() const { return const_iterator(this, 0); }

  /// Returns end iterator.
  ///
  /// @return End iterator.
  const_iterator end() const { return const_iterator(this, rows() * cols()); }

  /// Returns const begin iterator.
  ///
  /// @return Const begin iterator.
  const_iterator cbegin() const { return const_iterator(this, 0); }

  /// Returns const end iterator.
  ///
  /// @return Const end iterator.
  const_iterator cend() const { return const_iterator(this, rows() * cols()); }

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
  size_t size() const { return rows() * cols(); }

 private:
  Mat* m_mat = nullptr;

  Slice m_row_slice;
  int m_row_slice_length = 0;

  Slice m_col_slice;
  int m_col_slice_length = 0;
};

}  // namespace slp
