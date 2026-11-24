// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <complex>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/complex.hpp"
#include "sleipnir/autodiff/complex_submatrix.hpp"
#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/autodiff/real_matrix.hpp"
#include "sleipnir/autodiff/slice.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/function_ref.hpp"

namespace slp {

namespace detail {

/// Classifies an operand of complex matrix arithmetic.
///
/// @tparam T The operand type.
template <typename T>
struct ComplexMatrixOperandTraits {
  /// Whether T is a matrix operand.
  static constexpr bool is_matrix = false;
};

/// Classifies a ComplexMatrix operand.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
struct ComplexMatrixOperandTraits<ComplexMatrix<Scalar_>> {
  /// Whether T is a matrix operand.
  static constexpr bool is_matrix = true;
  /// Whether T holds autodiff expressions.
  static constexpr bool is_sleipnir = true;
  /// Whether T has complex entries.
  static constexpr bool is_complex = true;
  /// Whether T is an Eigen sparse matrix.
  static constexpr bool is_sparse = false;
  /// Scalar type of the entries' real and imaginary parts.
  using Scalar = Scalar_;
};

/// Classifies a ComplexSubmatrix operand.
///
/// @tparam Mat The type of the matrix the submatrix points to.
template <typename Mat>
struct ComplexMatrixOperandTraits<ComplexSubmatrix<Mat>>
    : ComplexMatrixOperandTraits<std::remove_cv_t<Mat>> {};

/// Classifies a real autodiff matrix operand.
///
/// @tparam T The operand type.
template <typename T>
  requires SleipnirMatrixLike<T, typename T::Scalar>
struct ComplexMatrixOperandTraits<T> {
  /// Whether T is a matrix operand.
  static constexpr bool is_matrix = true;
  /// Whether T holds autodiff expressions.
  static constexpr bool is_sleipnir = true;
  /// Whether T has complex entries.
  static constexpr bool is_complex = false;
  /// Whether T is an Eigen sparse matrix.
  static constexpr bool is_sparse = false;
  /// Scalar type of the entries' real and imaginary parts.
  using Scalar = typename T::Scalar;
};

/// Classifies an Eigen dense or sparse matrix operand.
///
/// @tparam T The operand type.
template <typename T>
  requires(std::derived_from<T, Eigen::MatrixBase<T>> ||
           std::derived_from<T, Eigen::SparseMatrixBase<T>>) &&
          (!std::is_void_v<typename T::Scalar>) &&
          (!SleipnirType<typename Eigen::NumTraits<typename T::Scalar>::Real>)
struct ComplexMatrixOperandTraits<T> {
  /// Whether T is a matrix operand.
  static constexpr bool is_matrix = true;
  /// Whether T holds autodiff expressions.
  static constexpr bool is_sleipnir = false;
  /// Whether T has complex entries.
  static constexpr bool is_complex =
      Eigen::NumTraits<typename T::Scalar>::IsComplex;
  /// Whether T is an Eigen sparse matrix.
  static constexpr bool is_sparse =
      std::derived_from<T, Eigen::SparseMatrixBase<T>>;
  /// Scalar type of the entries' real and imaginary parts.
  using Scalar = typename Eigen::NumTraits<typename T::Scalar>::Real;
};

/// Whether T is a matrix operand of complex matrix arithmetic with entries of
/// the given scalar type.
template <typename T, typename Scalar>
concept ComplexMatrixOperand =
    ComplexMatrixOperandTraits<std::decay_t<T>>::is_matrix &&
    std::same_as<typename ComplexMatrixOperandTraits<std::decay_t<T>>::Scalar,
                 Scalar>;

/// Whether T is a Sleipnir operand: a ComplexMatrix, a ComplexSubmatrix, a
/// real autodiff matrix, a Complex, or a Real.
template <typename T, typename Scalar>
concept SleipnirComplexOperand =
    std::same_as<std::decay_t<T>, ComplexMatrix<Scalar>> ||
    std::same_as<std::decay_t<T>, ComplexSubmatrix<ComplexMatrix<Scalar>>> ||
    std::same_as<std::decay_t<T>,
                 ComplexSubmatrix<const ComplexMatrix<Scalar>>> ||
    SleipnirMatrixLike<T, Scalar> || SleipnirComplex<T, Scalar> ||
    SleipnirScalarLike<T, Scalar>;

/// Whether T is a complex-valued operand.
template <typename T>
concept ComplexValued = ComplexLike<T>;

/// Whether two matrix operands combine into a ComplexMatrix. At least one
/// holds autodiff expressions and at least one has complex entries.
template <typename LHS, typename RHS, typename Scalar>
concept ComplexMatrixOperands =
    ComplexMatrixOperand<LHS, Scalar> && ComplexMatrixOperand<RHS, Scalar> &&
    (SleipnirComplexOperand<LHS, Scalar> ||
     SleipnirComplexOperand<RHS, Scalar>) &&
    (ComplexValued<LHS> || ComplexValued<RHS>);

/// Whether a matrix operand and a scalar operand combine into a ComplexMatrix.
/// At least one holds autodiff expressions and at least one is complex-valued.
template <typename M, typename S, typename Scalar>
concept ComplexMatrixScalarOperands =
    ComplexMatrixOperand<M, Scalar> && ComplexOperand<S, Scalar> &&
    (SleipnirComplexOperand<M, Scalar> || SleipnirComplexOperand<S, Scalar>) &&
    (ComplexValued<M> || ComplexValued<S>);

/// Returns the scalar type of a matrix operand.
template <typename T>
using complex_matrix_scalar_t =
    typename ComplexMatrixOperandTraits<std::decay_t<T>>::Scalar;

/// Returns entry (row, col) of a matrix operand as a Complex.
///
/// @tparam Scalar Scalar type.
/// @param m The matrix operand.
/// @param row The entry's row.
/// @param col The entry's column.
/// @return The entry.
template <typename Scalar, typename T>
Complex<Scalar> complex_entry(const T& m, int row, int col) {
  if constexpr (ComplexMatrixOperandTraits<T>::is_sparse) {
    return Complex<Scalar>{m.coeff(row, col)};
  } else {
    return Complex<Scalar>{m[row, col]};
  }
}

}  // namespace detail

/// A matrix of complex autodiff expressions.
///
/// Entries are Complex, and the solver sees two real decision variables per
/// complex decision variable. Arithmetic mixes freely with RealMatrix, Complex,
/// Real, and Eigen dense or sparse matrices of real or complex constants.
///
/// T() is the plain transpose, conj() is the elementwise conjugate, and H() is
/// the conjugate transpose. An equality constraint between complex matrices
/// constrains the real and imaginary parts of each entry. Complex numbers
/// aren't ordered, so inequality constraints are written on real-valued
/// matrices like real(), imag(), or abs2().
///
/// Slicing returns a ComplexSubmatrix that points to the matrix's entries, so
/// assigning to a slice assigns to the matrix.
///
/// @tparam Scalar_ Scalar type. std::complex is only specified for
///     floating-point types.
template <std::floating_point Scalar_>
class ComplexMatrix {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs an empty ComplexMatrix.
  ComplexMatrix() = default;

  /// Constructs a zero-initialized ComplexMatrix column vector with the given
  /// rows.
  ///
  /// @param rows The number of matrix rows.
  explicit ComplexMatrix(int rows) : ComplexMatrix{rows, 1} {}

  /// Constructs a zero-initialized ComplexMatrix with the given dimensions.
  ///
  /// @param rows The number of matrix rows.
  /// @param cols The number of matrix columns.
  ComplexMatrix(int rows, int cols) : m_rows{rows}, m_cols{cols} {
    m_storage.reserve(rows * cols);
    for (int index = 0; index < rows * cols; ++index) {
      m_storage.emplace_back();
    }
  }

  /// Constructs an empty ComplexMatrix with the given dimensions.
  ///
  /// @param rows The number of matrix rows.
  /// @param cols The number of matrix columns.
  ComplexMatrix(detail::empty_t, int rows, int cols)
      : m_rows{rows}, m_cols{cols} {
    m_storage.reserve(rows * cols);
    for (int index = 0; index < rows * cols; ++index) {
      m_storage.emplace_back(nullptr);
    }
  }

  /// Constructs a ComplexMatrix from a nested list of complex numbers.
  ///
  /// @param list The nested list of complex numbers.
  ComplexMatrix(
      std::initializer_list<std::initializer_list<Complex<Scalar>>> list) {
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

  /// Constructs a ComplexMatrix from a nested list of complex numbers.
  ///
  /// This overload is for Python bindings only.
  ///
  /// @param list The nested list of complex numbers.
  // NOLINTNEXTLINE (google-explicit-constructor)
  ComplexMatrix(const std::vector<std::vector<Complex<Scalar>>>& list) {
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

  /// Constructs a ComplexMatrix from its real and imaginary parts.
  ///
  /// @param real The real part.
  /// @param imag The imaginary part.
  ComplexMatrix(const RealMatrix<Scalar>& real, const RealMatrix<Scalar>& imag)
      : m_rows{real.rows()}, m_cols{real.cols()} {
    slp_assert(real.rows() == imag.rows() && real.cols() == imag.cols());

    m_storage.reserve(rows() * cols());
    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        m_storage.emplace_back(real[row, col], imag[row, col]);
      }
    }
  }

  /// Constructs a ComplexMatrix from a real matrix operand or an Eigen matrix
  /// of real or complex constants.
  ///
  /// @param values The matrix.
  template <typename T>
    requires detail::ComplexMatrixOperand<T, Scalar> &&
                 (!std::same_as<std::decay_t<T>, ComplexMatrix>)
  // NOLINTNEXTLINE (google-explicit-constructor)
  ComplexMatrix(const T& values)
      : m_rows{static_cast<int>(values.rows())},
        m_cols{static_cast<int>(values.cols())} {
    m_storage.reserve(rows() * cols());
    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        m_storage.emplace_back(detail::complex_entry<Scalar>(values, row, col));
      }
    }
  }

  /// Constructs a scalar ComplexMatrix from a Complex.
  ///
  /// @param value Complex.
  // NOLINTNEXTLINE (google-explicit-constructor)
  ComplexMatrix(Complex<Scalar> value) : m_rows{1}, m_cols{1} {
    m_storage.emplace_back(std::move(value));
  }

  /// Sets the ComplexMatrix's internal values.
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

  /// Returns the element at the given row and column.
  ///
  /// @param row The row.
  /// @param col The column.
  /// @return The element at the given row and column.
  Complex<Scalar>& operator[](int row, int col) {
    slp_assert(row >= 0 && row < rows());
    slp_assert(col >= 0 && col < cols());
    return m_storage[row * cols() + col];
  }

  /// Returns the element at the given row and column.
  ///
  /// @param row The row.
  /// @param col The column.
  /// @return The element at the given row and column.
  const Complex<Scalar>& operator[](int row, int col) const {
    slp_assert(row >= 0 && row < rows());
    slp_assert(col >= 0 && col < cols());
    return m_storage[row * cols() + col];
  }

  /// Returns the element at the given index.
  ///
  /// @param index The index.
  /// @return The element at the given index.
  Complex<Scalar>& operator[](int index) {
    slp_assert(index >= 0 && index < rows() * cols());
    return m_storage[index];
  }

  /// Returns the element at the given index.
  ///
  /// @param index The index.
  /// @return The element at the given index.
  const Complex<Scalar>& operator[](int index) const {
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
  ComplexSubmatrix<ComplexMatrix> block(int row_offset, int col_offset,
                                        int block_rows, int block_cols) {
    slp_assert(row_offset >= 0 && row_offset <= rows());
    slp_assert(col_offset >= 0 && col_offset <= cols());
    slp_assert(block_rows >= 0 && block_rows <= rows() - row_offset);
    slp_assert(block_cols >= 0 && block_cols <= cols() - col_offset);
    return ComplexSubmatrix{*this, row_offset, col_offset, block_rows,
                            block_cols};
  }

  /// Returns a block of the matrix.
  ///
  /// @param row_offset The row offset of the block selection.
  /// @param col_offset The column offset of the block selection.
  /// @param block_rows The number of rows in the block selection.
  /// @param block_cols The number of columns in the block selection.
  /// @return A block of the matrix.
  ComplexSubmatrix<const ComplexMatrix> block(int row_offset, int col_offset,
                                              int block_rows,
                                              int block_cols) const {
    slp_assert(row_offset >= 0 && row_offset <= rows());
    slp_assert(col_offset >= 0 && col_offset <= cols());
    slp_assert(block_rows >= 0 && block_rows <= rows() - row_offset);
    slp_assert(block_cols >= 0 && block_cols <= cols() - col_offset);
    return ComplexSubmatrix{*this, row_offset, col_offset, block_rows,
                            block_cols};
  }

  /// Returns a slice of the matrix.
  ///
  /// @param row_slice The row slice.
  /// @param col_slice The column slice.
  /// @return A slice of the matrix.
  ComplexSubmatrix<ComplexMatrix> operator[](Slice row_slice, Slice col_slice) {
    int row_slice_length = row_slice.adjust(rows());
    int col_slice_length = col_slice.adjust(cols());
    return ComplexSubmatrix{*this, std::move(row_slice), row_slice_length,
                            std::move(col_slice), col_slice_length};
  }

  /// Returns a slice of the matrix.
  ///
  /// @param row_slice The row slice.
  /// @param col_slice The column slice.
  /// @return A slice of the matrix.
  ComplexSubmatrix<const ComplexMatrix> operator[](Slice row_slice,
                                                   Slice col_slice) const {
    int row_slice_length = row_slice.adjust(rows());
    int col_slice_length = col_slice.adjust(cols());
    return ComplexSubmatrix{*this, std::move(row_slice), row_slice_length,
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
  ComplexSubmatrix<ComplexMatrix> operator[](Slice row_slice,
                                             int row_slice_length,
                                             Slice col_slice,
                                             int col_slice_length) {
    return ComplexSubmatrix{*this, std::move(row_slice), row_slice_length,
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
  ComplexSubmatrix<const ComplexMatrix> operator[](Slice row_slice,
                                                   int row_slice_length,
                                                   Slice col_slice,
                                                   int col_slice_length) const {
    return ComplexSubmatrix{*this, std::move(row_slice), row_slice_length,
                            std::move(col_slice), col_slice_length};
  }

  /// Returns a segment of the vector.
  ///
  /// @param offset The offset of the segment.
  /// @param length The length of the segment.
  /// @return A segment of the vector.
  ComplexSubmatrix<ComplexMatrix> segment(int offset, int length) {
    slp_assert(cols() == 1);
    return block(offset, 0, length, 1);
  }

  /// Returns a segment of the vector.
  ///
  /// @param offset The offset of the segment.
  /// @param length The length of the segment.
  /// @return A segment of the vector.
  ComplexSubmatrix<const ComplexMatrix> segment(int offset, int length) const {
    slp_assert(cols() == 1);
    return block(offset, 0, length, 1);
  }

  /// Returns a row slice of the matrix.
  ///
  /// @param row The row to slice.
  /// @return A row slice of the matrix.
  ComplexSubmatrix<ComplexMatrix> row(int row) {
    return block(row, 0, 1, cols());
  }

  /// Returns a row slice of the matrix.
  ///
  /// @param row The row to slice.
  /// @return A row slice of the matrix.
  ComplexSubmatrix<const ComplexMatrix> row(int row) const {
    return block(row, 0, 1, cols());
  }

  /// Returns a column slice of the matrix.
  ///
  /// @param col The column to slice.
  /// @return A column slice of the matrix.
  ComplexSubmatrix<ComplexMatrix> col(int col) {
    return block(0, col, rows(), 1);
  }

  /// Returns a column slice of the matrix.
  ///
  /// @param col The column to slice.
  /// @return A column slice of the matrix.
  ComplexSubmatrix<const ComplexMatrix> col(int col) const {
    return block(0, col, rows(), 1);
  }

  /// Compound matrix multiplication-assignment operator.
  ///
  /// @param rhs Complex to multiply.
  /// @return Result of multiplication.
  template <detail::ComplexMatrixOperand<Scalar> RHS>
  ComplexMatrix& operator*=(const RHS& rhs) {
    *this = *this * rhs;
    return *this;
  }

  /// Compound matrix-scalar multiplication-assignment operator.
  ///
  /// @param rhs Complex to multiply.
  /// @return Result of multiplication.
  template <detail::ComplexOperand<Scalar> RHS>
  ComplexMatrix& operator*=(const RHS& rhs) {
    for (auto& z : m_storage) {
      z = z * rhs;
    }
    return *this;
  }

  /// Compound matrix division-assignment operator.
  ///
  /// @param rhs Complex to divide.
  /// @return Result of division.
  template <detail::ComplexOperand<Scalar> RHS>
  ComplexMatrix& operator/=(const RHS& rhs) {
    for (auto& z : m_storage) {
      z = z / rhs;
    }
    return *this;
  }

  /// Compound addition-assignment operator.
  ///
  /// @param rhs Complex to add.
  /// @return Result of addition.
  template <detail::ComplexMatrixOperand<Scalar> RHS>
  ComplexMatrix& operator+=(const RHS& rhs) {
    slp_assert(rows() == rhs.rows() && cols() == rhs.cols());

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] =
            (*this)[row, col] + detail::complex_entry<Scalar>(rhs, row, col);
      }
    }

    return *this;
  }

  /// Compound subtraction-assignment operator.
  ///
  /// @param rhs Complex to subtract.
  /// @return Result of subtraction.
  template <detail::ComplexMatrixOperand<Scalar> RHS>
  ComplexMatrix& operator-=(const RHS& rhs) {
    slp_assert(rows() == rhs.rows() && cols() == rhs.cols());

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        (*this)[row, col] =
            (*this)[row, col] - detail::complex_entry<Scalar>(rhs, row, col);
      }
    }

    return *this;
  }

  /// Implicit conversion operator from 1x1 ComplexMatrix to Complex.
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator Complex<Scalar>() const {
    slp_assert(rows() == 1 && cols() == 1);
    return (*this)[0, 0];
  }

  /// Returns the transpose of the matrix.
  ///
  /// The entries aren't conjugated. For the conjugate transpose, use H().
  ///
  /// @return The transpose of the matrix.
  ComplexMatrix T() const {
    ComplexMatrix result{detail::empty, cols(), rows()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[col, row] = (*this)[row, col];
      }
    }

    return result;
  }

  /// Returns the elementwise complex conjugate of the matrix.
  ///
  /// @return The elementwise complex conjugate of the matrix.
  ComplexMatrix conj() const {
    return cwise_transform([](const Complex<Scalar>& z) { return z.conj(); });
  }

  /// Returns the conjugate transpose of the matrix.
  ///
  /// @return The conjugate transpose of the matrix.
  ComplexMatrix H() const {
    ComplexMatrix result{detail::empty, cols(), rows()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[col, row] = (*this)[row, col].conj();
      }
    }

    return result;
  }

  /// Returns the trace of the matrix, the sum of its diagonal.
  ///
  /// @return The trace of the matrix.
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
  std::complex<Scalar> value(int row, int col) {
    return (*this)[row, col].value();
  }

  /// Returns an element of the matrix.
  ///
  /// @param index The index of the element to return.
  /// @return An element of the matrix.
  std::complex<Scalar> value(int index) { return (*this)[index].value(); }

  /// Returns the contents of the matrix.
  ///
  /// @return The contents of the matrix.
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

    for (int index = 0; index < rows() * cols(); ++index) {
      result[index] = m_storage[index].real();
    }

    return result;
  }

  /// Returns the imaginary part of each element.
  ///
  /// @return The imaginary part of each element.
  RealMatrix<Scalar> imag() const {
    RealMatrix<Scalar> result{detail::empty, rows(), cols()};

    for (int index = 0; index < rows() * cols(); ++index) {
      result[index] = m_storage[index].imag();
    }

    return result;
  }

  /// Returns the squared magnitude of each element.
  ///
  /// @return The squared magnitude of each element.
  RealMatrix<Scalar> abs2() const {
    RealMatrix<Scalar> result{detail::empty, rows(), cols()};

    for (int index = 0; index < rows() * cols(); ++index) {
      result[index] = m_storage[index].abs2();
    }

    return result;
  }

  /// Returns the coefficient-wise product with another matrix.
  ///
  /// @param rhs The other matrix.
  /// @return The coefficient-wise product.
  template <detail::ComplexMatrixOperand<Scalar> RHS>
  ComplexMatrix cwise_product(const RHS& rhs) const {
    slp_assert(rows() == rhs.rows() && cols() == rhs.cols());

    ComplexMatrix result{detail::empty, rows(), cols()};

    for (int row = 0; row < rows(); ++row) {
      for (int col = 0; col < cols(); ++col) {
        result[row, col] =
            (*this)[row, col] * detail::complex_entry<Scalar>(rhs, row, col);
      }
    }

    return result;
  }

  /// Transforms the matrix coefficient-wise with an unary operator.
  ///
  /// @param unary_op The unary operator to use for the transform operation.
  /// @return Result of the unary operator.
  ComplexMatrix cwise_transform(
      function_ref<Complex<Scalar>(const Complex<Scalar>& z)> unary_op) const {
    ComplexMatrix result{detail::empty, rows(), cols()};

    for (int index = 0; index < rows() * cols(); ++index) {
      result[index] = unary_op(m_storage[index]);
    }

    return result;
  }

  /// Returns begin iterator.
  ///
  /// @return Begin iterator.
  auto begin() { return m_storage.begin(); }

  /// Returns end iterator.
  ///
  /// @return End iterator.
  auto end() { return m_storage.end(); }

  /// Returns const begin iterator.
  ///
  /// @return Const begin iterator.
  auto begin() const { return m_storage.begin(); }

  /// Returns const end iterator.
  ///
  /// @return Const end iterator.
  auto end() const { return m_storage.end(); }

  /// Returns const begin iterator.
  ///
  /// @return Const begin iterator.
  auto cbegin() const { return m_storage.cbegin(); }

  /// Returns const end iterator.
  ///
  /// @return Const end iterator.
  auto cend() const { return m_storage.cend(); }

  /// Returns number of elements in matrix.
  ///
  /// @return Number of elements in matrix.
  size_t size() const { return m_storage.size(); }

  /// Returns an identity matrix.
  ///
  /// @param rows The number of matrix rows.
  /// @return An identity matrix.
  static ComplexMatrix identity(int rows) {
    ComplexMatrix result{rows, rows};
    for (int i = 0; i < rows; ++i) {
      result[i, i] = Complex<Scalar>{Scalar(1)};
    }
    return result;
  }

  /// Returns a matrix filled with zeros.
  ///
  /// @param rows The number of matrix rows.
  /// @param cols The number of matrix columns.
  /// @return A matrix filled with zeros.
  static ComplexMatrix zero(int rows, int cols) {
    return ComplexMatrix{rows, cols};
  }

 private:
  gch::small_vector<Complex<Scalar>> m_storage;
  int m_rows = 0;
  int m_cols = 0;
};

template <typename Derived>
  requires(Eigen::NumTraits<typename Derived::Scalar>::IsComplex != 0)
ComplexMatrix(const Eigen::MatrixBase<Derived>&)
    -> ComplexMatrix<typename Eigen::NumTraits<typename Derived::Scalar>::Real>;

template <typename Scalar>
ComplexMatrix(const RealMatrix<Scalar>&, const RealMatrix<Scalar>&)
    -> ComplexMatrix<Scalar>;

/// Matrix multiplication operator.
///
/// Products with Eigen constants skip their zero entries, so a sparse constant
/// matrix like a bus admittance matrix only builds expressions for its
/// nonzeros.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return Result of multiplication.
template <typename LHS, typename RHS,
          typename Scalar = detail::complex_matrix_scalar_t<LHS>>
  requires detail::ComplexMatrixOperands<LHS, RHS, Scalar>
ComplexMatrix<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
  using LhsTraits = detail::ComplexMatrixOperandTraits<LHS>;
  using RhsTraits = detail::ComplexMatrixOperandTraits<RHS>;

  slp_assert(lhs.cols() == rhs.rows());

  ComplexMatrix<Scalar> result{static_cast<int>(lhs.rows()),
                               static_cast<int>(rhs.cols())};

  // Accumulates lhs(i, k) rhs(k, j) into result(i, j)
  auto accumulate = [&](int i, int j, const auto& lhs_ik, const auto& rhs_kj) {
    result[i, j] = result[i, j] + lhs_ik * rhs_kj;
  };

  if constexpr (LhsTraits::is_sparse) {
    // Binds without a copy if lhs is already a column-major SparseMatrix
    using Sparse = Eigen::SparseMatrix<typename LHS::Scalar>;
    const Sparse& A = lhs;
    for (int outer = 0; outer < A.outerSize(); ++outer) {
      for (typename Sparse::InnerIterator it{A, outer}; it; ++it) {
        int i = static_cast<int>(it.row());
        int k = static_cast<int>(it.col());
        for (int j = 0; j < rhs.cols(); ++j) {
          accumulate(i, j, Complex<Scalar>{it.value()},
                     detail::complex_entry<Scalar>(rhs, k, j));
        }
      }
    }
  } else if constexpr (RhsTraits::is_sparse) {
    // Binds without a copy if rhs is already a column-major SparseMatrix
    using Sparse = Eigen::SparseMatrix<typename RHS::Scalar>;
    const Sparse& B = rhs;
    for (int outer = 0; outer < B.outerSize(); ++outer) {
      for (typename Sparse::InnerIterator it{B, outer}; it; ++it) {
        int k = static_cast<int>(it.row());
        int j = static_cast<int>(it.col());
        for (int i = 0; i < lhs.rows(); ++i) {
          accumulate(i, j, detail::complex_entry<Scalar>(lhs, i, k),
                     Complex<Scalar>{it.value()});
        }
      }
    }
  } else {
    for (int i = 0; i < lhs.rows(); ++i) {
      for (int k = 0; k < lhs.cols(); ++k) {
        if constexpr (!LhsTraits::is_sleipnir) {
          if (lhs[i, k] == typename LHS::Scalar{}) {
            continue;
          }
        }
        for (int j = 0; j < rhs.cols(); ++j) {
          if constexpr (!RhsTraits::is_sleipnir) {
            if (rhs[k, j] == typename RHS::Scalar{}) {
              continue;
            }
          }
          accumulate(i, j, detail::complex_entry<Scalar>(lhs, i, k),
                     detail::complex_entry<Scalar>(rhs, k, j));
        }
      }
    }
  }

  return result;
}

/// Matrix-scalar multiplication operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return Result of multiplication.
template <typename LHS, typename RHS,
          typename Scalar = detail::complex_matrix_scalar_t<LHS>>
  requires detail::ComplexMatrixScalarOperands<LHS, RHS, Scalar>
ComplexMatrix<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
  ComplexMatrix<Scalar> result{detail::empty, static_cast<int>(lhs.rows()),
                               static_cast<int>(lhs.cols())};

  for (int row = 0; row < result.rows(); ++row) {
    for (int col = 0; col < result.cols(); ++col) {
      result[row, col] = detail::complex_entry<Scalar>(lhs, row, col) * rhs;
    }
  }

  return result;
}

/// Scalar-matrix multiplication operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return Result of multiplication.
template <typename LHS, typename RHS,
          typename Scalar = detail::complex_matrix_scalar_t<RHS>>
  requires detail::ComplexMatrixScalarOperands<RHS, LHS, Scalar>
ComplexMatrix<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
  return rhs * lhs;
}

/// Binary division operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return Result of division.
template <typename LHS, typename RHS,
          typename Scalar = detail::complex_matrix_scalar_t<LHS>>
  requires detail::ComplexMatrixScalarOperands<LHS, RHS, Scalar>
ComplexMatrix<Scalar> operator/(const LHS& lhs, const RHS& rhs) {
  ComplexMatrix<Scalar> result{detail::empty, static_cast<int>(lhs.rows()),
                               static_cast<int>(lhs.cols())};

  for (int row = 0; row < result.rows(); ++row) {
    for (int col = 0; col < result.cols(); ++col) {
      result[row, col] = detail::complex_entry<Scalar>(lhs, row, col) / rhs;
    }
  }

  return result;
}

/// Binary addition operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return Result of addition.
template <typename LHS, typename RHS,
          typename Scalar = detail::complex_matrix_scalar_t<LHS>>
  requires detail::ComplexMatrixOperands<LHS, RHS, Scalar>
ComplexMatrix<Scalar> operator+(const LHS& lhs, const RHS& rhs) {
  ComplexMatrix<Scalar> result{lhs};
  result += rhs;
  return result;
}

/// Binary subtraction operator.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return Result of subtraction.
template <typename LHS, typename RHS,
          typename Scalar = detail::complex_matrix_scalar_t<LHS>>
  requires detail::ComplexMatrixOperands<LHS, RHS, Scalar>
ComplexMatrix<Scalar> operator-(const LHS& lhs, const RHS& rhs) {
  ComplexMatrix<Scalar> result{lhs};
  result -= rhs;
  return result;
}

/// Unary minus operator.
///
/// @param z Operand for unary minus.
/// @return The negated matrix.
template <typename Scalar>
ComplexMatrix<Scalar> operator-(const ComplexMatrix<Scalar>& z) {
  return z.cwise_transform([](const Complex<Scalar>& entry) { return -entry; });
}

/// Unary plus operator.
///
/// @param z Operand for unary plus.
/// @return The matrix.
template <typename Scalar>
ComplexMatrix<Scalar> operator+(const ComplexMatrix<Scalar>& z) {
  return z;
}

/// Equality operator that returns equality constraints on the real and
/// imaginary parts of each entry of two matrices.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraints.
template <typename LHS, typename RHS,
          typename Scalar = detail::complex_matrix_scalar_t<LHS>>
  requires detail::ComplexMatrixOperands<LHS, RHS, Scalar>
EqualityConstraints<Scalar> operator==(const LHS& lhs, const RHS& rhs) {
  slp_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());

  ComplexMatrix<Scalar> l{lhs};
  ComplexMatrix<Scalar> r{rhs};
  return EqualityConstraints<Scalar>{l.real() == r.real(),
                                     l.imag() == r.imag()};
}

/// Equality operator that returns equality constraints on the real and
/// imaginary parts of each entry of a matrix and a scalar.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraints.
template <typename LHS, typename RHS,
          typename Scalar = detail::complex_matrix_scalar_t<LHS>>
  requires detail::ComplexMatrixScalarOperands<LHS, RHS, Scalar>
EqualityConstraints<Scalar> operator==(const LHS& lhs, const RHS& rhs) {
  ComplexMatrix<Scalar> l{lhs};
  Complex<Scalar> r{rhs};
  return EqualityConstraints<Scalar>{l.real() == r.real(),
                                     l.imag() == r.imag()};
}

/// Equality operator that returns equality constraints on the real and
/// imaginary parts of each entry of a scalar and a matrix.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraints.
template <typename LHS, typename RHS,
          typename Scalar = detail::complex_matrix_scalar_t<RHS>>
  requires detail::ComplexMatrixScalarOperands<RHS, LHS, Scalar>
EqualityConstraints<Scalar> operator==(const LHS& lhs, const RHS& rhs) {
  return rhs == lhs;
}

}  // namespace slp
