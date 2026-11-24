/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define MKD_EXPAND(x)                                      x
#define MKD_COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define MKD_VA_SIZE(...)                                   MKD_EXPAND(MKD_COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1, 0))
#define MKD_CAT1(a, b)                                     a ## b
#define MKD_CAT2(a, b)                                     MKD_CAT1(a, b)
#define MKD_DOC1(n1)                                       mkd_doc_##n1
#define MKD_DOC2(n1, n2)                                   mkd_doc_##n1##_##n2
#define MKD_DOC3(n1, n2, n3)                               mkd_doc_##n1##_##n2##_##n3
#define MKD_DOC4(n1, n2, n3, n4)                           mkd_doc_##n1##_##n2##_##n3##_##n4
#define MKD_DOC5(n1, n2, n3, n4, n5)                       mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5
#define MKD_DOC6(n1, n2, n3, n4, n5, n6)                   mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define MKD_DOC7(n1, n2, n3, n4, n5, n6, n7)               mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                           MKD_EXPAND(MKD_EXPAND(MKD_CAT2(MKD_DOC, MKD_VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


constexpr char mkd_doc_formatter[] = R"doc(Formatter for ExpressionType.)doc";

constexpr char mkd_doc_formatter_2[] = R"doc(Formatter for LintFinding.)doc";

constexpr char mkd_doc_formatter_3[] = R"doc(Formatter for ExitStatus.)doc";

constexpr char mkd_doc_formatter_format[] =
R"doc(Formats ExpressionType.

Args:
    type: Expression type.
    ctx: Format context.

Template Args:
    FmtContext: Format context type.

Returns:
    Format context iterator.

)doc";

constexpr char mkd_doc_formatter_format_2[] =
R"doc(Formats LintFinding as its message.

Args:
    finding: Lint finding.
    ctx: Format context.

Template Args:
    FmtContext: Format context type.

Returns:
    Format context iterator.

)doc";

constexpr char mkd_doc_formatter_format_3[] =
R"doc(Formats ExitStatus.

Args:
    exit_status: Exit status.
    ctx: Format context.

Template Args:
    FmtContext: Format context type.

Returns:
    Format context iterator.

)doc";

constexpr char mkd_doc_formatter_m_underlying[] = R"doc()doc";

constexpr char mkd_doc_formatter_m_underlying_2[] = R"doc()doc";

constexpr char mkd_doc_formatter_m_underlying_3[] = R"doc()doc";

constexpr char mkd_doc_formatter_parse[] =
R"doc(Parse format string.

Args:
    ctx: Format parse context.

Returns:
    Format parse context iterator.

)doc";

constexpr char mkd_doc_formatter_parse_2[] =
R"doc(Parses format string.

Args:
    ctx: Format parse context.

Returns:
    Format parse context iterator.

)doc";

constexpr char mkd_doc_formatter_parse_3[] =
R"doc(Parses format string.

Args:
    ctx: Format parse context.

Returns:
    Format parse context iterator.

)doc";

constexpr char mkd_doc_slp[] = R"doc()doc";

constexpr char mkd_doc_slp_2[] = R"doc()doc";

constexpr char mkd_doc_slp_3[] = R"doc()doc";

constexpr char mkd_doc_slp_4[] = R"doc()doc";

constexpr char mkd_doc_slp_5[] = R"doc()doc";

constexpr char mkd_doc_slp_6[] = R"doc()doc";

constexpr char mkd_doc_slp_7[] = R"doc()doc";

constexpr char mkd_doc_slp_8[] = R"doc()doc";

constexpr char mkd_doc_slp_9[] = R"doc()doc";

constexpr char mkd_doc_slp_Complex[] =
R"doc(A complex number of autodiff expressions.

A complex number z = a + bi stores its real part a and imaginary part
b as two Reals, so the solver sees two real decision variables per
complex decision variable. Costs must be real-valued, so a cost is
written with real(), imag(), abs2(), or similar.

An equality constraint between complex numbers constrains the real and
imaginary parts separately. Complex numbers aren't ordered, so
inequality constraints are written on real-valued expressions like
abs2() or real().

Complex numbers are created by Problem::complex_decision_variable(),
from constants, and from expressions.

Template Args:
    Scalar_: Scalar type. std::complex is only specified for floating-
             point types.)doc";

constexpr char mkd_doc_slp_ComplexMatrix[] =
R"doc(A matrix of complex autodiff expressions.

Entries are Complex, and the solver sees two real decision variables
per complex decision variable. Arithmetic mixes freely with
RealMatrix, Complex, Real, and Eigen dense or sparse matrices of real
or complex constants.

T() is the plain transpose, conj() is the elementwise conjugate, and
H() is the conjugate transpose. An equality constraint between complex
matrices constrains the real and imaginary parts of each entry.
Complex numbers aren't ordered, so inequality constraints are written
on real-valued matrices like real(), imag(), or abs2().

Slicing returns a ComplexSubmatrix that points to the matrix's
entries, so assigning to a slice assigns to the matrix.

Template Args:
    Scalar_: Scalar type. std::complex is only specified for floating-
             point types.)doc";

constexpr char mkd_doc_slp_ComplexMatrix_ComplexMatrix[] = R"doc(Constructs an empty ComplexMatrix.)doc";

constexpr char mkd_doc_slp_ComplexMatrix_ComplexMatrix_2[] =
R"doc(Constructs a zero-initialized ComplexMatrix column vector with the
given rows.

Args:
    rows: The number of matrix rows.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_ComplexMatrix_3[] =
R"doc(Constructs a zero-initialized ComplexMatrix with the given dimensions.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_ComplexMatrix_4[] =
R"doc(Constructs an empty ComplexMatrix with the given dimensions.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_ComplexMatrix_5[] =
R"doc(Constructs a ComplexMatrix from a nested list of complex numbers.

Args:
    list: The nested list of complex numbers.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_ComplexMatrix_6[] =
R"doc(Constructs a ComplexMatrix from a nested list of complex numbers.

This overload is for Python bindings only.

Args:
    list: The nested list of complex numbers.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_ComplexMatrix_7[] =
R"doc(Constructs a ComplexMatrix from its real and imaginary parts.

Args:
    real: The real part.
    imag: The imaginary part.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_ComplexMatrix_8[] =
R"doc(Constructs a ComplexMatrix from a real matrix operand or an Eigen
matrix of real or complex constants.

Args:
    values: The matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_ComplexMatrix_9[] =
R"doc(Constructs a scalar ComplexMatrix from a Complex.

Args:
    value: Complex.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_H[] =
R"doc(Returns the conjugate transpose of the matrix.

Returns:
    The conjugate transpose of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_T[] =
R"doc(Returns the transpose of the matrix.

The entries aren't conjugated. For the conjugate transpose, use H().

Returns:
    The transpose of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_abs2[] =
R"doc(Returns the squared magnitude of each element.

Returns:
    The squared magnitude of each element.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_begin[] =
R"doc(Returns begin iterator.

Returns:
    Begin iterator.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_begin_2[] =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_block[] =
R"doc(Returns a block of the matrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_block_2[] =
R"doc(Returns a block of the matrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_cbegin[] =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_cend[] =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_col[] =
R"doc(Returns a column slice of the matrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_col_2[] =
R"doc(Returns a column slice of the matrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_cols[] =
R"doc(Returns the number of columns in the matrix.

Returns:
    The number of columns in the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_conj[] =
R"doc(Returns the elementwise complex conjugate of the matrix.

Returns:
    The elementwise complex conjugate of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_cwise_product[] =
R"doc(Returns the coefficient-wise product with another matrix.

Args:
    rhs: The other matrix.

Returns:
    The coefficient-wise product.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_cwise_transform[] =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Args:
    unary_op: The unary operator to use for the transform operation.

Returns:
    Result of the unary operator.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_end[] =
R"doc(Returns end iterator.

Returns:
    End iterator.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_end_2[] =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_identity[] =
R"doc(Returns an identity matrix.

Args:
    rows: The number of matrix rows.

Returns:
    An identity matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_imag[] =
R"doc(Returns the imaginary part of each element.

Returns:
    The imaginary part of each element.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_m_cols[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexMatrix_m_rows[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexMatrix_m_storage[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_array[] =
R"doc(Returns the element at the given row and column.

Args:
    row: The row.
    col: The column.

Returns:
    The element at the given row and column.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_array_2[] =
R"doc(Returns the element at the given row and column.

Args:
    row: The row.
    col: The column.

Returns:
    The element at the given row and column.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_array_3[] =
R"doc(Returns the element at the given index.

Args:
    index: The index.

Returns:
    The element at the given index.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_array_4[] =
R"doc(Returns the element at the given index.

Args:
    index: The index.

Returns:
    The element at the given index.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_array_5[] =
R"doc(Returns a slice of the matrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_array_6[] =
R"doc(Returns a slice of the matrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_array_7[] =
R"doc(Returns a slice of the matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_array_8[] =
R"doc(Returns a slice of the matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_iadd[] =
R"doc(Compound addition-assignment operator.

Args:
    rhs: Complex to add.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_idiv[] =
R"doc(Compound matrix division-assignment operator.

Args:
    rhs: Complex to divide.

Returns:
    Result of division.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_imul[] =
R"doc(Compound matrix multiplication-assignment operator.

Args:
    rhs: Complex to multiply.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_imul_2[] =
R"doc(Compound matrix-scalar multiplication-assignment operator.

Args:
    rhs: Complex to multiply.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_isub[] =
R"doc(Compound subtraction-assignment operator.

Args:
    rhs: Complex to subtract.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_operator_slp_Complex[] = R"doc(Implicit conversion operator from 1x1 ComplexMatrix to Complex.)doc";

constexpr char mkd_doc_slp_ComplexMatrix_real[] =
R"doc(Returns the real part of each element.

Returns:
    The real part of each element.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_row[] =
R"doc(Returns a row slice of the matrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_row_2[] =
R"doc(Returns a row slice of the matrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_rows[] =
R"doc(Returns the number of rows in the matrix.

Returns:
    The number of rows in the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_segment[] =
R"doc(Returns a segment of the vector.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the vector.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_segment_2[] =
R"doc(Returns a segment of the vector.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the vector.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_set_value[] =
R"doc(Sets the ComplexMatrix's internal values.

Real values set the imaginary parts to zero.

Args:
    values: Eigen matrix of values.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_size[] =
R"doc(Returns number of elements in matrix.

Returns:
    Number of elements in matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_trace[] =
R"doc(Returns the trace of the matrix, the sum of its diagonal.

Returns:
    The trace of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_value[] =
R"doc(Returns an element of the matrix.

Args:
    row: The row of the element to return.
    col: The column of the element to return.

Returns:
    An element of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_value_2[] =
R"doc(Returns an element of the matrix.

Args:
    index: The index of the element to return.

Returns:
    An element of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_value_3[] =
R"doc(Returns the contents of the matrix.

Returns:
    The contents of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexMatrix_zero[] =
R"doc(Returns a matrix filled with zeros.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.

Returns:
    A matrix filled with zeros.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix[] =
R"doc(A submatrix of complex autodiff expressions with reference semantics.

Assigning to a ComplexSubmatrix assigns to the entries of the matrix
it points to.

Template Args:
    Mat: The type of the matrix whose storage this class points to.)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_ComplexSubmatrix[] = R"doc(Copy constructor.)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_ComplexSubmatrix_2[] = R"doc(Move constructor.)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_ComplexSubmatrix_3[] =
R"doc(Constructs a ComplexSubmatrix pointing to all of the given matrix.

Args:
    mat: The matrix to which to point.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_ComplexSubmatrix_4[] =
R"doc(Constructs a ComplexSubmatrix pointing to a subset of the given
matrix.

Args:
    mat: The matrix to which to point.
    row_offset: The block's row offset.
    col_offset: The block's column offset.
    block_rows: The number of rows in the block.
    block_cols: The number of columns in the block.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_ComplexSubmatrix_5[] =
R"doc(Constructs a ComplexSubmatrix pointing to a subset of the given
matrix.

Note that the slices are taken as is rather than adjusted.

Args:
    mat: The matrix to which to point.
    row_slice: The block's row slice.
    row_slice_length: The block's row length.
    col_slice: The block's column slice.
    col_slice_length: The block's column length.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_H[] =
R"doc(Returns the conjugate transpose of the submatrix.

Returns:
    The conjugate transpose of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_T[] =
R"doc(Returns the transpose of the submatrix.

The entries aren't conjugated. For the conjugate transpose, use H().

Returns:
    The transpose of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_abs2[] =
R"doc(Returns the squared magnitude of each element.

Returns:
    The squared magnitude of each element.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_begin[] =
R"doc(Returns begin iterator.

Returns:
    Begin iterator.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_begin_2[] =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_block[] =
R"doc(Returns a block of the submatrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_block_2[] =
R"doc(Returns a block of the submatrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_cbegin[] =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_cend[] =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_col[] =
R"doc(Returns a column slice of the submatrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_col_2[] =
R"doc(Returns a column slice of the submatrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_cols[] =
R"doc(Returns the number of columns in the matrix.

Returns:
    The number of columns in the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_compose[] =
R"doc(Returns a slice of this submatrix's slice in the matrix's indices.

Args:
    outer: This submatrix's slice of the matrix.
    inner: The slice of this submatrix.

Returns:
    The slice of the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_conj[] =
R"doc(Returns the elementwise complex conjugate of the submatrix.

Returns:
    The elementwise complex conjugate of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_const_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_const_iterator_const_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_const_iterator_const_iterator_2[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_const_iterator_m_index[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_const_iterator_m_mat[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_const_iterator_operator_dec[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_const_iterator_operator_dec_2[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_const_iterator_operator_eq[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_const_iterator_operator_inc[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_const_iterator_operator_inc_2[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_const_iterator_operator_mul[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_cwise_product[] =
R"doc(Returns the coefficient-wise product with another matrix.

Args:
    rhs: The other matrix.

Returns:
    The coefficient-wise product.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_cwise_transform[] =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Args:
    unary_op: The unary operator to use for the transform operation.

Returns:
    Result of the unary operator.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_end[] =
R"doc(Returns end iterator.

Returns:
    End iterator.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_end_2[] =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_imag[] =
R"doc(Returns the imaginary part of each element.

Returns:
    The imaginary part of each element.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_iterator_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_iterator_iterator_2[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_iterator_m_index[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_iterator_m_mat[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_iterator_operator_dec[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_iterator_operator_dec_2[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_iterator_operator_eq[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_iterator_operator_inc[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_iterator_operator_inc_2[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_iterator_operator_mul[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_m_col_slice[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_m_col_slice_length[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_m_mat[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_m_row_slice[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_m_row_slice_length[] = R"doc()doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_array[] =
R"doc(Returns the element at the given row and column.

Args:
    row: The row.
    col: The column.

Returns:
    The element at the given row and column.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_array_2[] =
R"doc(Returns the element at the given row and column.

Args:
    row: The row.
    col: The column.

Returns:
    The element at the given row and column.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_array_3[] =
R"doc(Returns the element at the given index.

Args:
    index: The index.

Returns:
    The element at the given index.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_array_4[] =
R"doc(Returns the element at the given index.

Args:
    index: The index.

Returns:
    The element at the given index.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_array_5[] =
R"doc(Returns a slice of the submatrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_array_6[] =
R"doc(Returns a slice of the submatrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_array_7[] =
R"doc(Returns a slice of the submatrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_array_8[] =
R"doc(Returns a slice of the submatrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_assign[] =
R"doc(Assigns a ComplexSubmatrix to the block.

Args:
    values: ComplexSubmatrix of values.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_assign_2[] =
R"doc(Assigns a ComplexSubmatrix to the block.

Args:
    values: ComplexSubmatrix of values.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_assign_3[] =
R"doc(Assigns a scalar to the block.

This only works for blocks with one row and one column.

Args:
    value: Value to assign.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_assign_4[] =
R"doc(Assigns an Eigen matrix of real or complex constants to the block.

Args:
    values: Eigen matrix of values to assign.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_assign_5[] =
R"doc(Assigns a ComplexMatrix to the block.

Args:
    values: ComplexMatrix of values.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_assign_6[] =
R"doc(Assigns a ComplexMatrix to the block.

Args:
    values: ComplexMatrix of values.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_iadd[] =
R"doc(Compound addition-assignment operator.

Args:
    rhs: Matrix to add.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_idiv[] =
R"doc(Compound division-assignment operator.

Args:
    rhs: Scalar to divide.

Returns:
    Result of division.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_imul[] =
R"doc(Compound multiplication-assignment operator.

Args:
    rhs: Matrix or scalar to multiply.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_isub[] =
R"doc(Compound subtraction-assignment operator.

Args:
    rhs: Matrix to subtract.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_operator_slp_Complex[] = R"doc(Implicit conversion operator from 1x1 ComplexSubmatrix to Complex.)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_real[] =
R"doc(Returns the real part of each element.

Returns:
    The real part of each element.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_row[] =
R"doc(Returns a row slice of the submatrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_row_2[] =
R"doc(Returns a row slice of the submatrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_rows[] =
R"doc(Returns the number of rows in the matrix.

Returns:
    The number of rows in the matrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_segment[] =
R"doc(Returns a segment of the submatrix.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_segment_2[] =
R"doc(Returns a segment of the submatrix.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_set_value[] =
R"doc(Sets the block's internal value.

This only works for blocks with one row and one column.

Args:
    value: Value to assign.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_set_value_2[] =
R"doc(Sets the block's internal values.

Real values set the imaginary parts to zero.

Args:
    values: Eigen matrix of values.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_size[] =
R"doc(Returns number of elements in matrix.

Returns:
    Number of elements in matrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_trace[] =
R"doc(Returns the trace of the submatrix, the sum of its diagonal.

Returns:
    The trace of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_value[] =
R"doc(Returns an element of the submatrix.

Args:
    row: The row of the element to return.
    col: The column of the element to return.

Returns:
    An element of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_value_2[] =
R"doc(Returns an element of the submatrix.

Args:
    index: The index of the element to return.

Returns:
    An element of the submatrix.

)doc";

constexpr char mkd_doc_slp_ComplexSubmatrix_value_3[] =
R"doc(Returns the contents of the submatrix.

Returns:
    The contents of the submatrix.

)doc";

constexpr char mkd_doc_slp_Complex_Complex[] = R"doc(Constructs a complex zero.)doc";

constexpr char mkd_doc_slp_Complex_Complex_2[] = R"doc(Constructs an empty complex number.)doc";

constexpr char mkd_doc_slp_Complex_Complex_3[] =
R"doc(Constructs a complex number from its real and imaginary parts.

Args:
    real: The real part.
    imag: The imaginary part.

)doc";

constexpr char mkd_doc_slp_Complex_Complex_4[] =
R"doc(Constructs a complex number with a zero imaginary part.

Args:
    real: The real part.

)doc";

constexpr char mkd_doc_slp_Complex_Complex_5[] =
R"doc(Constructs a constant complex number.

Args:
    value: The value.

)doc";

constexpr char mkd_doc_slp_Complex_abs[] =
R"doc(Returns the magnitude √(a² + b²).

The magnitude isn't differentiable at zero. For constraints and costs
that can reach zero, use abs2().

Returns:
    The magnitude.

)doc";

constexpr char mkd_doc_slp_Complex_abs2[] =
R"doc(Returns the squared magnitude a² + b².

Returns:
    The squared magnitude.

)doc";

constexpr char mkd_doc_slp_Complex_arg[] =
R"doc(Returns the angle atan2(b, a) in radians.

The angle isn't differentiable at zero or continuous across the
negative real axis.

Returns:
    The angle.

)doc";

constexpr char mkd_doc_slp_Complex_conj[] =
R"doc(Returns the complex conjugate a − bi.

Returns:
    The complex conjugate.

)doc";

constexpr char mkd_doc_slp_Complex_imag[] =
R"doc(Returns the imaginary part.

Returns:
    The imaginary part.

)doc";

constexpr char mkd_doc_slp_Complex_imag_2[] =
R"doc(Returns the imaginary part.

Returns:
    The imaginary part.

)doc";

constexpr char mkd_doc_slp_Complex_m_imag[] = R"doc()doc";

constexpr char mkd_doc_slp_Complex_m_real[] = R"doc()doc";

constexpr char mkd_doc_slp_Complex_polar[] =
R"doc(Returns a complex number from its polar form r e^{iθ}.

Args:
    magnitude: The magnitude r.
    angle: The angle θ in radians.

Returns:
    The complex number.

)doc";

constexpr char mkd_doc_slp_Complex_real[] =
R"doc(Returns the real part.

Returns:
    The real part.

)doc";

constexpr char mkd_doc_slp_Complex_real_2[] =
R"doc(Returns the real part.

Returns:
    The real part.

)doc";

constexpr char mkd_doc_slp_Complex_set_value[] =
R"doc(Sets the complex number's value.

This is only meaningful for complex decision variables and parameters.

Args:
    value: The value.

)doc";

constexpr char mkd_doc_slp_Complex_value[] =
R"doc(Returns the complex number's value.

Returns:
    The complex number's value.

)doc";

constexpr char mkd_doc_slp_DynamicsType[] = R"doc(Enum describing a type of system dynamics constraints.)doc";

constexpr char mkd_doc_slp_DynamicsType_DISCRETE[] = R"doc(The dynamics are a function in the form xₖ₊₁ = f(t, xₖ, uₖ).)doc";

constexpr char mkd_doc_slp_DynamicsType_EXPLICIT_ODE[] = R"doc(The dynamics are a function in the form dx/dt = f(t, x, u).)doc";

constexpr char mkd_doc_slp_EqualityConstraints[] =
R"doc(A vector of equality constraints of the form cₑ(x) = 0.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_EqualityConstraints_2[] =
R"doc(A vector of equality constraints of the form cₑ(x) = 0.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_EqualityConstraints_EqualityConstraints[] =
R"doc(Concatenates multiple equality constraints.

Args:
    equality_constraints: The list of EqualityConstraints to
                          concatenate.

)doc";

constexpr char mkd_doc_slp_EqualityConstraints_EqualityConstraints_2[] =
R"doc(Concatenates multiple equality constraints.

This overload is for Python bindings only.

Args:
    equality_constraints: The list of EqualityConstraints to
                          concatenate.

)doc";

constexpr char mkd_doc_slp_EqualityConstraints_EqualityConstraints_3[] =
R"doc(Constructs an equality constraint from a left and right side.

The standard form for equality constraints is c(x) = 0. This function
takes a constraint of the form lhs = rhs and converts it to lhs - rhs
= 0.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side.

)doc";

constexpr char mkd_doc_slp_EqualityConstraints_constraints[] = R"doc(A vector of scalar equality constraints.)doc";

constexpr char mkd_doc_slp_EqualityConstraints_operator_bool[] = R"doc(Implicit conversion operator to bool.)doc";

constexpr char mkd_doc_slp_ExitStatus[] = R"doc(Solver exit status. Negative values indicate failure.)doc";

constexpr char mkd_doc_slp_ExitStatus_CALLBACK_REQUESTED_STOP[] =
R"doc(The solver returned its solution so far after the user requested a
stop.)doc";

constexpr char mkd_doc_slp_ExitStatus_DIVERGING_ITERATES[] =
R"doc(The solver encountered diverging primal iterates xₖ and/or sₖ and gave
up.)doc";

constexpr char mkd_doc_slp_ExitStatus_FACTORIZATION_FAILED[] = R"doc(The linear system factorization failed.)doc";

constexpr char mkd_doc_slp_ExitStatus_FEASIBILITY_RESTORATION_FAILED[] =
R"doc(The solver failed to reach the desired tolerance, and feasibility
restoration failed to converge.)doc";

constexpr char mkd_doc_slp_ExitStatus_GLOBALLY_INFEASIBLE[] =
R"doc(The problem setup frontend determined the problem to have an empty
feasible region.)doc";

constexpr char mkd_doc_slp_ExitStatus_LINE_SEARCH_FAILED[] =
R"doc(The backtracking line search failed, and the problem isn't locally
infeasible.)doc";

constexpr char mkd_doc_slp_ExitStatus_LOCALLY_INFEASIBLE[] =
R"doc(The solver determined the problem to be locally infeasible and gave
up.)doc";

constexpr char mkd_doc_slp_ExitStatus_MAX_ITERATIONS_EXCEEDED[] =
R"doc(The solver returned its solution so far after exceeding the maximum
number of iterations.)doc";

constexpr char mkd_doc_slp_ExitStatus_NONFINITE_INITIAL_GUESS[] =
R"doc(The solver encountered nonfinite initial cost, constraints, or
derivatives and gave up.)doc";

constexpr char mkd_doc_slp_ExitStatus_SUCCESS[] = R"doc(Solved the problem to the desired tolerance.)doc";

constexpr char mkd_doc_slp_ExitStatus_TIMEOUT[] =
R"doc(The solver returned its solution so far after exceeding the maximum
elapsed wall clock time.)doc";

constexpr char mkd_doc_slp_ExitStatus_TOO_FEW_DOFS[] = R"doc(The solver determined the problem to be overconstrained and gave up.)doc";

constexpr char mkd_doc_slp_ExpressionType[] =
R"doc(Expression type.

Used for autodiff caching.)doc";

constexpr char mkd_doc_slp_ExpressionType_CONSTANT[] = R"doc(The expression is a constant.)doc";

constexpr char mkd_doc_slp_ExpressionType_LINEAR[] = R"doc(The expression is composed of linear and lower-order operators.)doc";

constexpr char mkd_doc_slp_ExpressionType_NONE[] = R"doc(There is no expression.)doc";

constexpr char mkd_doc_slp_ExpressionType_NONLINEAR[] = R"doc(The expression is composed of nonlinear and lower-order operators.)doc";

constexpr char mkd_doc_slp_ExpressionType_PARAMETER[] = R"doc(The expression is composed of parameters and constants.)doc";

constexpr char mkd_doc_slp_ExpressionType_QUADRATIC[] = R"doc(The expression is composed of quadratic and lower-order operators.)doc";

constexpr char mkd_doc_slp_Gradient[] =
R"doc(This class calculates the gradient of an expression with respect to a
vector of variables.

The gradient is only recomputed if the expression is quadratic or
higher order.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_Gradient_2[] = R"doc()doc";

constexpr char mkd_doc_slp_Gradient_Gradient[] =
R"doc(Constructs a Gradient object.

Args:
    expression: Expression of which to compute the gradient.
    wrt: Variable with respect to which to compute the gradient.

)doc";

constexpr char mkd_doc_slp_Gradient_Gradient_2[] =
R"doc(Constructs a Gradient object.

Args:
    expression: Expression of which to compute the gradient.
    wrt: Vector of variables with respect to which to compute the
         gradient.

)doc";

constexpr char mkd_doc_slp_Gradient_get[] =
R"doc(Returns the gradient as a RealMatrix.

This is useful when constructing optimization problems with
derivatives in them.

Returns:
    The gradient as a RealMatrix.

)doc";

constexpr char mkd_doc_slp_Gradient_m_g[] = R"doc()doc";

constexpr char mkd_doc_slp_Gradient_m_jacobian[] = R"doc()doc";

constexpr char mkd_doc_slp_Gradient_value[] =
R"doc(Evaluates the gradient at wrt's value.

Returns:
    The gradient at wrt's value.

)doc";

constexpr char mkd_doc_slp_Hessian[] =
R"doc(This class calculates the Hessian of an expression with respect to a
vector of variables.

The gradient tree is cached so subsequent Hessian calculations are
faster, and the Hessian is only recomputed if the expression is
nonlinear or its quadratic coefficients depend on parameters.

Template Args:
    Scalar: Scalar type.
    UpLo: Which part of the Hessian to compute (Lower or Lower |
          Upper). Default is Lower | Upper.)doc";

constexpr char mkd_doc_slp_Hessian_2[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_3[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_Hessian[] =
R"doc(Constructs a Hessian object.

Args:
    expression: Expression of which to compute the Hessian.
    wrt: Variable with respect to which to compute the Hessian.

)doc";

constexpr char mkd_doc_slp_Hessian_Hessian_2[] =
R"doc(Constructs a Hessian object.

Args:
    expression: Expression of which to compute the Hessian.
    wrt: Vector of variables with respect to which to compute the
         Hessian.

)doc";

constexpr char mkd_doc_slp_Hessian_get[] =
R"doc(Returns the Hessian as a RealMatrix.

This is useful when constructing optimization problems with
derivatives in them.

Returns:
    The Hessian as a RealMatrix.

)doc";

constexpr char mkd_doc_slp_Hessian_m_H[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_m_cached_triplets[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_m_expressions[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_m_nonlinear_rows[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_m_output_lists[] = R"doc(List of output rows as column-node pairs)doc";

constexpr char mkd_doc_slp_Hessian_m_top_lists[] =
R"doc(List of topologically sorted graphs from parent to child, one for each
row)doc";

constexpr char mkd_doc_slp_Hessian_m_wrt[] = R"doc()doc";

constexpr char mkd_doc_slp_Hessian_value[] =
R"doc(Evaluates the Hessian at wrt's value.

Returns:
    The Hessian at wrt's value.

)doc";

constexpr char mkd_doc_slp_InequalityConstraints[] =
R"doc(A vector of inequality constraints of the form cᵢ(x) ≥ 0.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_InequalityConstraints_2[] =
R"doc(A vector of inequality constraints of the form cᵢ(x) ≥ 0.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_InequalityConstraints_InequalityConstraints[] =
R"doc(Concatenates multiple inequality constraints.

Args:
    inequality_constraints: The list of InequalityConstraints to
                            concatenate.

)doc";

constexpr char mkd_doc_slp_InequalityConstraints_InequalityConstraints_2[] =
R"doc(Concatenates multiple inequality constraints.

This overload is for Python bindings only.

Args:
    inequality_constraints: The list of InequalityConstraints to
                            concatenate.

)doc";

constexpr char mkd_doc_slp_InequalityConstraints_InequalityConstraints_3[] =
R"doc(Constructs an inequality constraint from a left and right side.

The standard form for inequality constraints is c(x) ≥ 0. This
function takes a constraints of the form lhs ≥ rhs and converts it to
lhs - rhs ≥ 0.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side.
    strict: Whether the comparison is strict (lhs > rhs).

)doc";

constexpr char mkd_doc_slp_InequalityConstraints_constraints[] = R"doc(A vector of scalar inequality constraints.)doc";

constexpr char mkd_doc_slp_InequalityConstraints_operator_bool[] = R"doc(Implicit conversion operator to bool.)doc";

constexpr char mkd_doc_slp_InequalityConstraints_strict[] =
R"doc(Whether each scalar inequality constraint came from a strict
comparison (< or >).

The solver treats strict and non-strict inequality constraints the
same. if_else() and the conversion to bool distinguish them.)doc";

constexpr char mkd_doc_slp_IterationInfo[] =
R"doc(Solver iteration information exposed to an iteration callback.

If some decision variables lie on a manifold (e.g., from
Problem::unit_vector_decision_variable()), x is the point in the
ambient space, and the derivatives are with respect to the manifold's
tangent space at x, so they have fewer columns than x has rows.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_IterationInfo_A_e[] = R"doc(The equality constraint Jacobian.)doc";

constexpr char mkd_doc_slp_IterationInfo_A_i[] = R"doc(The inequality constraint Jacobian.)doc";

constexpr char mkd_doc_slp_IterationInfo_H[] = R"doc(The Hessian of the Lagrangian.)doc";

constexpr char mkd_doc_slp_IterationInfo_g[] = R"doc(The gradient of the cost function.)doc";

constexpr char mkd_doc_slp_IterationInfo_iteration[] = R"doc(The solver iteration.)doc";

constexpr char mkd_doc_slp_IterationInfo_s[] = R"doc(The inequality constraint slack variables.)doc";

constexpr char mkd_doc_slp_IterationInfo_x[] = R"doc(The decision variables.)doc";

constexpr char mkd_doc_slp_IterationInfo_y[] = R"doc(The equality constraint dual variables.)doc";

constexpr char mkd_doc_slp_IterationInfo_z[] = R"doc(The inequality constraint dual variables.)doc";

constexpr char mkd_doc_slp_Jacobian[] =
R"doc(This class calculates the Jacobian of a vector of expressions with
respect to a vector of variables.

The Jacobian is only recomputed if the expression is quadratic or
higher order, or if it's linear with coefficients that depend on
parameters.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_Jacobian_2[] = R"doc()doc";

constexpr char mkd_doc_slp_Jacobian_3[] = R"doc()doc";

constexpr char mkd_doc_slp_Jacobian_Jacobian[] =
R"doc(Constructs a Jacobian object.

Args:
    expression: Expression of which to compute the Jacobian.
    wrt: Variable with respect to which to compute the Jacobian.

)doc";

constexpr char mkd_doc_slp_Jacobian_Jacobian_2[] =
R"doc(Constructs a Jacobian object.

Args:
    expression: Expression of which to compute the Jacobian.
    wrt: Vector of variables with respect to which to compute the
         Jacobian.

)doc";

constexpr char mkd_doc_slp_Jacobian_Jacobian_3[] =
R"doc(Constructs a Jacobian object.

Args:
    expressions: Vector of expressions of which to compute the
                 Jacobian.
    wrt: Vector of variables with respect to which to compute the
         Jacobian.

)doc";

constexpr char mkd_doc_slp_Jacobian_get[] =
R"doc(Returns the Jacobian as a RealMatrix.

This is useful when constructing optimization problems with
derivatives in them.

Returns:
    The Jacobian as a RealMatrix.

)doc";

constexpr char mkd_doc_slp_Jacobian_m_J[] = R"doc()doc";

constexpr char mkd_doc_slp_Jacobian_m_cached_triplets[] = R"doc(Cached triplets for gradients of linear rows)doc";

constexpr char mkd_doc_slp_Jacobian_m_expressions[] = R"doc()doc";

constexpr char mkd_doc_slp_Jacobian_m_nonlinear_rows[] =
R"doc(List of row indices for nonlinear rows and linear rows with parameters
whose gradients will be computed in value())doc";

constexpr char mkd_doc_slp_Jacobian_m_output_lists[] = R"doc(List of output rows as column-node pairs)doc";

constexpr char mkd_doc_slp_Jacobian_m_top_lists[] =
R"doc(List of topologically sorted graphs from parent to child, one for each
row)doc";

constexpr char mkd_doc_slp_Jacobian_m_wrt[] = R"doc()doc";

constexpr char mkd_doc_slp_Jacobian_value[] =
R"doc(Evaluates the Jacobian at wrt's value.

Returns:
    The Jacobian at wrt's value.

)doc";

constexpr char mkd_doc_slp_LintFinding[] =
R"doc(A suggested improvement to a problem's formulation from
Problem::lint().

Decision variable indices count decision variables in the order they
were created. Constraint indices count scalar constraints in the order
they were added, with matrix constraints adding one per element.
Messages refer to decision variable i as x[i].)doc";

constexpr char mkd_doc_slp_LintFinding_cost[] = R"doc(Whether the cost function is involved.)doc";

constexpr char mkd_doc_slp_LintFinding_decision_variables[] = R"doc(The indices of the decision variables involved.)doc";

constexpr char mkd_doc_slp_LintFinding_equality_constraints[] = R"doc(The indices of the equality constraints involved.)doc";

constexpr char mkd_doc_slp_LintFinding_inequality_constraints[] = R"doc(The indices of the inequality constraints involved.)doc";

constexpr char mkd_doc_slp_LintFinding_kind[] = R"doc(The kind of formulation issue.)doc";

constexpr char mkd_doc_slp_LintFinding_message[] = R"doc(A description of the issue and how to fix it.)doc";

constexpr char mkd_doc_slp_LintKind[] = R"doc(The kind of formulation issue a lint finding reports.)doc";

constexpr char mkd_doc_slp_LintKind_BADLY_SCALED[] =
R"doc(Coefficients of the cost function or linear constraints span so many
orders of magnitude that the solver loses precision, and rescaling
decision variables or constraints fixes it.)doc";

constexpr char mkd_doc_slp_LintKind_CONFLICTING_CONSTRAINTS[] =
R"doc(Linear constraints on the same linear combination of decision
variables contradict each other, so the problem is infeasible.)doc";

constexpr char mkd_doc_slp_LintKind_CONSTANT_CONSTRAINT[] =
R"doc(A constraint doesn't depend on any decision variables, so it's either
always satisfied or never satisfied.)doc";

constexpr char mkd_doc_slp_LintKind_EQUALITY_AS_INEQUALITIES[] =
R"doc(A lower bound and an upper bound on the same linear combination of
decision variables are equal, so together they're an equality
constraint.)doc";

constexpr char mkd_doc_slp_LintKind_NONSMOOTH_FUNCTION[] =
R"doc(The cost function or a constraint uses a function of decision
variables that isn't differentiable everywhere, like abs() or a norm,
in a form with a differentiable replacement.)doc";

constexpr char mkd_doc_slp_LintKind_REDUNDANT_CONSTRAINTS[] =
R"doc(Linear constraints are implied by other linear constraints on the same
linear combination of decision variables.)doc";

constexpr char mkd_doc_slp_LintKind_UNUSED_DECISION_VARIABLE[] =
R"doc(Decision variables don't appear in the cost function or any
constraint.)doc";

constexpr char mkd_doc_slp_OCP[] =
R"doc(This class allows the user to pose and solve a constrained optimal
control problem (OCP) in a variety of ways.

The system is transcripted by one of three methods (direct
transcription, direct collocation, or single-shooting) and additional
constraints can be added.

In direct transcription, each state is a decision variable constrained
to the integrated dynamics of the previous state. In direct
collocation, the trajectory is modeled as a series of cubic
polynomials where the centerpoint slope is constrained. In single-
shooting, states depend explicitly as a function of all previous
states and all previous inputs.

Explicit ODEs are integrated using RK4.

For explicit ODEs, the function must be in the form dx/dt = f(t, x,
u). For discrete state transition functions, the function must be in
the form xₖ₊₁ = f(t, xₖ, uₖ).

Direct collocation requires an explicit ODE. Direct transcription and
single-shooting can use either an ODE or state transition function.

https://underactuated.mit.edu/trajopt.html goes into more detail on
each transcription method.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_OCP_2[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_OCP[] =
R"doc(Builds an optimization problem using a system evolution function
(explicit ODE or discrete state transition function).

Args:
    num_states: The number of system states.
    num_inputs: The number of system inputs.
    dt: The timestep for fixed-step integration.
    num_steps: The number of control points.
    dynamics: Function representing an explicit or implicit ODE, or a
              discrete state transition function.

* Explicit: dx/dt = f(x, u, *)

 

* Implicit: f([x dx/dt]', u, *) = 0

 

* State transition: xₖ₊₁ = f(xₖ, uₖ)

 
    dynamics_type: The type of system evolution function.
    timestep_method: The timestep method.
    transcription_method: The transcription method.

)doc";

constexpr char mkd_doc_slp_OCP_OCP_2[] =
R"doc(Builds an optimization problem using a system evolution function
(explicit ODE or discrete state transition function).

Args:
    num_states: The number of system states.
    num_inputs: The number of system inputs.
    dt: The timestep for fixed-step integration.
    num_steps: The number of control points.
    dynamics: Function representing an explicit or implicit ODE, or a
              discrete state transition function.

* Explicit: dx/dt = f(t, x, u, *)

 

* Implicit: f(t, [x dx/dt]', u, *) = 0

 

* State transition: xₖ₊₁ = f(t, xₖ, uₖ, dt)

 
    dynamics_type: The type of system evolution function.
    timestep_method: The timestep method.
    transcription_method: The transcription method.

)doc";

constexpr char mkd_doc_slp_OCP_U[] =
R"doc(Gets the input variables. After the problem is solved, this will
contain the inputs corresponding to the optimized trajectory.

Shaped (num_inputs)x(num_steps+1), although the last input step is
unused in the trajectory.

Returns:
    The input variable matrix.

)doc";

constexpr char mkd_doc_slp_OCP_X[] =
R"doc(Gets the state variables. After the problem is solved, this will
contain the optimized trajectory.

Shaped (num_states)x(num_steps+1).

Returns:
    The state variable matrix.

)doc";

constexpr char mkd_doc_slp_OCP_constrain_direct_collocation[] = R"doc(Applies direct collocation dynamics constraints.)doc";

constexpr char mkd_doc_slp_OCP_constrain_direct_transcription[] = R"doc(Applies direct transcription dynamics constraints.)doc";

constexpr char mkd_doc_slp_OCP_constrain_final_state[] =
R"doc(Constrains the final state.

Args:
    final_state: the final state to constrain to.

)doc";

constexpr char mkd_doc_slp_OCP_constrain_initial_state[] =
R"doc(Constrains the initial state.

Args:
    initial_state: the initial state to constrain to.

)doc";

constexpr char mkd_doc_slp_OCP_constrain_single_shooting[] = R"doc(Applies single shooting dynamics constraints.)doc";

constexpr char mkd_doc_slp_OCP_dt[] =
R"doc(Gets the timestep variables. After the problem is solved, this will
contain the timesteps corresponding to the optimized trajectory.

Shaped 1x(num_steps+1), although the last timestep is unused in the
trajectory.

Returns:
    The timestep variable matrix.

)doc";

constexpr char mkd_doc_slp_OCP_final_state[] =
R"doc(Gets the final state in the trajectory.

Returns:
    The final state of the trajectory.

)doc";

constexpr char mkd_doc_slp_OCP_for_each_step[] =
R"doc(Sets the constraint evaluation function. This function is called
`num_steps+1` times, with the corresponding state and input Matrices.

Args:
    callback: The callback f(x, u) where x is the state and u is the
              input vector.

)doc";

constexpr char mkd_doc_slp_OCP_for_each_step_2[] =
R"doc(Sets the constraint evaluation function. This function is called
`num_steps+1` times, with the corresponding state and input Matrices.

Args:
    callback: The callback f(t, x, u, dt) where t is time, x is the
              state vector, u is the input vector, and dt is the
              timestep duration.

)doc";

constexpr char mkd_doc_slp_OCP_initial_state[] =
R"doc(Gets the initial state in the trajectory.

Returns:
    The initial state of the trajectory.

)doc";

constexpr char mkd_doc_slp_OCP_m_DT[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_U[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_X[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_dynamics[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_dynamics_type[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_num_steps[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_m_transcription_method[] = R"doc()doc";

constexpr char mkd_doc_slp_OCP_rk4[] =
R"doc(Performs 4th order Runge-Kutta integration of dx/dt = f(t, x, u) for
dt.

Args:
    f: The function to integrate. It must take two arguments x and u.
    x: The initial value of x.
    u: The value u held constant over the integration period.
    t0: The initial time.
    dt: The time over which to integrate.

)doc";

constexpr char mkd_doc_slp_OCP_set_lower_input_bound[] =
R"doc(Sets a lower bound on the input.

Args:
    lower_bound: The lower bound that inputs must always be above.
                 Must be shaped (num_inputs)x1.

)doc";

constexpr char mkd_doc_slp_OCP_set_max_timestep[] =
R"doc(Sets an upper bound on the timestep.

Args:
    max_timestep: The maximum timestep.

)doc";

constexpr char mkd_doc_slp_OCP_set_min_timestep[] =
R"doc(Sets a lower bound on the timestep.

Args:
    min_timestep: The minimum timestep.

)doc";

constexpr char mkd_doc_slp_OCP_set_upper_input_bound[] =
R"doc(Sets an upper bound on the input.

Args:
    upper_bound: The upper bound that inputs must always be below.
                 Must be shaped (num_inputs)x1.

)doc";

constexpr char mkd_doc_slp_Options[] =
R"doc(Solver options.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_Options_diagnostics[] =
R"doc(Enables diagnostic output.

See https://sleipnirgroup.github.io/Sleipnir/md_usage.html#output for
more information.)doc";

constexpr char mkd_doc_slp_Options_feasible_ipm[] =
R"doc(Enables the feasible interior-point method.

When the inequality constraints are all feasible, step sizes are
reduced when necessary to prevent them becoming infeasible again. This
is useful when parts of the problem are ill-conditioned in infeasible
regions (e.g., square root of a negative value). This can slow or
prevent progress toward a solution though, so only enable it if
necessary.)doc";

constexpr char mkd_doc_slp_Options_max_iterations[] = R"doc(The maximum number of solver iterations before returning a solution.)doc";

constexpr char mkd_doc_slp_Options_timeout[] = R"doc(The maximum elapsed wall clock time before returning a solution.)doc";

constexpr char mkd_doc_slp_Options_tolerance[] =
R"doc(The solver will stop once the error is below this tolerance.

Defaults to the square root of the scalar type's machine epsilon.)doc";

constexpr char mkd_doc_slp_Problem[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_2[] =
R"doc(This class allows the user to pose a constrained nonlinear
optimization problem in natural mathematical notation and solve it.

This class supports problems of the form:

```
minₓ f(x)
subject to cₑ(x) = 0
           cᵢ(x) ≥ 0
```



where f(x) is the scalar cost function, x is the vector of decision
variables (variables the solver can tweak to minimize the cost
function), cᵢ(x) are the inequality constraints, and cₑ(x) are the
equality constraints. Constraints are equations or inequalities of the
decision variables that constrain what values the solver is allowed to
use when searching for an optimal solution.

The nice thing about this class is users don't have to put their
system in the form shown above manually; they can write it in natural
mathematical form and it'll be converted for them.

Template Args:
    Scalar: Scalar type.)doc";

constexpr char mkd_doc_slp_Problem_3[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_Problem[] = R"doc(Constructs the optimization problem.)doc";

constexpr char mkd_doc_slp_Problem_add_callback[] =
R"doc(Adds a callback to be called at the beginning of each solver
iteration.

The callback for this overload should return void.

Args:
    callback: The callback.

)doc";

constexpr char mkd_doc_slp_Problem_add_callback_2[] =
R"doc(Adds a callback to be called at the beginning of each solver
iteration.

The callback for this overload should return ControlFlow.

Args:
    callback: The callback. Returning ControlFlow::BREAK from the
              callback stops solving and returns the solution so far.

)doc";

constexpr char mkd_doc_slp_Problem_add_persistent_callback[] =
R"doc(Adds a callback to be called at the beginning of each solver
iteration.

Language bindings should call this in the Problem constructor to
register callbacks that shouldn't be removed by clear_callbacks().
Persistent callbacks run after non-persistent callbacks.

Args:
    callback: The callback. Returning ControlFlow::BREAK from the
              callback stops solving and returns the solution so far.

)doc";

constexpr char mkd_doc_slp_Problem_clear_callbacks[] = R"doc(Clears the registered callbacks.)doc";

constexpr char mkd_doc_slp_Problem_complex_decision_variable[] =
R"doc(Creates a complex decision variable in the optimization problem.

The real and imaginary parts are two real decision variables. Complex
decision variables have an initial value of zero.

Returns:
    A complex decision variable in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_complex_decision_variable_2[] =
R"doc(Creates a matrix of complex decision variables in the optimization
problem.

The real and imaginary parts of each entry are two real decision
variables. Complex decision variables have an initial value of zero.

Args:
    rows: Number of matrix rows.
    cols: Number of matrix columns.

Returns:
    A matrix of complex decision variables in the optimization
    problem.

)doc";

constexpr char mkd_doc_slp_Problem_complex_parameter[] =
R"doc(Creates a complex parameter in the optimization problem.

A parameter is a constant whose value can be changed between solves
with set_value(). Unlike constants, parameters aren't folded into
expressions.

Complex parameters have an initial value of zero.

Returns:
    A complex parameter in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_complex_parameter_2[] =
R"doc(Creates a matrix of complex parameters in the optimization problem.

A parameter is a constant whose value can be changed between solves
with set_value(). Unlike constants, parameters aren't folded into
expressions.

Complex parameters have an initial value of zero.

Args:
    rows: Number of matrix rows.
    cols: Number of matrix columns.

Returns:
    A matrix of complex parameters in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_cost_function_type[] =
R"doc(Returns the cost function's type.

Returns:
    The cost function's type.

)doc";

constexpr char mkd_doc_slp_Problem_equality_constraint_type[] =
R"doc(Returns the type of the highest order equality constraint.

Returns:
    The type of the highest order equality constraint.

)doc";

constexpr char mkd_doc_slp_Problem_inequality_constraint_type[] =
R"doc(Returns the type of the highest order inequality constraint.

Returns:
    The type of the highest order inequality constraint.

)doc";

constexpr char mkd_doc_slp_Problem_lint[] =
R"doc(Checks the problem's formulation for issues and suggests improvements.

Linting analyzes the cost function and constraints without changing or
solving the problem. It reports:


  

* Decision variables that don't appear in the cost function or any
    constraint.


  

* Constraints that don't depend on any decision variables.


  

* Linear constraints on the same linear combination of decision
    variables that are redundant, conflict, or form an equality
    constraint.


  

* Functions of decision variables that aren't differentiable
    everywhere, in forms with a differentiable replacement: abs(a) ≤
    b, max(a, b) ≤ c, min(a, b) ≥ c, a norm compared to a constant,
    and a cost function that's a norm.


  

* Coefficients of the cost function or linear constraints that span
    so many orders of magnitude that the solver loses precision, when
    rescaling decision variables or constraints fixes it.




Values in the findings, such as constraint values and bounds, use the
decision variables' and parameters' current values.

Returns:
    The findings, which are empty if there's nothing to improve.

)doc";

constexpr char mkd_doc_slp_Problem_m_decision_variables[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_equality_constraints[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_f[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_inequality_constraints[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_iteration_callbacks[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_manifold_blocks[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_m_persistent_iteration_callbacks[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_maximize[] =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Args:
    objective: The objective function to maximize. A 1x1 RealMatrix
               will implicitly convert to a Real, and a non-1x1
               RealMatrix will raise an assertion.

)doc";

constexpr char mkd_doc_slp_Problem_maximize_2[] =
R"doc(Tells the solver to maximize the output of the given objective
function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Args:
    objective: The objective function to maximize. A 1x1 RealMatrix
               will implicitly convert to a Real, and a non-1x1
               RealMatrix will raise an assertion.

)doc";

constexpr char mkd_doc_slp_Problem_minimize[] =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Args:
    cost: The cost function to minimize. A 1x1 RealMatrix will
          implicitly convert to a Real, and a non-1x1 RealMatrix will
          raise an assertion.

)doc";

constexpr char mkd_doc_slp_Problem_minimize_2[] =
R"doc(Tells the solver to minimize the output of the given cost function.

Note that this is optional. If only constraints are specified, the
solver will find the closest solution to the initial conditions that's
in the feasible set.

Args:
    cost: The cost function to minimize. A 1x1 RealMatrix will
          implicitly convert to a Real, and a non-1x1 RealMatrix will
          raise an assertion.

)doc";

constexpr char mkd_doc_slp_Problem_parameter[] =
R"doc(Creates a parameter in the optimization problem.

A parameter is a constant whose value can be changed between solves
with set_value(). Unlike constants, parameters aren't folded into
expressions.

Parameters have an initial value of zero.

Returns:
    A parameter in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_parameter_2[] =
R"doc(Creates a matrix of parameters in the optimization problem.

A parameter is a constant whose value can be changed between solves
with set_value(). Unlike constants, parameters aren't folded into
expressions.

Parameters have an initial value of zero.

Args:
    rows: Number of matrix rows.
    cols: Number of matrix columns.

Returns:
    A matrix of parameters in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_print_exit_conditions[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_print_problem_analysis[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_real_decision_variable[] =
R"doc(Creates a decision variable in the optimization problem.

Decision variables have an initial value of zero.

Returns:
    A decision variable in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_real_decision_variable_2[] =
R"doc(Creates a matrix of decision variables in the optimization problem.

Decision variables have an initial value of zero.

Args:
    rows: Number of matrix rows.
    cols: Number of matrix columns.

Returns:
    A matrix of decision variables in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_solve[] =
R"doc(Solves the optimization problem. The solution will be stored in the
original variables used to construct the problem.

Args:
    options: Solver options.
    spy: Enables writing sparsity patterns of H, Aₑ, and Aᵢ to files
         named H.spy, A_e.spy, and A_i.spy respectively during solve.
         Use tools/spy.py to plot them.

Returns:
    The solver status.

)doc";

constexpr char mkd_doc_slp_Problem_solve_ipm[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_solve_newton[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_solve_sqp[] = R"doc()doc";

constexpr char mkd_doc_slp_Problem_subject_to[] =
R"doc(Tells the solver to solve the problem while satisfying the given
equality constraint.

Args:
    constraint: The constraint to satisfy.

)doc";

constexpr char mkd_doc_slp_Problem_subject_to_2[] =
R"doc(Tells the solver to solve the problem while satisfying the given
equality constraint.

Args:
    constraint: The constraint to satisfy.

)doc";

constexpr char mkd_doc_slp_Problem_subject_to_3[] =
R"doc(Tells the solver to solve the problem while satisfying the given
inequality constraint.

Args:
    constraint: The constraint to satisfy.

)doc";

constexpr char mkd_doc_slp_Problem_subject_to_4[] =
R"doc(Tells the solver to solve the problem while satisfying the given
inequality constraint.

Args:
    constraint: The constraint to satisfy.

)doc";

constexpr char mkd_doc_slp_Problem_translation2_decision_variable[] =
R"doc(Creates a 2D translation decision variable in the optimization
problem.

The translation is two ordinary decision variables.

The translation has an initial value of zero.

Returns:
    A 2D translation of decision variables in the optimization
    problem.

)doc";

constexpr char mkd_doc_slp_Problem_translation2_parameter[] =
R"doc(Creates a 2D translation parameter in the optimization problem.

A parameter is a constant whose value can be changed between solves
with set_value(). Unlike constants, parameters aren't folded into
expressions.

The translation has an initial value of zero.

Returns:
    A 2D translation of parameters in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_translation3_decision_variable[] =
R"doc(Creates a 3D translation decision variable in the optimization
problem.

The translation is three ordinary decision variables.

The translation has an initial value of zero.

Returns:
    A 3D translation of decision variables in the optimization
    problem.

)doc";

constexpr char mkd_doc_slp_Problem_translation3_parameter[] =
R"doc(Creates a 3D translation parameter in the optimization problem.

A parameter is a constant whose value can be changed between solves
with set_value(). Unlike constants, parameters aren't folded into
expressions.

The translation has an initial value of zero.

Returns:
    A 3D translation of parameters in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_unit_vector_decision_variable[] =
R"doc(Creates a unit vector of decision variables in the optimization
problem.

The solver keeps the vector on the unit sphere by stepping along the
sphere instead of through ℝⁿ, so no equality constraint is needed.

The vector has an initial value of e₁ = [1, 0, …, 0]ᵀ. Values set
elsewhere are normalized when the solve starts.

Args:
    rows: The vector's dimension. Must be at least 2.

Returns:
    A unit vector of decision variables in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Problem_unit_vector_parameter[] =
R"doc(Creates a unit vector of parameters in the optimization problem.

A parameter is a constant whose value can be changed between solves
with set_value(). Unlike constants, parameters aren't folded into
expressions.

The vector has an initial value of e₁ = [1, 0, …, 0]ᵀ, and set_value()
normalizes its argument.

Args:
    rows: The vector's dimension. Must be at least 2.

Returns:
    A unit vector of parameters in the optimization problem.

)doc";

constexpr char mkd_doc_slp_Real[] =
R"doc(A real-valued autodiff expression.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_Real_2[] =
R"doc(A real-valued autodiff expression.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_RealMatrix[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_2[] =
R"doc(A matrix of autodiff expressions.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_RealMatrix_3[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix[] = R"doc(Constructs an empty RealMatrix.)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_2[] =
R"doc(Constructs a zero-initialized RealMatrix column vector with the given
rows.

Args:
    rows: The number of matrix rows.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_3[] =
R"doc(Constructs a zero-initialized RealMatrix with the given dimensions.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_4[] =
R"doc(Constructs an empty RealMatrix with the given dimensions.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_5[] =
R"doc(Constructs a RealMatrix from a nested list of Reals.

Args:
    list: The nested list of Reals.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_6[] =
R"doc(Constructs a RealMatrix from a nested list of scalars.

This overload is for Python bindings only.

Args:
    list: The nested list of Reals.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_7[] =
R"doc(Constructs a RealMatrix from a nested list of Reals.

This overload is for Python bindings only.

Args:
    list: The nested list of Reals.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_8[] =
R"doc(Constructs a RealMatrix from an Eigen matrix.

Args:
    values: Eigen matrix of values.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_9[] =
R"doc(Constructs a RealMatrix from an Eigen diagonal matrix.

Args:
    values: Diagonal matrix of values.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_10[] =
R"doc(Constructs a scalar RealMatrix from a Real.

Args:
    variable: Real.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_11[] =
R"doc(Constructs a scalar RealMatrix from a Real.

Args:
    variable: Real.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_12[] =
R"doc(Constructs a RealMatrix from a RealSubmatrix.

Args:
    values: RealSubmatrix of values.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_13[] =
R"doc(Constructs a RealMatrix from a RealSubmatrix.

Args:
    values: RealSubmatrix of values.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_14[] =
R"doc(Constructs a RealMatrix column vector from a Real array.

Args:
    values: Real array of values.

)doc";

constexpr char mkd_doc_slp_RealMatrix_RealMatrix_15[] =
R"doc(Constructs a RealMatrix from a Real array.

Args:
    values: Real array of values.
    rows: The number of matrix rows.
    cols: The number of matrix columns.

)doc";

constexpr char mkd_doc_slp_RealMatrix_T[] =
R"doc(Returns the transpose of the matrix.

Returns:
    The transpose of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_begin[] =
R"doc(Returns begin iterator.

Returns:
    Begin iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_begin_2[] =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_block[] =
R"doc(Returns a block of the matrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_block_2[] =
R"doc(Returns a block of the matrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_cbegin[] =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_cend[] =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_col[] =
R"doc(Returns a column slice of the matrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_col_2[] =
R"doc(Returns a column slice of the matrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_cols[] =
R"doc(Returns the number of columns in the matrix.

Returns:
    The number of columns in the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_const_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_const_iterator_const_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_const_iterator_const_iterator_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_const_iterator_m_it[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_const_iterator_operator_dec[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_const_iterator_operator_dec_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_const_iterator_operator_eq[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_const_iterator_operator_inc[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_const_iterator_operator_inc_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_const_iterator_operator_mul[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_constant[] =
R"doc(Returns a matrix filled with a constant.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.
    constant: The constant.

Returns:
    A matrix filled with a constant.

)doc";

constexpr char mkd_doc_slp_RealMatrix_crbegin[] =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_crend[] =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_cwise_transform[] =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Args:
    unary_op: The unary operator to use for the transform operation.

Returns:
    Result of the unary operator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_end[] =
R"doc(Returns end iterator.

Returns:
    End iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_end_2[] =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_exp[] =
R"doc(Returns the matrix exponential.

Returns:
    The matrix exponential.

)doc";

constexpr char mkd_doc_slp_RealMatrix_identity[] =
R"doc(Returns an identity matrix.

Args:
    rows: The number of matrix rows.

Returns:
    An identity matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_iterator_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_iterator_iterator_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_iterator_m_it[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_iterator_operator_dec[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_iterator_operator_dec_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_iterator_operator_eq[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_iterator_operator_inc[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_iterator_operator_inc_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_iterator_operator_mul[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_m_cols[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_m_rows[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_m_storage[] = R"doc()doc";

constexpr char mkd_doc_slp_RealMatrix_one[] =
R"doc(Returns a matrix filled with ones.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.

Returns:
    A matrix filled with ones.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_array[] =
R"doc(Returns the element at the given row and column.

Args:
    row: The row.
    col: The column.

Returns:
    The element at the given row and column.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_array_2[] =
R"doc(Returns the element at the given row and column.

Args:
    row: The row.
    col: The column.

Returns:
    The element at the given row and column.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_array_3[] =
R"doc(Returns the element at the given index.

Args:
    index: The index.

Returns:
    The element at the given index.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_array_4[] =
R"doc(Returns the element at the given index.

Args:
    index: The index.

Returns:
    The element at the given index.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_array_5[] =
R"doc(Returns a slice of the matrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_array_6[] =
R"doc(Returns a slice of the matrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_array_7[] =
R"doc(Returns a slice of the matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_array_8[] =
R"doc(Returns a slice of the matrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_assign[] =
R"doc(Assigns an Eigen matrix to the matrix.

Args:
    values: Eigen matrix of values.

Returns:
    This matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_assign_2[] =
R"doc(Assigns a scalar to the matrix.

This only works for matrices with one row and one column.

Args:
    value: Value to assign.

Returns:
    This matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_iadd[] =
R"doc(Compound addition-assignment operator.

Args:
    rhs: Real to add.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_iadd_2[] =
R"doc(Compound addition-assignment operator.

Args:
    rhs: Real to add.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_idiv[] =
R"doc(Compound matrix division-assignment operator.

Args:
    rhs: Real to divide.

Returns:
    Result of division.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_imul[] =
R"doc(Compound matrix multiplication-assignment operator.

Args:
    rhs: Real to multiply.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_imul_2[] =
R"doc(Compound matrix-scalar multiplication-assignment operator.

Args:
    rhs: Real to multiply.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_isub[] =
R"doc(Compound subtraction-assignment operator.

Args:
    rhs: Real to subtract.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_isub_2[] =
R"doc(Compound subtraction-assignment operator.

Args:
    rhs: Real to subtract.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_RealMatrix_operator_slp_Real[] = R"doc(Implicit conversion operator from 1x1 RealMatrix to Real.)doc";

constexpr char mkd_doc_slp_RealMatrix_rbegin[] =
R"doc(Returns reverse begin iterator.

Returns:
    Reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_rbegin_2[] =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_rend[] =
R"doc(Returns reverse end iterator.

Returns:
    Reverse end iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_rend_2[] =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.

)doc";

constexpr char mkd_doc_slp_RealMatrix_row[] =
R"doc(Returns a row slice of the matrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_row_2[] =
R"doc(Returns a row slice of the matrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_rows[] =
R"doc(Returns the number of rows in the matrix.

Returns:
    The number of rows in the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_segment[] =
R"doc(Returns a segment of the vector.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the vector.

)doc";

constexpr char mkd_doc_slp_RealMatrix_segment_2[] =
R"doc(Returns a segment of the vector.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the vector.

)doc";

constexpr char mkd_doc_slp_RealMatrix_set_value[] =
R"doc(Sets the RealMatrix's internal values.

Args:
    values: Eigen matrix of values.

)doc";

constexpr char mkd_doc_slp_RealMatrix_size[] =
R"doc(Returns number of elements in matrix.

Returns:
    Number of elements in matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_to_eigen[] =
R"doc(Converts the RealMatrix to an Eigen matrix.

Returns:
    Eigen matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_trace[] =
R"doc(Returns the trace of the matrix, the sum of its diagonal.

Returns:
    The trace of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_value[] =
R"doc(Returns an element of the matrix.

Args:
    row: The row of the element to return.
    col: The column of the element to return.

Returns:
    An element of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_value_2[] =
R"doc(Returns an element of the matrix.

Args:
    index: The index of the element to return.

Returns:
    An element of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_value_3[] =
R"doc(Returns the contents of the matrix.

Returns:
    The contents of the matrix.

)doc";

constexpr char mkd_doc_slp_RealMatrix_zero[] =
R"doc(Returns a matrix filled with zeros.

Args:
    rows: The number of matrix rows.
    cols: The number of matrix columns.

Returns:
    A matrix filled with zeros.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix[] =
R"doc(A submatrix of autodiff expressions with reference semantics.

Template Args:
    Mat: The type of the matrix whose storage this class points to.)doc";

constexpr char mkd_doc_slp_RealSubmatrix_RealSubmatrix[] = R"doc(Copy constructor.)doc";

constexpr char mkd_doc_slp_RealSubmatrix_RealSubmatrix_2[] = R"doc(Move constructor.)doc";

constexpr char mkd_doc_slp_RealSubmatrix_RealSubmatrix_3[] =
R"doc(Constructs a RealSubmatrix pointing to all of the given matrix.

Args:
    mat: The matrix to which to point.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_RealSubmatrix_4[] =
R"doc(Constructs a RealSubmatrix pointing to a subset of the given matrix.

Args:
    mat: The matrix to which to point.
    row_offset: The block's row offset.
    col_offset: The block's column offset.
    block_rows: The number of rows in the block.
    block_cols: The number of columns in the block.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_RealSubmatrix_5[] =
R"doc(Constructs a RealSubmatrix pointing to a subset of the given matrix.

Note that the slices are taken as is rather than adjusted.

Args:
    mat: The matrix to which to point.
    row_slice: The block's row slice.
    row_slice_length: The block's row length.
    col_slice: The block's column slice.
    col_slice_length: The block's column length.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_T[] =
R"doc(Returns the transpose of the submatrix.

Returns:
    The transpose of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_begin[] =
R"doc(Returns begin iterator.

Returns:
    Begin iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_begin_2[] =
R"doc(Returns begin iterator.

Returns:
    Begin iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_block[] =
R"doc(Returns a block of the submatrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_block_2[] =
R"doc(Returns a block of the submatrix.

Args:
    row_offset: The row offset of the block selection.
    col_offset: The column offset of the block selection.
    block_rows: The number of rows in the block selection.
    block_cols: The number of columns in the block selection.

Returns:
    A block of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_cbegin[] =
R"doc(Returns const begin iterator.

Returns:
    Const begin iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_cend[] =
R"doc(Returns const end iterator.

Returns:
    Const end iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_col[] =
R"doc(Returns a column slice of the submatrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_col_2[] =
R"doc(Returns a column slice of the submatrix.

Args:
    col: The column to slice.

Returns:
    A column slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_cols[] =
R"doc(Returns the number of columns in the matrix.

Returns:
    The number of columns in the matrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_const_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_const_iterator_const_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_const_iterator_const_iterator_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_const_iterator_m_index[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_const_iterator_m_mat[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_const_iterator_operator_dec[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_const_iterator_operator_dec_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_const_iterator_operator_eq[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_const_iterator_operator_inc[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_const_iterator_operator_inc_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_const_iterator_operator_mul[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_crbegin[] =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_crend[] =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_cwise_transform[] =
R"doc(Transforms the matrix coefficient-wise with an unary operator.

Args:
    unary_op: The unary operator to use for the transform operation.

Returns:
    Result of the unary operator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_end[] =
R"doc(Returns end iterator.

Returns:
    End iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_end_2[] =
R"doc(Returns end iterator.

Returns:
    End iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_exp[] =
R"doc(Returns the matrix exponential.

Returns:
    The matrix exponential.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_iterator_iterator[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_iterator_iterator_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_iterator_m_index[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_iterator_m_mat[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_iterator_operator_dec[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_iterator_operator_dec_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_iterator_operator_eq[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_iterator_operator_inc[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_iterator_operator_inc_2[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_iterator_operator_mul[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_m_col_slice[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_m_col_slice_length[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_m_mat[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_m_row_slice[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_m_row_slice_length[] = R"doc()doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_array[] =
R"doc(Returns the element at the given row and column.

Args:
    row: The row.
    col: The column.

Returns:
    The element at the given row and column.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_array_2[] =
R"doc(Returns the element at the given row and column.

Args:
    row: The row.
    col: The column.

Returns:
    The element at the given row and column.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_array_3[] =
R"doc(Returns the element at the given index.

Args:
    index: The index.

Returns:
    The element at the given index.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_array_4[] =
R"doc(Returns the element at the given index.

Args:
    index: The index.

Returns:
    The element at the given index.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_array_5[] =
R"doc(Returns a slice of the submatrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_array_6[] =
R"doc(Returns a slice of the submatrix.

Args:
    row_slice: The row slice.
    col_slice: The column slice.

Returns:
    A slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_array_7[] =
R"doc(Returns a slice of the submatrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_array_8[] =
R"doc(Returns a slice of the submatrix.

The given slices aren't adjusted. This overload is for Python bindings
only.

Args:
    row_slice: The row slice.
    row_slice_length: The row slice length.
    col_slice: The column slice.
    col_slice_length: The column slice length.

Returns:
    A slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_assign[] =
R"doc(Assigns a RealSubmatrix to the block.

Args:
    values: RealSubmatrix of values.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_assign_2[] =
R"doc(Assigns a RealSubmatrix to the block.

Args:
    values: RealSubmatrix of values.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_assign_3[] =
R"doc(Assigns a scalar to the block.

This only works for blocks with one row and one column.

Args:
    value: Value to assign.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_assign_4[] =
R"doc(Assigns an Eigen matrix to the block.

Args:
    values: Eigen matrix of values to assign.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_assign_5[] =
R"doc(Assigns a RealMatrix to the block.

Args:
    values: RealMatrix of values.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_assign_6[] =
R"doc(Assigns a RealMatrix to the block.

Args:
    values: RealMatrix of values.

Returns:
    This submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_iadd[] =
R"doc(Compound addition-assignment operator.

Args:
    rhs: Real to add.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_iadd_2[] =
R"doc(Compound addition-assignment operator.

Args:
    rhs: Real to add.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_idiv[] =
R"doc(Compound matrix division-assignment operator.

Args:
    rhs: Real to divide.

Returns:
    Result of division.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_imul[] =
R"doc(Compound matrix multiplication-assignment operator.

Args:
    rhs: Real to multiply.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_imul_2[] =
R"doc(Compound matrix multiplication-assignment operator.

Args:
    rhs: Real to multiply.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_isub[] =
R"doc(Compound subtraction-assignment operator.

Args:
    rhs: Real to subtract.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_isub_2[] =
R"doc(Compound subtraction-assignment operator.

Args:
    rhs: Real to subtract.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_operator_slp_Real[] = R"doc(Implicit conversion operator from 1x1 RealSubmatrix to Real.)doc";

constexpr char mkd_doc_slp_RealSubmatrix_rbegin[] =
R"doc(Returns reverse begin iterator.

Returns:
    Reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_rbegin_2[] =
R"doc(Returns const reverse begin iterator.

Returns:
    Const reverse begin iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_rend[] =
R"doc(Returns reverse end iterator.

Returns:
    Reverse end iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_rend_2[] =
R"doc(Returns const reverse end iterator.

Returns:
    Const reverse end iterator.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_row[] =
R"doc(Returns a row slice of the submatrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_row_2[] =
R"doc(Returns a row slice of the submatrix.

Args:
    row: The row to slice.

Returns:
    A row slice of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_rows[] =
R"doc(Returns the number of rows in the matrix.

Returns:
    The number of rows in the matrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_segment[] =
R"doc(Returns a segment of the submatrix.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_segment_2[] =
R"doc(Returns a segment of the submatrix.

Args:
    offset: The offset of the segment.
    length: The length of the segment.

Returns:
    A segment of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_set_value[] =
R"doc(Assigns a scalar to the block.

This only works for blocks with one row and one column.

Args:
    value: Value to assign.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_set_value_2[] =
R"doc(Sets the block's internal values.

Args:
    values: Eigen matrix of values.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_size[] =
R"doc(Returns number of elements in matrix.

Returns:
    Number of elements in matrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_trace[] =
R"doc(Returns the trace of the submatrix, the sum of its diagonal.

Returns:
    The trace of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_value[] =
R"doc(Returns an element of the submatrix.

Args:
    row: The row of the element to return.
    col: The column of the element to return.

Returns:
    An element of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_value_2[] =
R"doc(Returns an element of the submatrix.

Args:
    index: The index of the element to return.

Returns:
    An element of the submatrix.

)doc";

constexpr char mkd_doc_slp_RealSubmatrix_value_3[] =
R"doc(Returns the contents of the submatrix.

Returns:
    The contents of the submatrix.

)doc";

constexpr char mkd_doc_slp_Real_Real[] = R"doc(Constructs a linear Real with a value of zero.)doc";

constexpr char mkd_doc_slp_Real_Real_2[] = R"doc(Constructs an empty Real.)doc";

constexpr char mkd_doc_slp_Real_Real_3[] =
R"doc(Constructs a Real from a scalar type.

Args:
    value: The value of the Real.

)doc";

constexpr char mkd_doc_slp_Real_Real_4[] =
R"doc(Constructs a Real from a scalar type.

Args:
    value: The value of the Real.

)doc";

constexpr char mkd_doc_slp_Real_Real_5[] =
R"doc(Constructs a Real from a floating-point type.

Args:
    value: The value of the Real.

)doc";

constexpr char mkd_doc_slp_Real_Real_6[] =
R"doc(Constructs a Real from an integral type.

Args:
    value: The value of the Real.

)doc";

constexpr char mkd_doc_slp_Real_Real_7[] =
R"doc(Constructs a Real pointing to the specified expression.

Args:
    expr: The autodiff variable.

)doc";

constexpr char mkd_doc_slp_Real_Real_8[] =
R"doc(Constructs a Real pointing to the specified expression.

Args:
    expr: The autodiff variable.

)doc";

constexpr char mkd_doc_slp_Real_expr[] = R"doc(The expression node)doc";

constexpr char mkd_doc_slp_Real_m_graph[] =
R"doc(Used to update the value of this variable based on the values of its
dependent variables)doc";

constexpr char mkd_doc_slp_Real_m_graph_initialized[] = R"doc(Used for lazy initialization of m_graph)doc";

constexpr char mkd_doc_slp_Real_operator_assign[] =
R"doc(Assignment operator for scalar.

Args:
    value: The value of the Real.

Returns:
    This variable.

)doc";

constexpr char mkd_doc_slp_Real_operator_iadd[] =
R"doc(Real-Real compound addition operator.

Args:
    rhs: Operator right-hand side.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_Real_operator_idiv[] =
R"doc(Real-Real compound division operator.

Args:
    rhs: Operator right-hand side.

Returns:
    Result of division.

)doc";

constexpr char mkd_doc_slp_Real_operator_imul[] =
R"doc(Real-Real compound multiplication operator.

Args:
    rhs: Operator right-hand side.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_Real_operator_isub[] =
R"doc(Real-Real compound subtraction operator.

Args:
    rhs: Operator right-hand side.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_Real_set_value[] =
R"doc(Sets Real's internal value.

Args:
    value: The value of the Real.

)doc";

constexpr char mkd_doc_slp_Real_type[] =
R"doc(Returns the type of this expression (constant, linear, quadratic, or
nonlinear).

Returns:
    The type of this expression.

)doc";

constexpr char mkd_doc_slp_Real_value[] =
R"doc(Returns the value of this variable.

Returns:
    The value of this variable.

)doc";

constexpr char mkd_doc_slp_TimestepMethod[] = R"doc(Enum describing the type of system timestep.)doc";

constexpr char mkd_doc_slp_TimestepMethod_FIXED[] = R"doc(The timestep is a fixed constant.)doc";

constexpr char mkd_doc_slp_TimestepMethod_VARIABLE[] = R"doc(The timesteps are allowed to vary as independent decision variables.)doc";

constexpr char mkd_doc_slp_TimestepMethod_VARIABLE_SINGLE[] =
R"doc(The timesteps are equal length but allowed to vary as a single
decision variable.)doc";

constexpr char mkd_doc_slp_TranscriptionMethod[] = R"doc(Enum describing an OCP transcription method.)doc";

constexpr char mkd_doc_slp_TranscriptionMethod_DIRECT_COLLOCATION[] =
R"doc(The trajectory is modeled as a series of cubic polynomials where the
centerpoint slope is constrained.)doc";

constexpr char mkd_doc_slp_TranscriptionMethod_DIRECT_TRANSCRIPTION[] =
R"doc(Each state is a decision variable constrained to the integrated
dynamics of the previous state.)doc";

constexpr char mkd_doc_slp_TranscriptionMethod_SINGLE_SHOOTING[] =
R"doc(States depend explicitly as a function of all previous states and all
previous inputs.)doc";

constexpr char mkd_doc_slp_Translation2[] =
R"doc(A 2D translation of autodiff expressions.

A translation is a point or a displacement (x, y). It converts
implicitly to a 2x1 RealMatrix.

Translations are created by Problem::translation2_decision_variable(),
from constants, and from expressions.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_Translation2_Translation2[] =
R"doc(Constructs a translation from its components.

Args:
    x: The x component.
    y: The y component.

)doc";

constexpr char mkd_doc_slp_Translation2_Translation2_2[] =
R"doc(Constructs a translation from a vector.

Args:
    vector: The vector (x, y). Must be 2x1.

)doc";

constexpr char mkd_doc_slp_Translation2_Translation2_3[] =
R"doc(Constructs a constant translation.

Args:
    vector: The vector (x, y). Must be 2x1.

)doc";

constexpr char mkd_doc_slp_Translation2_distance[] =
R"doc(Returns the distance to another translation.

The distance isn't differentiable where it's zero. For constraints and
costs that can reach zero distance, use squared_distance().

Args:
    other: The other translation.

Returns:
    The distance.

)doc";

constexpr char mkd_doc_slp_Translation2_m_x[] = R"doc()doc";

constexpr char mkd_doc_slp_Translation2_m_y[] = R"doc()doc";

constexpr char mkd_doc_slp_Translation2_operator_slp_RealMatrix[] =
R"doc(Implicit conversion to a 2x1 RealMatrix.

Returns:
    The translation as a 2x1 RealMatrix.

)doc";

constexpr char mkd_doc_slp_Translation2_set_value[] =
R"doc(Sets the translation's value.

This is only meaningful for translations of decision variables or
parameters.

Args:
    value: The value.

)doc";

constexpr char mkd_doc_slp_Translation2_squared_distance[] =
R"doc(Returns the squared distance to another translation.

Args:
    other: The other translation.

Returns:
    The squared distance.

)doc";

constexpr char mkd_doc_slp_Translation2_value[] =
R"doc(Returns the translation's value.

Returns:
    The translation's value.

)doc";

constexpr char mkd_doc_slp_Translation2_x[] =
R"doc(Returns the x component.

Returns:
    The x component.

)doc";

constexpr char mkd_doc_slp_Translation2_x_2[] =
R"doc(Returns the x component.

Returns:
    The x component.

)doc";

constexpr char mkd_doc_slp_Translation2_y[] =
R"doc(Returns the y component.

Returns:
    The y component.

)doc";

constexpr char mkd_doc_slp_Translation2_y_2[] =
R"doc(Returns the y component.

Returns:
    The y component.

)doc";

constexpr char mkd_doc_slp_Translation3[] =
R"doc(A 3D translation of autodiff expressions.

A translation is a point or a displacement (x, y, z). It converts
implicitly to a 3x1 RealMatrix.

Translations are created by Problem::translation3_decision_variable(),
from constants, and from expressions.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_Translation3_Translation3[] =
R"doc(Constructs a translation from its components.

Args:
    x: The x component.
    y: The y component.
    z: The z component.

)doc";

constexpr char mkd_doc_slp_Translation3_Translation3_2[] =
R"doc(Constructs a translation from a vector.

Args:
    vector: The vector (x, y, z). Must be 3x1.

)doc";

constexpr char mkd_doc_slp_Translation3_Translation3_3[] =
R"doc(Constructs a constant translation.

Args:
    vector: The vector (x, y, z). Must be 3x1.

)doc";

constexpr char mkd_doc_slp_Translation3_distance[] =
R"doc(Returns the distance to another translation.

The distance isn't differentiable where it's zero. For constraints and
costs that can reach zero distance, use squared_distance().

Args:
    other: The other translation.

Returns:
    The distance.

)doc";

constexpr char mkd_doc_slp_Translation3_m_x[] = R"doc()doc";

constexpr char mkd_doc_slp_Translation3_m_y[] = R"doc()doc";

constexpr char mkd_doc_slp_Translation3_m_z[] = R"doc()doc";

constexpr char mkd_doc_slp_Translation3_operator_slp_RealMatrix[] =
R"doc(Implicit conversion to a 3x1 RealMatrix.

Returns:
    The translation as a 3x1 RealMatrix.

)doc";

constexpr char mkd_doc_slp_Translation3_set_value[] =
R"doc(Sets the translation's value.

This is only meaningful for translations of decision variables or
parameters.

Args:
    value: The value.

)doc";

constexpr char mkd_doc_slp_Translation3_squared_distance[] =
R"doc(Returns the squared distance to another translation.

Args:
    other: The other translation.

Returns:
    The squared distance.

)doc";

constexpr char mkd_doc_slp_Translation3_value[] =
R"doc(Returns the translation's value.

Returns:
    The translation's value.

)doc";

constexpr char mkd_doc_slp_Translation3_x[] =
R"doc(Returns the x component.

Returns:
    The x component.

)doc";

constexpr char mkd_doc_slp_Translation3_x_2[] =
R"doc(Returns the x component.

Returns:
    The x component.

)doc";

constexpr char mkd_doc_slp_Translation3_y[] =
R"doc(Returns the y component.

Returns:
    The y component.

)doc";

constexpr char mkd_doc_slp_Translation3_y_2[] =
R"doc(Returns the y component.

Returns:
    The y component.

)doc";

constexpr char mkd_doc_slp_Translation3_z[] =
R"doc(Returns the z component.

Returns:
    The z component.

)doc";

constexpr char mkd_doc_slp_Translation3_z_2[] =
R"doc(Returns the z component.

Returns:
    The z component.

)doc";

constexpr char mkd_doc_slp_UnitVector[] =
R"doc(A unit-length column vector of autodiff expressions.

UnitVectors are created by Problem::unit_vector_decision_variable(n),
which keeps them on the unit sphere during the solve, from constant
Eigen vectors, and by operations that preserve unit length, like
negation. Operations that don't preserve unit length, like addition
and scaling, return a RealMatrix.

Elements are read-only. A UnitVector converts implicitly to a
RealMatrix for use in general expressions.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_UnitVector_2[] =
R"doc(A unit-length column vector of autodiff expressions.

UnitVectors are created by Problem::unit_vector_decision_variable(n),
which keeps them on the unit sphere during the solve, from constant
Eigen vectors, and by operations that preserve unit length, like
negation. Operations that don't preserve unit length, like addition
and scaling, return a RealMatrix.

Elements are read-only. A UnitVector converts implicitly to a
RealMatrix for use in general expressions.

Template Args:
    Scalar_: Scalar type.)doc";

constexpr char mkd_doc_slp_UnitVector_T[] =
R"doc(Returns the transpose.

Returns:
    The transpose.

)doc";

constexpr char mkd_doc_slp_UnitVector_UnitVector[] =
R"doc(Constructs a constant unit vector from the normalized values.

Args:
    values: The values. They must form a column vector with nonzero
            norm.

)doc";

constexpr char mkd_doc_slp_UnitVector_UnitVector_2[] = R"doc()doc";

constexpr char mkd_doc_slp_UnitVector_begin[] =
R"doc(Returns an iterator to the beginning of the vector.

Returns:
    An iterator to the beginning of the vector.

)doc";

constexpr char mkd_doc_slp_UnitVector_cols[] =
R"doc(Returns the number of columns, which is always 1.

Returns:
    The number of columns.

)doc";

constexpr char mkd_doc_slp_UnitVector_cross[] =
R"doc(Returns the cross product with another vector. Both vectors must be
3D.

Args:
    rhs: The other vector.

Returns:
    The cross product.

)doc";

constexpr char mkd_doc_slp_UnitVector_dot[] =
R"doc(Returns the dot product with another vector.

Args:
    rhs: The other vector.

Returns:
    The dot product.

)doc";

constexpr char mkd_doc_slp_UnitVector_end[] =
R"doc(Returns an iterator to the end of the vector.

Returns:
    An iterator to the end of the vector.

)doc";

constexpr char mkd_doc_slp_UnitVector_m_storage[] = R"doc()doc";

constexpr char mkd_doc_slp_UnitVector_operator_array[] =
R"doc(Returns an element.

Args:
    row: The row of the element to return.

Returns:
    An element.

)doc";

constexpr char mkd_doc_slp_UnitVector_operator_array_2[] =
R"doc(Returns an element.

Args:
    row: The row of the element to return.
    col: The column of the element to return, which must be 0.

Returns:
    An element.

)doc";

constexpr char mkd_doc_slp_UnitVector_operator_slp_RealMatrix[] =
R"doc(Converts the vector to a RealMatrix.

Returns:
    The vector as a RealMatrix.

)doc";

constexpr char mkd_doc_slp_UnitVector_rows[] =
R"doc(Returns the number of rows.

Returns:
    The number of rows.

)doc";

constexpr char mkd_doc_slp_UnitVector_set_value[] =
R"doc(Sets the vector's values to the normalized values.

This is only meaningful for UnitVectors created by
Problem::unit_vector_decision_variable(n) or
Problem::unit_vector_parameter(n).

Args:
    values: The values. They must have nonzero norm.

)doc";

constexpr char mkd_doc_slp_UnitVector_value[] =
R"doc(Returns an element's value.

Args:
    row: The row of the element whose value to return.

Returns:
    An element's value.

)doc";

constexpr char mkd_doc_slp_UnitVector_value_2[] =
R"doc(Returns the vector's values.

Returns:
    The vector's values.

)doc";

constexpr char mkd_doc_slp_abs[] =
R"doc(abs() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_acos[] =
R"doc(acos() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_asin[] =
R"doc(asin() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_atan[] =
R"doc(atan() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_atan2[] =
R"doc(atan2() for Reals.

Args:
    y: The y argument.
    x: The x argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_atan2_2[] =
R"doc(atan2() for Reals.

Args:
    y: The y argument.
    x: The x argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_atan2_3[] =
R"doc(atan2() for Reals.

Args:
    y: The y argument.
    x: The x argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_block[] =
R"doc(Assembles a RealMatrix from a nested list of blocks.

Each row's blocks must have the same height, and the assembled block
rows must have the same width. For example, for the block matrix [[A,
B], [C]] to be constructible, the number of rows in A and B must
match, and the number of columns in [A, B] and [C] must match.

Args:
    list: The nested list of blocks.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_block_2[] =
R"doc(Assembles a RealMatrix from a nested list of blocks.

Each row's blocks must have the same height, and the assembled block
rows must have the same width. For example, for the block matrix [[A,
B], [C]] to be constructible, the number of rows in A and B must
match, and the number of columns in [A, B] and [C] must match.

This overload is for Python bindings only.

Args:
    list: The nested list of blocks.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_bounds[] =
R"doc(Helper function for creating bound constraints.

Args:
    l: Lower bound.
    x: Real to bound.
    u: Upper bound.

)doc";

constexpr char mkd_doc_slp_cbrt[] =
R"doc(cbrt() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_cos[] =
R"doc(cos() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_cosh[] =
R"doc(cosh() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_cwise_reduce[] =
R"doc(Applies a coefficient-wise reduce operation to two matrices.

Args:
    lhs: The left-hand side of the binary operator.
    rhs: The right-hand side of the binary operator.
    binary_op: The binary operator to use for the reduce operation.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_detail_ComplexMatrixOperandTraits[] =
R"doc(Classifies an operand of complex matrix arithmetic.

Template Args:
    T: The operand type.)doc";

constexpr char mkd_doc_slp_detail_ExpressionRole[] = R"doc()doc";

constexpr char mkd_doc_slp_detail_ExpressionRole_2[] = R"doc(Where an expression appears in a problem.)doc";

constexpr char mkd_doc_slp_detail_ExpressionRole_COST[] = R"doc(The cost function.)doc";

constexpr char mkd_doc_slp_detail_ExpressionRole_EQUALITY_CONSTRAINT[] = R"doc(An equality constraint.)doc";

constexpr char mkd_doc_slp_detail_ExpressionRole_INEQUALITY_CONSTRAINT[] = R"doc(An inequality constraint.)doc";

constexpr char mkd_doc_slp_detail_atan2_over_norm[] =
R"doc(Returns atan2(n, w)/n, where n = √n².

Near n = 0, atan2(n, w)/n loses precision, and its derivatives are
infinite at n = 0. Where w > 0 and n is small relative to w, this uses

  atan2(n, w)/n = 1/w atan(√t)/√t
                = 1/w (1 − t/3 + t²/5 − t³/7 + …)

with t = n²/w², a series in n² that's smooth at n = 0. Truncating it
after t⁷ makes its Hessian error grow like n¹⁴, and the atan2 form's
Hessian roundoff grows like ε/n³, so they cross at t ≈ ε^(2/17).

The result is infinite at n = 0 with w < 0.

Args:
    n_sq: n².
    w: w.

Returns:
    atan2(n, w)/n.

)doc";

constexpr char mkd_doc_slp_detail_capitalize[] =
R"doc(Returns the string with its first letter capitalized.

Args:
    str: The string.

Returns:
    The capitalized string.

)doc";

constexpr char mkd_doc_slp_detail_complex_entry[] =
R"doc(Returns entry (row, col) of a matrix operand as a Complex.

Args:
    m: The matrix operand.
    row: The entry's row.
    col: The entry's column.

Template Args:
    Scalar: Scalar type.

Returns:
    The entry.

)doc";

constexpr char mkd_doc_slp_detail_describe_constraints[] =
R"doc(Describes constraints as "equality constraint i", "inequality
constraints i and j", and so on.

Args:
    kind: "equality" or "inequality".
    indices: The constraint indices.

Returns:
    The description.

)doc";

constexpr char mkd_doc_slp_detail_describe_decision_variables[] =
R"doc(Describes decision variables as "x[i]", "x[i] and x[j]", and so on.

Args:
    indices: The decision variable indices.

Returns:
    The description.

)doc";

constexpr char mkd_doc_slp_detail_gradient_tree[] = R"doc()doc";

constexpr char mkd_doc_slp_detail_gradient_tree_2[] =
R"doc(Returns the variable's gradient tree.

This function lazily allocates variables, so elements of the returned
RealMatrix will be empty if the corresponding element of wrt had no
adjoint. Ensure Real::expr isn't nullptr before calling member
functions.

Args:
    top_list: Topologically sorted graph from parent to child.
    wrt: Variables with respect to which to compute the gradient.

Template Args:
    Scalar: Scalar type.

Returns:
    The variable's gradient tree.

)doc";

constexpr char mkd_doc_slp_detail_is_constant[] =
R"doc(Returns whether every element of a unit vector is constant.

Args:
    v: The unit vector.

Returns:
    Whether every element is constant.

)doc";

constexpr char mkd_doc_slp_detail_is_sum_of_squares[] =
R"doc(Returns true if an expression is a sum of squares of expressions of
decision variables, like a vector's squared norm.

Args:
    node: The expression.

Template Args:
    Scalar: Scalar type.

Returns:
    True if the expression is a sum of squares.

)doc";

constexpr char mkd_doc_slp_detail_join_english[] =
R"doc(Joins items into an English list ("a", "a and b", or "a, b, and c").

Args:
    items: The items.

Returns:
    The list.

)doc";

constexpr char mkd_doc_slp_detail_lint[] =
R"doc(Checks a problem's formulation for issues and suggests improvements.

See Problem::lint().

Args:
    decision_variables: The decision variables.
    cost: The cost function, if any.
    equality_constraints: The equality constraints cₑ(x) = 0.
    inequality_constraints: The inequality constraints cᵢ(x) ≥ 0.

Template Args:
    Scalar: Scalar type.

Returns:
    The findings.

)doc";

constexpr char mkd_doc_slp_detail_near[] =
R"doc(Returns true if two values are equal to within a relative tolerance.

Args:
    a: The first value.
    b: The second value.
    tolerance: The relative tolerance.

Template Args:
    Scalar: Scalar type.

Returns:
    True if the values are equal to within the tolerance.

)doc";

constexpr char mkd_doc_slp_detail_replaceable_nonsmooth_functions[] = R"doc()doc";

constexpr char mkd_doc_slp_detail_replaceable_nonsmooth_functions_2[] =
R"doc(Returns the functions of decision variables in an expression that
aren't differentiable everywhere and appear in a form with a
differentiable replacement.

The forms and their replacements are:


  

* "abs": abs(a) ≤ b → −b ≤ a ≤ b


  

* "max": max(a, b) ≤ c → a ≤ c and b ≤ c


  

* "min": min(a, b) ≥ c → a ≥ c and b ≥ c


  

* "norm": a norm (sqrt() of a sum of squares or hypot()) compared to a
    value that doesn't depend on decision variables → the squared norm
    compared to the value's square


  

* "cost norm": a cost function that's a norm → the squared norm, which
    has the same minimizer




Args:
    x: The expression.
    role: Where the expression appears in the problem.

Template Args:
    Scalar: Scalar type.

Returns:
    The forms found, without duplicates.

)doc";

constexpr char mkd_doc_slp_detail_sphere_log[] =
R"doc(Returns the sphere's logarithm map log_b(a) in an orthonormal basis of
b's tangent space.

log_b(a) is the tangent vector at b that points toward a, with length
equal to the angle between a and b. Its only zero is a = b.

The basis comes from a Householder reflection built around b's current
value. It's valid unless b later moves to that reflection's one
singular point, which is more than 90° from b's current value.

Args:
    a: The unit vector the logarithm points to.
    b: The unit vector whose tangent space the logarithm is in.

Returns:
    The n − 1 components of log_b(a).

)doc";

constexpr char mkd_doc_slp_erf[] =
R"doc(erf() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_exp[] =
R"doc(exp() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_hypot[] =
R"doc(hypot() for Reals.

Args:
    x: The x argument.
    y: The y argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_hypot_2[] =
R"doc(hypot() for Reals.

Args:
    x: The x argument.
    y: The y argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_hypot_3[] =
R"doc(hypot() for Reals.

Args:
    x: The x argument.
    y: The y argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_hypot_4[] =
R"doc(hypot() for Reals.

Args:
    x: The x argument.
    y: The y argument.
    z: The z argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_if_else[] =
R"doc(Returns t if the equality constraint cond is satisfied, otherwise f.

For example, if_else(x == y, t, f) returns t if x = y, otherwise f.

The condition is evaluated on the current values of its operands each
time the result is evaluated, and the gradient only flows through the
selected branch. Both branches are evaluated regardless of which one
is selected.

Args:
    cond: A scalar equality constraint.
    t: Value selected when cond is satisfied.
    f: Value selected when cond is violated.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_if_else_2[] =
R"doc(Returns t if the inequality constraint cond is satisfied, otherwise f.

For example, if_else(x < y, t, f) returns t if x < y, otherwise f.
Strict comparisons (< and >) exclude equality and non-strict
comparisons (≤ and ≥) include it.

The condition is evaluated on the current values of its operands each
time the result is evaluated, and the gradient only flows through the
selected branch. Both branches are evaluated regardless of which one
is selected.

Args:
    cond: A scalar inequality constraint.
    t: Value selected when cond is satisfied.
    f: Value selected when cond is violated.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_log[] =
R"doc(log() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_log10[] =
R"doc(log10() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_make_constraints[] = R"doc()doc";

constexpr char mkd_doc_slp_make_constraints_2[] = R"doc()doc";

constexpr char mkd_doc_slp_make_constraints_3[] = R"doc()doc";

constexpr char mkd_doc_slp_make_constraints_4[] = R"doc()doc";

constexpr char mkd_doc_slp_max[] =
R"doc(max() for Reals.

Returns the greater of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_max_2[] =
R"doc(max() for Reals.

Returns the greater of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_max_3[] =
R"doc(max() for Reals.

Returns the greater of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_min[] =
R"doc(min() for Reals.

Returns the lesser of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_min_2[] =
R"doc(min() for Reals.

Returns the lesser of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_min_3[] =
R"doc(min() for Reals.

Returns the lesser of a and b. If the values are equivalent, returns
a.

Args:
    a: The a argument.
    b: The b argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_operator_add[] =
R"doc(Adds two complex numbers.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The sum.

)doc";

constexpr char mkd_doc_slp_operator_add_2[] =
R"doc(Adds two complex numbers.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The sum.

)doc";

constexpr char mkd_doc_slp_operator_add_3[] =
R"doc(Adds a Real and a constant complex number.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The sum.

)doc";

constexpr char mkd_doc_slp_operator_add_4[] =
R"doc(Adds a constant complex number and a Real.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The sum.

)doc";

constexpr char mkd_doc_slp_operator_add_5[] =
R"doc(Returns a complex number unchanged.

Args:
    z: The complex number.

Returns:
    The complex number.

)doc";

constexpr char mkd_doc_slp_operator_add_6[] =
R"doc(Binary addition operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    Result of addition.

)doc";

constexpr char mkd_doc_slp_operator_add_7[] =
R"doc(Unary plus operator.

Args:
    z: Operand for unary plus.

Returns:
    The matrix.

)doc";

constexpr char mkd_doc_slp_operator_add_8[] =
R"doc(Adds two translations.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The sum.

)doc";

constexpr char mkd_doc_slp_operator_add_9[] =
R"doc(Adds two translations.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The sum.

)doc";

constexpr char mkd_doc_slp_operator_add_10[] =
R"doc(Unary plus operator.

Args:
    v: The vector.

Returns:
    The vector.

)doc";

constexpr char mkd_doc_slp_operator_add_11[] =
R"doc(Addition operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The sum.

)doc";

constexpr char mkd_doc_slp_operator_add_12[] =
R"doc(Addition operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The sum.

)doc";

constexpr char mkd_doc_slp_operator_add_13[] =
R"doc(Addition operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The sum.

)doc";

constexpr char mkd_doc_slp_operator_div[] =
R"doc(Divides two complex numbers.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The quotient.

)doc";

constexpr char mkd_doc_slp_operator_div_2[] =
R"doc(Divides two complex numbers.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The quotient.

)doc";

constexpr char mkd_doc_slp_operator_div_3[] =
R"doc(Divides a Real by a constant complex number.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The quotient.

)doc";

constexpr char mkd_doc_slp_operator_div_4[] =
R"doc(Divides a constant complex number by a Real.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The quotient.

)doc";

constexpr char mkd_doc_slp_operator_div_5[] =
R"doc(Binary division operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    Result of division.

)doc";

constexpr char mkd_doc_slp_operator_div_6[] =
R"doc(Binary division operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Template Args:
    Scalar: Scalar type.

Returns:
    Result of division.

)doc";

constexpr char mkd_doc_slp_operator_div_7[] =
R"doc(Divides a translation by a scalar.

Args:
    lhs: The translation.
    rhs: The divisor.

Returns:
    The divided translation.

)doc";

constexpr char mkd_doc_slp_operator_div_8[] =
R"doc(Divides a translation by a scalar.

Args:
    lhs: The translation.
    rhs: The divisor.

Returns:
    The divided translation.

)doc";

constexpr char mkd_doc_slp_operator_div_9[] =
R"doc(Division operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The quotient.

)doc";

constexpr char mkd_doc_slp_operator_eq[] =
R"doc(Equality operator that returns equality constraints on the real and
imaginary parts of two complex numbers.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side.

Returns:
    The equality constraints.

)doc";

constexpr char mkd_doc_slp_operator_eq_2[] =
R"doc(Equality operator that returns equality constraints on the real and
imaginary parts of a complex number and another operand.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side. It's a Real, a std::complex, or a real
         constant.

Returns:
    The equality constraints.

)doc";

constexpr char mkd_doc_slp_operator_eq_3[] =
R"doc(Equality operator that returns equality constraints on the real and
imaginary parts of another operand and a complex number.

Args:
    lhs: Left-hand side. It's a Real, a std::complex, or a real
         constant.
    rhs: Right-hand side.

Returns:
    The equality constraints.

)doc";

constexpr char mkd_doc_slp_operator_eq_4[] =
R"doc(Equality operator that returns equality constraints on the real and
imaginary parts of each entry of two matrices.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side.

Returns:
    The equality constraints.

)doc";

constexpr char mkd_doc_slp_operator_eq_5[] =
R"doc(Equality operator that returns equality constraints on the real and
imaginary parts of each entry of a matrix and a scalar.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side.

Returns:
    The equality constraints.

)doc";

constexpr char mkd_doc_slp_operator_eq_6[] =
R"doc(Equality operator that returns equality constraints on the real and
imaginary parts of each entry of a scalar and a matrix.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side.

Returns:
    The equality constraints.

)doc";

constexpr char mkd_doc_slp_operator_eq_7[] =
R"doc(Equality operator that returns an equality constraint for two Reals.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_eq_8[] =
R"doc(Equality operator that returns an equality constraint for two Reals.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_eq_9[] =
R"doc(Equality operator that returns an equality constraint for two Reals.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_eq_10[] =
R"doc(Equality operator that returns an equality constraint for two
translations.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side.

Returns:
    The equality constraint.

)doc";

constexpr char mkd_doc_slp_operator_eq_11[] =
R"doc(Equality operator that returns an equality constraint for two
translations.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side.

Returns:
    The equality constraint.

)doc";

constexpr char mkd_doc_slp_operator_eq_12[] =
R"doc(Equality operator that returns an equality constraint for two unit
vectors.

The constraint is that the angle between the vectors is zero. Unit
vectors in ℝⁿ have n − 1 degrees of freedom, so it's n − 1 equations
instead of the n redundant equations of lhs − rhs = 0. Its only
solution is lhs = rhs.

If both vectors are decision variables, the constraint is built around
rhs's value when it's created, so set initial values first.

Args:
    lhs: Left-hand side.
    rhs: Right-hand side.

Returns:
    The equality constraint.

)doc";

constexpr char mkd_doc_slp_operator_eq_13[] =
R"doc(Equality operator that returns an equality constraint for a unit
vector and a constant direction.

The direction is normalized. See operator==(const UnitVector&, const
UnitVector&).

Args:
    lhs: Left-hand side.
    rhs: Right-hand side. It must have nonzero norm.

Returns:
    The equality constraint.

)doc";

constexpr char mkd_doc_slp_operator_eq_14[] =
R"doc(Equality operator that returns an equality constraint for a constant
direction and a unit vector.

The direction is normalized. See operator==(const UnitVector&, const
UnitVector&).

Args:
    lhs: Left-hand side. It must have nonzero norm.
    rhs: Right-hand side.

Returns:
    The equality constraint.

)doc";

constexpr char mkd_doc_slp_operator_ge[] =
R"doc(Greater-than-or-equal-to comparison operator that returns an
inequality constraint for two Reals.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_ge_2[] =
R"doc(Greater-than-or-equal-to comparison operator that returns an
inequality constraint for two Reals.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_ge_3[] =
R"doc(Greater-than-or-equal-to comparison operator that returns an
inequality constraint for two Reals.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_gt[] =
R"doc(Greater-than comparison operator that returns an inequality constraint
for two Reals.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_le[] =
R"doc(Less-than-or-equal-to comparison operator that returns an inequality
constraint for two Reals.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_lt[] =
R"doc(Less-than comparison operator that returns an inequality constraint
for two Reals.

Args:
    lhs: Left-hand side.
    rhs: Left-hand side.

)doc";

constexpr char mkd_doc_slp_operator_mul[] =
R"doc(Multiplies two complex numbers.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The product.

)doc";

constexpr char mkd_doc_slp_operator_mul_2[] =
R"doc(Multiplies two complex numbers.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The product.

)doc";

constexpr char mkd_doc_slp_operator_mul_3[] =
R"doc(Multiplies a Real by a constant complex number.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The product.

)doc";

constexpr char mkd_doc_slp_operator_mul_4[] =
R"doc(Multiplies a constant complex number by a Real.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The product.

)doc";

constexpr char mkd_doc_slp_operator_mul_5[] =
R"doc(Matrix multiplication operator.

Products with Eigen constants skip their zero entries, so a sparse
constant matrix like a bus admittance matrix only builds expressions
for its nonzeros.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_operator_mul_6[] =
R"doc(Matrix-scalar multiplication operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_operator_mul_7[] =
R"doc(Scalar-matrix multiplication operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    Result of multiplication.

)doc";

constexpr char mkd_doc_slp_operator_mul_8[] =
R"doc(Matrix-scalar multiplication operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_operator_mul_9[] =
R"doc(Scalar-matrix multiplication operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_operator_mul_10[] =
R"doc(Scales a translation.

Args:
    lhs: The translation.
    rhs: The factor.

Returns:
    The scaled translation.

)doc";

constexpr char mkd_doc_slp_operator_mul_11[] =
R"doc(Scales a translation.

Args:
    lhs: The factor.
    rhs: The translation.

Returns:
    The scaled translation.

)doc";

constexpr char mkd_doc_slp_operator_mul_12[] =
R"doc(Scales a translation.

Args:
    lhs: The translation.
    rhs: The factor.

Returns:
    The scaled translation.

)doc";

constexpr char mkd_doc_slp_operator_mul_13[] =
R"doc(Scales a translation.

Args:
    lhs: The factor.
    rhs: The translation.

Returns:
    The scaled translation.

)doc";

constexpr char mkd_doc_slp_operator_mul_14[] =
R"doc(Multiplication operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The product.

)doc";

constexpr char mkd_doc_slp_operator_mul_15[] =
R"doc(Multiplication operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The product.

)doc";

constexpr char mkd_doc_slp_operator_mul_16[] =
R"doc(Multiplication operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The product.

)doc";

constexpr char mkd_doc_slp_operator_mul_17[] =
R"doc(Multiplication operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The product.

)doc";

constexpr char mkd_doc_slp_operator_sub[] =
R"doc(Subtracts two complex numbers.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The difference.

)doc";

constexpr char mkd_doc_slp_operator_sub_2[] =
R"doc(Subtracts two complex numbers.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The difference.

)doc";

constexpr char mkd_doc_slp_operator_sub_3[] =
R"doc(Subtracts a constant complex number from a Real.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The difference.

)doc";

constexpr char mkd_doc_slp_operator_sub_4[] =
R"doc(Subtracts a Real from a constant complex number.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The difference.

)doc";

constexpr char mkd_doc_slp_operator_sub_5[] =
R"doc(Negates a complex number.

Args:
    z: The complex number.

Returns:
    The negated complex number.

)doc";

constexpr char mkd_doc_slp_operator_sub_6[] =
R"doc(Binary subtraction operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    Result of subtraction.

)doc";

constexpr char mkd_doc_slp_operator_sub_7[] =
R"doc(Unary minus operator.

Args:
    z: Operand for unary minus.

Returns:
    The negated matrix.

)doc";

constexpr char mkd_doc_slp_operator_sub_8[] =
R"doc(Subtracts two translations.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The difference.

)doc";

constexpr char mkd_doc_slp_operator_sub_9[] =
R"doc(Negates a translation.

Args:
    translation: The translation.

Returns:
    The negated translation.

)doc";

constexpr char mkd_doc_slp_operator_sub_10[] =
R"doc(Subtracts two translations.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The difference.

)doc";

constexpr char mkd_doc_slp_operator_sub_11[] =
R"doc(Negates a translation.

Args:
    translation: The translation.

Returns:
    The negated translation.

)doc";

constexpr char mkd_doc_slp_operator_sub_12[] =
R"doc(Unary minus operator.

Args:
    v: The vector.

Returns:
    The negated vector, which is also unit-length.

)doc";

constexpr char mkd_doc_slp_operator_sub_13[] =
R"doc(Subtraction operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The difference.

)doc";

constexpr char mkd_doc_slp_operator_sub_14[] =
R"doc(Subtraction operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The difference.

)doc";

constexpr char mkd_doc_slp_operator_sub_15[] =
R"doc(Subtraction operator.

Args:
    lhs: Operator left-hand side.
    rhs: Operator right-hand side.

Returns:
    The difference.

)doc";

constexpr char mkd_doc_slp_pow[] =
R"doc(pow() for Reals.

Args:
    base: The base.
    power: The power.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_pow_2[] =
R"doc(pow() for Reals.

Args:
    base: The base.
    power: The power.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_pow_3[] =
R"doc(pow() for Reals.

Args:
    base: The base.
    power: The power.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_sign[] =
R"doc(sign() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_sin[] =
R"doc(sin() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_sinh[] =
R"doc(sinh() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_solve[] =
R"doc(Solves the matrix equation AX = B for X.

Args:
    A: The left-hand side.
    B: The right-hand side.

Template Args:
    Scalar: Scalar type.

Returns:
    The solution X.

)doc";

constexpr char mkd_doc_slp_sqrt[] =
R"doc(sqrt() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_tan[] =
R"doc(tan() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

constexpr char mkd_doc_slp_tanh[] =
R"doc(tanh() for Reals.

Args:
    x: The argument.

Template Args:
    Scalar: Scalar type.

)doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

