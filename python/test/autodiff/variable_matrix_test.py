import math
import sys

import numpy as np
from sleipnir import autodiff
from sleipnir.autodiff import ExpressionType, Variable, VariableMatrix

# Machine epsilon for the binding's scalar type (double)
ε = sys.float_info.epsilon


def test_mixed_list_constructor() -> None:
    x = Variable()
    x.set_value(3.0)

    # float and int elements are implicitly converted to constant Variables
    mat = VariableMatrix([[x, 1.0], [2, x]])

    assert mat.shape == (2, 2)
    assert (mat.value() == np.array([[3.0, 1.0], [2.0, 3.0]])).all()
    assert mat[0, 0].type() == ExpressionType.LINEAR
    assert mat[0, 1].type() == ExpressionType.CONSTANT
    assert mat[1, 0].type() == ExpressionType.CONSTANT


def test_assignment_to_default() -> None:
    mat = VariableMatrix()

    assert mat.rows() == 0
    assert mat.cols() == 0
    assert mat.shape == (0, 0)

    mat = VariableMatrix(2, 2)

    assert mat.rows() == 2
    assert mat.cols() == 2
    assert mat.shape == (2, 2)
    assert mat[0, 0].value() == 0.0
    assert mat[0, 1].value() == 0.0
    assert mat[1, 0].value() == 0.0
    assert mat[1, 1].value() == 0.0

    mat[0, 0].set_value(1.0)
    mat[0, 1].set_value(2.0)
    mat[1, 0].set_value(3.0)
    mat[1, 1].set_value(4.0)

    assert mat[0, 0].value() == 1.0
    assert mat[0, 1].value() == 2.0
    assert mat[1, 0].value() == 3.0
    assert mat[1, 1].value() == 4.0


def test_assignment_aliasing() -> None:
    A = VariableMatrix([[1.0, 2.0], [3.0, 4.0]])
    B = VariableMatrix([[5.0, 6.0], [7.0, 8.0]])

    # A and B initially contain different values
    expected_A = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_B = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert (A.value() == expected_A).all()
    assert (B.value() == expected_B).all()

    # Make A point to B's storage
    A = B
    assert (A.value() == expected_B).all()
    assert (B.value() == expected_B).all()

    # Changes to B should be reflected in A
    B[0, 0].set_value(2.0)
    expected_B[0, 0] = 2.0
    assert (A.value() == expected_B).all()
    assert (B.value() == expected_B).all()


def test_slicing() -> None:
    mat = VariableMatrix(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    assert mat.shape == (4, 4)

    # Single-arg index operator on full matrix
    for i in range(mat.shape[0] * mat.shape[1]):
        assert mat[i] == i + 1

    # Slice from start
    s = mat[1:, 2:]
    assert s.shape == (3, 2)
    # Single-arg index operator on forward slice
    assert s[0] == 7.0
    assert s[1] == 8.0
    assert s[2] == 11.0
    assert s[3] == 12.0
    assert s[4] == 15.0
    assert s[5] == 16.0
    # Double-arg index operator on forward slice
    assert s[0, 0] == 7.0
    assert s[0, 1] == 8.0
    assert s[1, 0] == 11.0
    assert s[1, 1] == 12.0
    assert s[2, 0] == 15.0
    assert s[2, 1] == 16.0

    # Slice from end
    s = mat[-1:, -2:]
    assert s.shape == (1, 2)
    # Single-arg index operator on reverse slice
    assert s[0] == 15.0
    assert s[1] == 16.0
    # Double-arg index operator on reverse slice
    assert s[0, 0] == 15.0
    assert s[0, 1] == 16.0

    # Slice from start with step of 2
    s = mat[:, ::2]
    assert s.shape == (4, 2)
    assert (
        s.value() == np.array([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0], [13.0, 15.0]])
    ).all()

    # Slice from end with negative step for row and column
    s = mat[::-1, ::-2]
    assert s.shape == (4, 2)
    assert (
        s.value()
        == np.array(
            [
                [16.0, 14.0],
                [12.0, 10.0],
                [8.0, 6.0],
                [4.0, 2.0],
            ]
        )
    ).all()

    # Slice from start and column -1
    s = mat[1:, -1]
    assert s.shape == (3, 1)
    assert (s.value() == np.array([[8.0], [12.0], [16.0]])).all()

    # Slice from start and column -2
    s = mat[1:, -2]
    assert s.shape == (3, 1)
    assert (s.value() == np.array([[7.0], [11.0], [15.0]])).all()

    # Block assignment
    assert mat[::2, ::2].shape == (2, 2)
    mat[::2, ::2] = np.array([[17.0, 18.0], [19.0, 20.0]])
    assert (
        mat.value()
        == np.array(
            [
                [17.0, 2.0, 18.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [19.0, 10.0, 20.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        )
    ).all()


def test_subslicing() -> None:
    # Block-of-block assignment (row skip forward)
    mat = VariableMatrix(5, 5)
    assert mat[::2, ::1][1:3, 1:4].shape == (2, 3)
    mat[::2, ::1][1:3, 1:4] = np.array([[1, 2, 3], [4, 5, 6]])

    assert (
        mat.value()
        == np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 2, 3, 0],
                [0, 0, 0, 0, 0],
                [0, 4, 5, 6, 0],
            ]
        )
    ).all()

    # Block-of-block assignment (row skip backward)
    mat = VariableMatrix(5, 5)
    assert mat[::-2, ::-1][1:3, 1:4].shape == (2, 3)
    mat[::-2, ::-1][1:3, 1:4] = np.array([[1, 2, 3], [4, 5, 6]])

    assert (
        mat.value()
        == np.array(
            [
                [0, 6, 5, 4, 0],
                [0, 0, 0, 0, 0],
                [0, 3, 2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
    ).all()

    # Block-of-block assignment (column skip forward)
    mat = VariableMatrix(5, 5)
    assert mat[::1, ::2][1:4, 1:3].shape == (3, 2)
    mat[::1, ::2][1:4, 1:3] = np.array([[1, 2], [3, 4], [5, 6]])

    assert (
        mat.value()
        == np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 2],
                [0, 0, 3, 0, 4],
                [0, 0, 5, 0, 6],
                [0, 0, 0, 0, 0],
            ]
        )
    ).all()

    # Block-of-block assignment (column skip backward)
    mat = VariableMatrix(5, 5)
    assert mat[::-1, ::-2][1:4, 1:3].shape == (3, 2)
    mat[::-1, ::-2][1:4, 1:3] = np.array([[1, 2], [3, 4], [5, 6]])

    assert (
        mat.value()
        == np.array(
            [
                [0, 0, 0, 0, 0],
                [6, 0, 5, 0, 0],
                [4, 0, 3, 0, 0],
                [2, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
    ).all()


def test_compound_assignment_operators() -> None:
    A1 = VariableMatrix([[1]])
    A2 = VariableMatrix([[1, 2], [3, 4]])
    B = VariableMatrix([[1, 2], [3, 4]])
    b = 2

    # VariableMatrix matrix-matrix operators
    A2 += B
    assert (A2.value() == np.array([[2, 4], [6, 8]])).all()
    A2 -= B
    assert (A2.value() == np.array([[1, 2], [3, 4]])).all()
    A2 *= B
    assert (A2.value() == np.array([[7, 10], [15, 22]])).all()
    A2.set_value(np.array([[1, 2], [3, 4]]))

    # VariableBlock matrix-matrix operators
    A2[:, :] += B
    assert (A2.value() == np.array([[2, 4], [6, 8]])).all()
    A2[:, :] -= B
    assert (A2.value() == np.array([[1, 2], [3, 4]])).all()
    A2[:, :] *= B
    assert (A2.value() == np.array([[7, 10], [15, 22]])).all()
    A2.set_value(np.array([[1, 2], [3, 4]]))

    # VariableMatrix matrix-scalar operators
    A1 += b
    assert A1.value() == 3
    A1 -= b
    assert A1.value() == 1
    A2 *= b
    assert (A2.value() == np.array([[2, 4], [6, 8]])).all()
    A2 /= b
    assert (A2.value() == np.array([[1, 2], [3, 4]])).all()

    # VariableBlock matrix-scalar operators
    A2[:1, :1] += b
    assert (A2.value() == np.array([[3, 2], [3, 4]])).all()
    A2[:1, :1] -= b
    assert (A2.value() == np.array([[1, 2], [3, 4]])).all()
    A2[:, :] *= b
    assert (A2.value() == np.array([[2, 4], [6, 8]])).all()
    A2[:, :] /= b
    assert (A2.value() == np.array([[1, 2], [3, 4]])).all()


def test_iterators() -> None:
    A = VariableMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    sub_A = A[2:3, 1:3]

    assert sum(1 for _ in A) == 9
    assert sum(1 for _ in sub_A) == 2

    i = 1
    for elem in A:
        assert elem.value() == i
        i += 1

    i = 8
    for elem in sub_A:
        assert elem.value() == i
        i += 1


def test_value() -> None:
    A = VariableMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Full matrix
    assert (A.value() == expected).all()
    assert A.value(3) == 4.0
    assert A.T.value(3) == 2.0

    # Slice
    assert (A[1:3, 1:3].value() == expected[1:3, 1:3]).all()
    assert A[1:3, 1:3].value(2) == 8.0
    assert A[1:3, 1:3].T.value(2) == 6.0

    # Slice-of-slice
    assert (A[1:3, 1:3][:, 1:].value() == expected[1:3, 1:3][:, 1:]).all()
    assert A[1:3, 1:3][:, 1:].value(1) == 9.0
    assert A[1:3, 1:3].T[:, 1:].value(1) == 9.0


def test_pow() -> None:
    # VariableMatrix coefficient-wise power
    A = VariableMatrix([[-2.0, -3.0, -4.0], [-5.0, -6.0, -7.0]])
    assert A**2 == np.array([[4.0, 9.0, 16.0], [25.0, 36.0, 49.0]])
    assert A**3 == np.array([[-8.0, -27.0, -64.0], [-125.0, -216.0, -343.0]])

    # VariableBlock coefficient-wise power
    sub_A = A[:2, :2]
    assert sub_A**2 == np.array([[4.0, 9.0], [25.0, 36.0]])
    assert sub_A**3 == np.array([[-8.0, -27.0], [-125.0, -216.0]])


def test_cwise_map() -> None:
    # VariableMatrix cwise_map
    A = VariableMatrix([[-2.0, -3.0, -4.0], [-5.0, -6.0, -7.0]])

    result1 = A.cwise_map(autodiff.abs)
    expected1 = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

    # Don't modify original matrix
    assert (A.value() == -expected1).all()

    assert (result1.value() == expected1).all()

    # VariableBlock cwise_map
    sub_A = A[:2, :2]

    result2 = sub_A.cwise_map(autodiff.abs)
    expected2 = np.array([[2.0, 3.0], [5.0, 6.0]])

    # Don't modify original matrix
    assert (A.value() == -expected1).all()
    assert (sub_A.value() == -expected2).all()

    assert (result2.value() == expected2).all()


def test_unary_broadcast() -> None:
    # VariableMatrix broadcast
    A = VariableMatrix([[-2.0, -3.0, -4.0], [-5.0, -6.0, -7.0]])

    result1 = autodiff.abs(A)
    expected1 = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

    # Don't modify original matrix
    assert (A.value() == -expected1).all()

    assert (result1.value() == expected1).all()

    # VariableBlock broadcast
    sub_A = A[:2, :2]

    result2 = autodiff.abs(sub_A)
    expected2 = np.array([[2.0, 3.0], [5.0, 6.0]])

    # Don't modify original matrix
    assert (A.value() == -expected1).all()
    assert (sub_A.value() == -expected2).all()

    assert (result2.value() == expected2).all()


def test_identity_static_function() -> None:
    A = VariableMatrix.identity(3)

    assert A.shape == (3, 3)

    assert (A.value() == np.identity(3)).all()


def test_zero_static_function() -> None:
    A = VariableMatrix.zero(2, 3)

    assert A.shape == (2, 3)

    for row in range(A.rows()):
        for col in range(A.cols()):
            assert A[row, col].value() == 0.0


def test_one_static_function() -> None:
    A = VariableMatrix.one(2, 3)

    assert A.shape == (2, 3)

    for row in range(A.rows()):
        for col in range(A.cols()):
            assert A[row, col].value() == 1.0


def test_constant_static_function() -> None:
    A = VariableMatrix.constant(2, 3, 2.0)

    assert A.shape == (2, 3)

    for row in range(A.rows()):
        for col in range(A.cols()):
            assert A[row, col].value() == 2.0


def test_cwise_reduce() -> None:
    A = VariableMatrix([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    B = VariableMatrix([[8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])
    result = autodiff.cwise_reduce(A, B, lambda a, b: a * b)

    assert (result.value() == np.array([[16.0, 27.0, 40.0], [55.0, 72.0, 91.0]])).all()


def test_trace() -> None:
    A = VariableMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    assert A.trace().value() == 15.0

    # Block
    assert A[1:3, 1:3].trace().value() == 14.0
    assert A[0:2, 1:3].trace().value() == 8.0


def test_exp() -> None:
    # Scalar
    A1 = VariableMatrix([[4.0]])
    np.testing.assert_allclose(
        A1.exp().value(), np.array([[math.exp(4.0)]]), rtol=16 * ε, atol=0
    )
    np.testing.assert_allclose(
        A1[:, :].exp().value(), np.array([[math.exp(4.0)]]), rtol=16 * ε, atol=0
    )

    # exp(A) exp(−A) = I cancels terms as large as ‖exp(A)‖‖exp(−A)‖, so the
    # rounding error scales with that product instead of with I
    def exp_product_margin(A: VariableMatrix) -> float:
        return float(
            8 * ε * np.linalg.norm(A.exp().value()) * np.linalg.norm((-A).exp().value())
        )

    A2 = VariableMatrix([[0.0, 1.0], [0.0, -0.5]])
    np.testing.assert_allclose(
        A2.exp().value() @ (-A2).exp().value(),
        np.identity(2),
        atol=exp_product_margin(A2),
    )
    np.testing.assert_allclose(
        A2[:, :].exp().value() @ (-A2[:, :]).exp().value(),
        np.identity(2),
        atol=exp_product_margin(A2),
    )

    A3 = VariableMatrix([[0.0, 1.0], [0.0, 10.0]])
    np.testing.assert_allclose(
        A3.exp().value() @ (-A3).exp().value(),
        np.identity(2),
        atol=exp_product_margin(A3),
    )
    np.testing.assert_allclose(
        A3[:, :].exp().value() @ (-A3[:, :]).exp().value(),
        np.identity(2),
        atol=exp_product_margin(A3),
    )
    A4 = VariableMatrix([[1.0, 10.0], [0.0, 0.0]])
    np.testing.assert_allclose(
        A4.exp().value() @ (-A4).exp().value(),
        np.identity(2),
        atol=exp_product_margin(A4),
    )
    np.testing.assert_allclose(
        A4[:, :].exp().value() @ (-A4[:, :]).exp().value(),
        np.identity(2),
        atol=exp_product_margin(A4),
    )

    A5 = VariableMatrix([[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(
        A5.exp().value() @ (-A5).exp().value(),
        np.identity(2),
        atol=exp_product_margin(A5),
    )
    np.testing.assert_allclose(
        A5[:, :].exp().value() @ (-A5[:, :]).exp().value(),
        np.identity(2),
        atol=exp_product_margin(A5),
    )

    # Pascal matrix
    #
    #    ([0  0  0  0  0  0  0])   [1  0   0   0   0  0  0]
    #    ([1  0  0  0  0  0  0])   [1  1   0   0   0  0  0]
    #    ([0  2  0  0  0  0  0])   [1  2   1   0   0  0  0]
    # exp([0  0  3  0  0  0  0]) = [1  3   3   1   0  0  0]
    #    ([0  0  0  4  0  0  0])   [1  4   6   4   1  0  0]
    #    ([0  0  0  0  5  0  0])   [1  5  10  10   5  1  0]
    #    ([0  0  0  0  0  6  0])   [1  6  15  20  15  6  1]
    pascal = VariableMatrix.zero(7, 7)
    for col in range(6):
        pascal[col + 1, col] = col + 1
    expected_pascal = np.zeros((7, 7))
    expected_pascal[:, 0] = 1.0
    for col in range(1, 7):
        for row in range(col, 7):
            expected_pascal[row, col] = (
                expected_pascal[row - 1, col - 1] + expected_pascal[row - 1, col]
            )
    np.testing.assert_allclose(
        pascal.exp().value(), expected_pascal, rtol=32 * ε, atol=0
    )
    np.testing.assert_allclose(
        pascal[:, :].exp().value(), expected_pascal, rtol=32 * ε, atol=0
    )


def test_block_free_function() -> None:
    A = VariableMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = VariableMatrix([[7.0], [8.0]])

    mat1 = autodiff.block([[A, B]])
    expected1 = np.array([[1.0, 2.0, 3.0, 7.0], [4.0, 5.0, 6.0, 8.0]])
    assert mat1.shape == (2, 4)
    assert (mat1.value() == expected1).all()

    C = VariableMatrix([[9.0, 10.0, 11.0, 12.0]])

    mat2 = autodiff.block([[A, B], [C]])
    expected2 = np.array(
        [[1.0, 2.0, 3.0, 7.0], [4.0, 5.0, 6.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    )
    assert mat2.shape == (3, 4)
    assert (mat2.value() == expected2).all()


def check_solve(A: VariableMatrix, B: VariableMatrix) -> None:
    print(f"solve(A, B) {A.shape[0]}x{A.shape[1]}")

    for X in (
        autodiff.solve(A, B),
        autodiff.solve(A, B.value()),
        autodiff.solve(A.value(), B),
    ):
        assert X.shape == (A.shape[1], B.shape[1])

        # Backward error bound of a stable solve
        residual = np.linalg.norm(A.value() @ X.value() - B.value())
        assert residual <= 8 * ε * np.linalg.norm(A.value()) * np.linalg.norm(X.value())


def test_solve_free_function() -> None:
    # 1x1 special case
    check_solve(VariableMatrix([[2.0]]), VariableMatrix([[5.0]]))

    # 2x2 special case
    check_solve(
        VariableMatrix([[1.0, 2.0], [3.0, 4.0]]), VariableMatrix([[5.0], [6.0]])
    )

    # 3x3 special case
    check_solve(
        VariableMatrix([[1.0, 2.0, 3.0], [-4.0, -5.0, 6.0], [7.0, 8.0, 9.0]]),
        VariableMatrix([[10.0], [11.0], [12.0]]),
    )

    # 4x4 special case
    check_solve(
        VariableMatrix(
            [
                [1.0, 2.0, 3.0, -4.0],
                [-5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        ),
        VariableMatrix([[17.0], [18.0], [19.0], [20.0]]),
    )

    # 5x5 general case
    check_solve(
        VariableMatrix(
            [
                [1.0, 2.0, 3.0, -4.0, 5.0],
                [-5.0, 6.0, 7.0, 8.0, 9.0],
                [9.0, 10.0, 11.0, 12.0, 13.0],
                [13.0, 14.0, 15.0, 16.0, 17.0],
                [17.0, 18.0, 19.0, 20.0, 21.0],
            ]
        ),
        VariableMatrix([[21.0], [22.0], [23.0], [24.0], [25.0]]),
    )
