import math

import numpy as np
import pytest
import scipy.sparse
from sleipnir.autodiff import (
    Complex,
    ComplexMatrix,
    ComplexSubmatrix,
    Gradient,
    Real,
    RealMatrix,
)
from sleipnir.optimization import ExitStatus, Problem


def test_complex_arithmetic() -> None:
    problem = Problem()
    z = problem.complex_decision_variable()
    assert isinstance(z, Complex)
    assert z.value() == 0j

    z.set_value(1 + 2j)
    w = 3 - 1j

    assert (z * w).value() == pytest.approx((1 + 2j) * w)
    assert (w * z).value() == pytest.approx((1 + 2j) * w)
    assert (z / w).value() == pytest.approx((1 + 2j) / w)
    assert (w / z).value() == pytest.approx(w / (1 + 2j))
    assert (z + w).value() == pytest.approx(4 + 1j)
    assert (w - z).value() == pytest.approx(2 - 3j)
    assert (z * 2.0).value() == pytest.approx(2 + 4j)
    assert (2 * z).value() == pytest.approx(2 + 4j)
    assert (-z).value() == pytest.approx(-1 - 2j)
    assert (z * np.complex128(w)).value() == pytest.approx((1 + 2j) * w)
    assert (np.complex128(w) * z).value() == pytest.approx((1 + 2j) * w)
    assert z.conj().value() == pytest.approx(1 - 2j)
    assert z.real().value() == 1.0
    assert z.imag().value() == 2.0
    assert z.abs2().value() == pytest.approx(5.0)
    assert abs(z).value() == pytest.approx(math.sqrt(5.0))
    assert z.arg().value() == pytest.approx(math.atan2(2.0, 1.0))

    x = problem.real_decision_variable()
    x.set_value(2.0)
    assert (x * w).value() == pytest.approx(6 - 2j)
    assert (w * x).value() == pytest.approx(6 - 2j)
    assert (x + z).value() == pytest.approx(3 + 2j)
    assert (z - x).value() == pytest.approx(-1 + 2j)
    assert (x / w).value() == pytest.approx(2 / w)

    # NumPy complex scalars keep their imaginary part
    assert (x * np.complex128(w)).value() == pytest.approx(6 - 2j)
    assert (np.complex128(w) * x).value() == pytest.approx(6 - 2j)
    assert (x + np.complex128(w)).value() == pytest.approx(5 - 1j)
    M = RealMatrix(np.array([[1.0, 2.0]]))
    assert np.allclose((M * np.complex128(w)).value(), np.array([[w, 2 * w]]))
    assert np.allclose((np.complex128(w) * M).value(), np.array([[w, 2 * w]]))

    # Python floats still give a Real
    assert isinstance(x * 2.0, Real)
    assert isinstance(x * np.float64(2.0), Real)

    assert Complex.polar(x, math.pi / 2).value() == pytest.approx(2j)
    assert Complex(x, 3.0).value() == pytest.approx(2 + 3j)
    assert Complex(1 - 1j).value() == pytest.approx(1 - 1j)


def test_complex_gradient() -> None:
    problem = Problem()
    z = problem.complex_decision_variable()
    z.set_value(1 + 2j)

    # Re((a + bi)(c + di)) = ac − bd
    f = (z * (3 - 1j)).real()
    assert Gradient(f, z.real()).value()[0, 0] == 3.0
    assert Gradient(f, z.imag()).value()[0, 0] == 1.0


def test_complex_equality_constraints() -> None:
    problem = Problem()
    z = problem.complex_decision_variable()
    z.set_value(1 + 2j)

    assert z == 1 + 2j
    assert not bool(z == 1 + 3j)
    assert 1 + 2j == z


def test_complex_matrix_transposes_and_parts() -> None:
    problem = Problem()
    Z = problem.complex_decision_variable(2, 3)
    assert isinstance(Z, ComplexMatrix)
    assert Z.shape == (2, 3)

    value = np.array([[1 + 2j, 3 - 1j, 1j], [-2, 1 + 1j, 4 - 3j]])
    Z.set_value(value)

    assert np.array_equal(Z.value(), value)
    assert np.array_equal(Z.T.value(), value.T)
    assert np.array_equal(Z.conj().value(), value.conj())
    assert np.array_equal(Z.H.value(), value.conj().T)
    assert np.array_equal(Z.real().value(), value.real)
    assert np.array_equal(Z.imag().value(), value.imag)
    assert np.allclose(Z.abs2().value(), np.abs(value) ** 2)

    assert Z[1, 2].value() == value[1, 2]
    assert Z[-1, -1].value() == value[1, 2]
    assert Z[4].value() == value[1, 1]
    assert np.array_equal(Z[:, 1:].value(), value[:, 1:])
    assert np.array_equal(Z[1, ::2].value(), value[1:, ::2])
    assert np.array_equal(Z.row(1).value(), value[1:, :])
    assert np.array_equal(Z.col(2).value(), value[:, 2:])
    assert np.array_equal(Z.block(0, 1, 2, 2).value(), value[:, 1:])

    # Entries are handles to the same expressions
    Z[0, 0].set_value(5j)
    assert Z.value(0, 0) == 5j

    Z[0, 1] = 7.0
    assert Z.value(0, 1) == 7.0


def test_complex_matrix_slices() -> None:
    problem = Problem()
    Z = problem.complex_decision_variable(2, 3)
    value = np.array([[1 + 2j, 3 - 1j, 1j], [-2, 1 + 1j, 4 - 3j]])
    Z.set_value(value)

    # Slices point to the matrix's entries
    block = Z.block(0, 1, 2, 2)
    assert isinstance(block, ComplexSubmatrix)
    block.set_value(np.array([[5, 6], [7, 8 + 1j]]))
    assert Z.value(0, 1) == 5
    assert Z.value(1, 2) == 8 + 1j

    S = Z[:, ::2]
    assert S.shape == (2, 2)
    assert S.value(1, 1) == Z.value(1, 2)
    assert S[:, 1:].value(0, 0) == Z.value(0, 2)

    # Assignment writes through to the matrix
    Z[1, :] = np.array([[1j, 2j, 3j]])
    assert np.array_equal(Z.row(1).value(), np.array([[1j, 2j, 3j]]))

    Z[:, 0] = Z[:, 2]
    assert Z.value(0, 0) == Z.value(0, 2)
    assert Z.value(1, 0) == Z.value(1, 2)

    Z[0:1, 1:2] = 2 - 2j
    assert Z.value(0, 1) == 2 - 2j

    Z.row(0)[0, 2:] = RealMatrix(np.array([[4.0]]))
    assert Z.value(0, 2) == 4

    with pytest.raises(ValueError):
        Z[:, 0] = 1j

    # Slices are operands of complex matrix arithmetic
    expected = Z.value()[0, :] @ Z.value()[1, :]
    assert (Z.row(0) @ Z.row(1).T).value()[0, 0] == pytest.approx(expected)
    assert np.allclose((Z.row(0) + Z.row(1)).value(), Z.value()[:1] + Z.value()[1:])
    assert np.allclose((2j * Z.col(1)).value(), 2j * Z.value()[:, 1:2])
    assert np.allclose((np.eye(2) @ Z[:, 1:]).value(), Z.value()[:, 1:])
    assert np.allclose((Z @ Z.T[:, :1]).value(), Z.value() @ Z.value().T[:, :1])
    assert np.allclose(Z.col(1).H.value(), Z.value()[:, 1:2].conj().T)
    assert np.array_equal(Z.row(1).real().value(), Z.value()[1:].real)

    assert Z.col(0) == Z.col(0)
    assert not bool(Z.col(0) == Z.col(1))
    assert Z.col(1) == Z.value()[:, 1:2]


def test_complex_matrix_arithmetic() -> None:
    problem = Problem()
    V = problem.complex_decision_variable(3)

    v = np.array([[1.0], [0.9 - 0.1j], [1.05 + 0.2j]])
    V.set_value(v)

    Y = np.array(
        [
            [2 - 20j, -1 + 10j, -1 + 10j],
            [-1 + 10j, 1 - 10j, 0],
            [-1 + 10j, 0, 1 - 10j],
        ]
    )
    Y_sparse = scipy.sparse.csc_matrix(Y)

    assert np.allclose((Y @ V).value(), Y @ v)
    assert np.allclose((V.T @ Y).value(), v.T @ Y)
    assert np.allclose((V.T @ Y_sparse).value(), v.T @ Y)

    # Complex power injection S = V ∘ conj(Y V)
    S = v * (Y @ v).conj()
    assert np.allclose(V.cwise_product((Y @ V).conj()).value(), S)

    # Real matrices mix with complex ones
    A = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0], [3.0, 0.0, 1.0]])
    A_var = RealMatrix(A)
    assert np.allclose((A @ V).value(), A @ v)
    assert np.allclose((A_var @ V).value(), A @ v)
    assert np.allclose((A_var * (3 - 1j)).value(), A * (3 - 1j))
    assert np.allclose(((3 - 1j) * A_var).value(), A * (3 - 1j))

    assert np.allclose((V + V).value(), 2 * v)
    assert np.allclose((V - Y @ v).value(), v - Y @ v)
    assert np.allclose((Y @ v - V).value(), Y @ v - v)
    assert np.allclose((V + A_var[:, 0]).value(), v + A[:, :1])
    assert np.allclose((-V).value(), -v)

    w = 3 - 1j
    assert np.allclose((V * w).value(), v * w)
    assert np.allclose((2 * V).value(), 2 * v)
    assert np.allclose((V * np.float64(2.0)).value(), 2 * v)
    assert np.allclose((V * np.complex128(w)).value(), v * w)
    assert np.allclose((w * V).value(), v * w)
    assert np.allclose((V / w).value(), v / w)

    z = problem.complex_decision_variable()
    z.set_value(w)
    assert np.allclose((V * z).value(), v * w)
    assert np.allclose((z * V).value(), v * w)
    assert np.allclose((z * A_var).value(), A * w)

    assert np.allclose((V.H @ Y @ V).value(), v.conj().T @ Y @ v)
    assert (Y @ V @ V.T).trace().value() == pytest.approx(np.trace(Y @ v @ v.T))
    assert np.array_equal(ComplexMatrix.identity(2).value(), np.eye(2))

    Z = ComplexMatrix([[1 + 2j, z], [3.0, problem.real_decision_variable()]])
    assert Z.shape == (2, 2)
    assert Z.value(0, 1) == w


def test_complex_matrix_equality_constraints() -> None:
    problem = Problem()
    V = problem.complex_decision_variable(2)

    v = np.array([[1 - 1j], [2 + 0.5j]])
    V.set_value(v)

    assert V == v
    assert v == V
    assert not bool(V == 0j)


def test_complex_parameters() -> None:
    problem = Problem()
    z = problem.complex_parameter()
    Z = problem.complex_parameter(2)
    assert z.value() == 0j
    assert np.array_equal(Z.value(), np.zeros((2, 1)))

    z.set_value(1 - 2j)
    assert z.value() == 1 - 2j


def test_complex_ac_optimal_power_flow() -> None:
    # Three buses with generators at buses 0 and 1 and a load at bus 2. Every
    # pair of buses is connected by a line with impedance 0.01 + 0.1i p.u., so
    # swapping buses 0 and 1 leaves the problem unchanged.
    buses = 3
    y = 1 / (0.01 + 0.1j)
    Y = np.zeros((buses, buses), dtype=complex)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        Y[i, i] += y
        Y[j, j] += y
        Y[i, j] -= y
        Y[j, i] -= y

    S_load = 1 + 0.3j

    problem = Problem()

    V = problem.complex_decision_variable(buses)
    S_gen = problem.complex_decision_variable(2)
    V.set_value(np.ones((buses, 1)))

    # Complex power injected at each bus
    S = V.cwise_product((Y @ V).conj())

    problem.minimize(S_gen[0].real() ** 2 + S_gen[1].real() ** 2)

    problem.subject_to(V[0].imag() == 0)
    problem.subject_to(S[:2, :] == S_gen)
    problem.subject_to(S[2] == -S_load)
    problem.subject_to(V.abs2() >= 0.95**2)
    problem.subject_to(V.abs2() <= 1.05**2)
    problem.subject_to(S_gen.real() >= 0)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    v = V.value()
    s_gen = S_gen.value()
    s = v * (Y @ v).conj()

    assert s[:2] == pytest.approx(s_gen, abs=1e-6)
    assert s[2, 0] == pytest.approx(-S_load, abs=1e-6)
    assert s_gen[0, 0].real == pytest.approx(s_gen[1, 0].real, abs=1e-6)
