import math
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sleipnir.autodiff import (
    Hessian,
    Jacobian,
    Rotation3,
    UnitVector,
    Variable,
    VariableMatrix,
)
from sleipnir.optimization import Problem, so3, sphere
from tolerance import rounding_tolerance


def test_rotation3_default_and_set_value() -> None:
    problem = Problem()
    q = problem.decision_variable(so3)

    assert isinstance(q, Rotation3)
    assert (q.value() == np.array([1.0, 0.0, 0.0, 0.0])).all()

    # set_value() normalizes
    q.set_value(2.0, 0.0, 0.0, 0.0)
    assert q.value()[0] == 1.0

    q.set_value(1.0, 1.0, 1.0, 1.0)
    assert q.value() == pytest.approx(np.full(4, 0.5))


def test_rotation3_operations() -> None:
    problem = Problem()
    q1 = problem.decision_variable(so3)
    q2 = problem.decision_variable(so3)
    v = problem.decision_variable(sphere, 3)

    q1_axis = np.array([1.0, 2.0, -1.0])
    q2_axis = np.array([0.5, -1.0, 3.0])
    q1_value = Rotation.from_rotvec(0.7 * q1_axis / np.linalg.norm(q1_axis))
    q2_value = Rotation.from_rotvec(-2.1 * q2_axis / np.linalg.norm(q2_axis))
    v_value = np.array([0.2, -0.9, 0.4])
    v_value /= np.linalg.norm(v_value)
    p_value = np.array([1.5, -2.0, 0.5])

    q1.set_value(*q1_value.as_quat(scalar_first=True))
    q2.set_value(*q2_value.as_quat(scalar_first=True))
    v.set_value(v_value)

    # Composition
    q12_expected = q1_value * q2_value
    assert (q1 * q2).value() == pytest.approx(q12_expected.as_quat(scalar_first=True))

    # Inverse
    q1_inv = q1.inverse()
    assert q1_inv.value() == pytest.approx(q1_value.inv().as_quat(scalar_first=True))
    assert (q1 * q1_inv).value() == pytest.approx(np.array([1.0, 0.0, 0.0, 0.0]))

    # Rotation matrix
    assert q1.to_matrix().value() == pytest.approx(q1_value.as_matrix())

    # Rotating a UnitVector returns a UnitVector
    rotated_v = v.rotate_by(q1)
    assert isinstance(rotated_v, UnitVector)
    assert rotated_v.value() == pytest.approx(q1_value.apply(v_value))

    # Rotating a general vector returns a VariableMatrix
    rotated_p = VariableMatrix(p_value.reshape(3, 1)).rotate_by(q1)
    assert isinstance(rotated_p, VariableMatrix)
    assert rotated_p.value() == pytest.approx(q1_value.apply(p_value).reshape(3, 1))


# Small angles use the series, and large angles use atan2
@pytest.mark.parametrize("angle", [0.0, 1e-8, 1e-3, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0])
def test_rotation3_to_vector_value(angle: float) -> None:
    problem = Problem()
    q = problem.decision_variable(so3)
    r = q.to_vector()

    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)

    # q and −q give the same rotation vector
    for sign in [1.0, -1.0]:
        q_value = Rotation.from_rotvec(angle * axis).as_quat(scalar_first=True)
        q.set_value(*(sign * q_value))

        expected = (angle * axis).reshape(3, 1)
        assert r.value() == pytest.approx(expected, abs=1e-5)


def test_rotation3_to_vector_derivatives() -> None:
    ε = sys.float_info.epsilon

    problem = Problem()
    q = problem.decision_variable(so3)
    r = q.to_vector()
    wrt = VariableMatrix([[q.w()], [q.x()], [q.y()], [q.z()]])

    # ‖r‖² = θ_sq, which is smooth near the identity
    θ_sq = (r.T @ r)[0]

    J = Jacobian(r, wrt)
    H = Hessian(θ_sq, wrt)

    # At the identity, r ≈ 2u/w, so ∂r/∂u = 2I and ∂r/∂w = 0, and
    # θ_sq ≈ 4‖u‖²/w², so ∇²θ_sq = diag(0, 8, 8, 8)
    q.set_value(1.0, 0.0, 0.0, 0.0)

    J_expected = np.array(
        [[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0]]
    )
    assert J.value().todense() == pytest.approx(J_expected, abs=rounding_tolerance(1))

    H_expected = np.diag([0.0, 8.0, 8.0, 8.0])
    assert H.value().todense() == pytest.approx(H_expected, abs=rounding_tolerance(1))

    # The series and atan2 branches agree on either side of the switch
    # t = ‖u‖²/w² = ε^(2/17)
    t_max = ε ** (2 / 17)
    half_angle_max = math.atan(math.sqrt(t_max))
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)

    q.set_value(
        *Rotation.from_rotvec(2 * half_angle_max * (1 - 4 * ε) * axis).as_quat(
            scalar_first=True
        )
    )
    J_series = J.value().todense()
    H_series = H.value().todense()

    q.set_value(
        *Rotation.from_rotvec(2 * half_angle_max * (1 + 4 * ε) * axis).as_quat(
            scalar_first=True
        )
    )
    J_atan2 = J.value().todense()
    H_atan2 = H.value().todense()

    assert J_series == pytest.approx(J_atan2, abs=rounding_tolerance(100))
    assert H_series == pytest.approx(H_atan2, abs=rounding_tolerance(1e4))
    assert np.isfinite(J_series).all()
    assert np.isfinite(H_series).all()


def test_unit_vector_values_and_operators() -> None:
    problem = Problem()
    n = problem.decision_variable(sphere, 3)
    m = problem.decision_variable(sphere, 3)

    assert isinstance(n, UnitVector)
    assert n.shape == (3, 1)
    assert len(n) == 3
    assert (n.value() == np.array([1.0, 0.0, 0.0])).all()

    # set_value() normalizes
    n.set_value(np.array([0.0, 3.0, 4.0]))
    assert n.value() == pytest.approx(np.array([0.0, 0.6, 0.8]))
    m.set_value(np.array([0.0, 0.0, 1.0]))

    n_value = np.array([[0.0], [0.6], [0.8]])
    m_value = np.array([[0.0], [0.0], [1.0]])

    # Element access and iteration
    assert n.value(1) == pytest.approx(0.6)
    assert n[-1].value() == pytest.approx(0.8)
    assert [elem.value() for elem in n] == pytest.approx([0.0, 0.6, 0.8])
    with pytest.raises(IndexError):
        n[3]

    # Closed operations
    assert isinstance(-n, UnitVector)
    assert (-n).value() == pytest.approx(-n_value.ravel())

    # Operations that drop to VariableMatrix
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    s = Variable(2.0)
    for result, expected in [
        (A @ n, A @ n_value),
        (VariableMatrix(A) @ n, A @ n_value),
        (2 * n, 2 * n_value),
        (n * 2.0, 2 * n_value),
        (s * n, 2 * n_value),
        (n * s, 2 * n_value),
        (n / 2, n_value / 2),
        (n + m, n_value + m_value),
        (n - m, n_value - m_value),
        (n + m_value, n_value + m_value),
        (m_value - n, m_value - n_value),
        (n - VariableMatrix(m_value), n_value - m_value),
        (n.T @ m, n_value.T @ m_value),
        (n.cross(m), np.cross(n_value.ravel(), m_value.ravel()).reshape(3, 1)),
    ]:
        assert isinstance(result, VariableMatrix)
        assert result.value() == pytest.approx(expected)

    assert n.dot(m).value() == pytest.approx(0.8)

    # Implicit conversion
    assert VariableMatrix(n).value() == pytest.approx(n_value)
