import math
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sleipnir.autodiff import (
    Hessian,
    Jacobian,
    Pose3,
    Rotation3,
    Translation3,
    Twist3,
    VariableMatrix,
    block,
)
from sleipnir.optimization import EqualityConstraints, Problem
from tolerance import rounding_tolerance


def test_rotation3_from_rotation_vector() -> None:
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)

    # Angles on both sides of the series switch at θ = 2
    for angle in [0.0, 1e-9, 0.5, 1.999, 2.001, 3.0, 4.0]:
        q = Rotation3(angle * axis)
        expected = Rotation.from_rotvec(angle * axis)

        assert q.value() == pytest.approx(
            expected.as_quat(scalar_first=True), abs=rounding_tolerance(10)
        )

        # to_vector() is the inverse for angles in [0, π]
        if angle <= math.pi:
            assert q.to_vector().value() == pytest.approx(
                (angle * axis).reshape(3, 1), abs=rounding_tolerance(10)
            )


def test_twist3_exp_and_log() -> None:
    # Moving along x and z while turning about z at rate ω traces a helix
    for ω in [0.0, 1e-9, 0.5, 1.5, 3.0]:
        twist = Twist3(np.array([2.0, 0.0, 1.0]), np.array([0.0, 0.0, ω]))
        pose = twist.exp()

        if ω == 0.0:
            expected_translation = np.array([2.0, 0.0, 1.0])
        else:
            expected_translation = np.array(
                [2 * math.sin(ω) / ω, 4 * math.sin(ω / 2) ** 2 / ω, 1.0]
            )
        assert pose.translation().value() == pytest.approx(
            expected_translation, abs=rounding_tolerance(10)
        )

    ξ = Twist3(np.array([1.0, -2.0, 0.5]), np.array([0.3, -1.2, 2.0]))
    half = (ξ * 0.5).exp()
    assert (half * half).to_matrix().value() == pytest.approx(
        ξ.exp().to_matrix().value()
    )

    log = ξ.exp().log()
    assert log.linear().value() == pytest.approx(np.array([[1.0], [-2.0], [0.5]]))
    assert log.angular().value() == pytest.approx(np.array([[0.3], [-1.2], [2.0]]))

    # exp(log(T)) = T
    pose_axis = np.array([1.0, 1.0, 1.0])
    pose_rotation = Rotation.from_rotvec(2.9 * pose_axis / np.linalg.norm(pose_axis))
    pose = Pose3(
        np.array([3.0, 1.0, -2.0]),
        Rotation3(*pose_rotation.as_quat(scalar_first=True)),
    )
    assert pose.log().exp().to_matrix().value() == pytest.approx(
        pose.to_matrix().value()
    )

    assert isinstance(ξ == 2.0 * ξ, EqualityConstraints)


def test_twist3_derivatives_at_zero_and_across_the_series_switch() -> None:
    ε = sys.float_info.epsilon

    problem = Problem()
    linear = problem.decision_variable(3)
    angular = problem.decision_variable(3)
    wrt = block([[linear], [angular]])

    ξ = Twist3(linear, angular)
    pose = ξ.exp()
    twist = pose.log()

    # A function that depends on every series
    out = block(
        [[VariableMatrix(pose.translation())], [twist.linear()], [twist.angular()]]
    )
    f = (
        pose.translation().squared_distance(Translation3(np.zeros(3)))
        + (twist.linear().T @ twist.linear())[0]
    )

    J = Jacobian(out, wrt)
    H = Hessian(f, wrt)

    # At zero, exp() is the identity to first order, so the translation's
    # Jacobian is [I 0], and log(exp(ξ)) = ξ has Jacobian I
    linear.set_value(np.zeros((3, 1)))
    angular.set_value(np.zeros((3, 1)))
    J_expected = np.zeros((9, 6))
    J_expected[:3, :3] = np.identity(3)
    J_expected[3:, :] = np.identity(6)
    assert J.value().todense() == pytest.approx(J_expected, abs=rounding_tolerance(1))
    assert np.isfinite(H.value().todense()).all()

    # The series and closed forms agree on either side of the switch at θ = 1
    # and θ = 2
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    linear.set_value(np.array([[0.3], [-1.0], [2.0]]))
    for θ in [1.0, 2.0]:
        angular.set_value((θ * (1 - 4 * ε) * axis).reshape(3, 1))
        J_series = J.value().todense()
        H_series = H.value().todense()

        angular.set_value((θ * (1 + 4 * ε) * axis).reshape(3, 1))
        J_closed = J.value().todense()
        H_closed = H.value().todense()

        assert J_series == pytest.approx(J_closed, abs=rounding_tolerance(100))
        assert H_series == pytest.approx(H_closed, abs=rounding_tolerance(500))
