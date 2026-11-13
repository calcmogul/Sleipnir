import math

import numpy as np
import pytest
from sleipnir.autodiff import Gradient, Rotation2, UnitVector, VariableMatrix
from sleipnir.optimization import Problem, so2, sphere
from tolerance import rounding_tolerance


def test_rotation2_default_and_set_value() -> None:
    problem = Problem()
    z = problem.decision_variable(so2)

    assert isinstance(z, Rotation2)
    assert z.value() == 0.0
    assert z.angle().value() == 0.0

    z.set_value(0.5)
    assert z.value() == pytest.approx(0.5)
    assert z.angle().value() == pytest.approx(0.5)
    assert z.cos().value() == pytest.approx(math.cos(0.5))
    assert z.sin().value() == pytest.approx(math.sin(0.5))

    # Angles are reported in (−π, π]
    z.set_value(4.0)
    assert z.value() == pytest.approx(4.0 - 2 * math.pi)


def test_rotation2_operations() -> None:
    problem = Problem()
    z1 = problem.decision_variable(so2)
    v = problem.decision_variable(sphere, 2)
    z1.set_value(0.7)
    v.set_value(np.array([0.6, 0.8]))
    z2 = Rotation2(-2.1)

    # Composition adds angles, and inversion negates them
    assert (z1 * z2).value() == pytest.approx(0.7 - 2.1)
    assert z1.inverse().value() == pytest.approx(-0.7)
    assert (z1 * z1.inverse()).value() == pytest.approx(0.0)

    R = np.array([[math.cos(0.7), -math.sin(0.7)], [math.sin(0.7), math.cos(0.7)]])
    assert z1.to_matrix().value() == pytest.approx(R)

    # Rotating a UnitVector returns a UnitVector
    rotated_v = v.rotate_by(z1)
    assert isinstance(rotated_v, UnitVector)
    assert rotated_v.value() == pytest.approx(R @ np.array([0.6, 0.8]))

    # Rotating a general vector returns a VariableMatrix
    p = np.array([[1.5], [-2.0]])
    rotated_p = VariableMatrix(p).rotate_by(z1)
    assert isinstance(rotated_p, VariableMatrix)
    assert rotated_p.value() == pytest.approx(R @ p)


def test_rotation2_angle_gradient() -> None:
    problem = Problem()
    z = problem.decision_variable(so2)
    wrt = VariableMatrix([[z.cos()], [z.sin()]])
    g = Gradient(z.angle(), wrt)

    # ∂θ/∂(cos θ, sin θ) = (−sin θ, cos θ) on the unit circle
    for θ in [-3.0, -1.0, 0.0, 0.5, 2.0]:
        z.set_value(θ)
        assert g.value().todense() == pytest.approx(
            np.array([[-math.sin(θ)], [math.cos(θ)]]), abs=rounding_tolerance(5)
        )
