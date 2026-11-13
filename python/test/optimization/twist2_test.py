import math

import numpy as np
import pytest
from sleipnir.autodiff import Rotation2, Twist2
from sleipnir.optimization import EqualityConstraints, Problem
from tolerance import rounding_tolerance


def test_rotation2_from_variable_angle() -> None:
    problem = Problem()
    θ = problem.decision_variable()
    θ.set_value(2.5)
    z = Rotation2(θ)

    assert z.value() == pytest.approx(2.5)
    assert z.angle().value() == pytest.approx(2.5)

    # The rotation depends on θ
    θ.set_value(-1.0)
    assert z.value() == pytest.approx(-1.0)
    assert z.cos().value() == pytest.approx(math.cos(-1.0))

    # Constants still use the float constructor
    assert Rotation2(0.5).value() == pytest.approx(0.5)


def test_twist2_exp_and_log() -> None:
    # Moving at speed v while turning at rate ω traces a circular arc of radius
    # v/ω
    for ω in [0.0, 1e-9, 0.5, 1.5, 3.0]:
        twist = Twist2(np.array([2.0, 0.0]), ω)
        pose = twist.exp()

        if ω == 0.0:
            expected_translation = np.array([2.0, 0.0])
        else:
            expected_translation = np.array(
                [2 * math.sin(ω) / ω, 4 * math.sin(ω / 2) ** 2 / ω]
            )
        assert pose.translation().value() == pytest.approx(
            expected_translation, abs=rounding_tolerance(10)
        )
        assert pose.rotation().value() == pytest.approx(ω, abs=rounding_tolerance(10))

    # exp() of a twist is a one-parameter subgroup, so exp(ξ) = exp(ξ/2)², and
    # log() inverts it for angular parts in (−π, π]
    ξ = Twist2(np.array([1.0, -2.0]), 2.5)
    half = (ξ * 0.5).exp()
    assert (half * half).to_matrix().value() == pytest.approx(
        ξ.exp().to_matrix().value()
    )

    log = ξ.exp().log()
    assert log.linear().value() == pytest.approx(np.array([[1.0], [-2.0]]))
    assert log.angular().value() == pytest.approx(2.5)

    # Twist arithmetic
    sum = ξ + ξ - (-ξ)
    assert isinstance(sum, Twist2)
    assert sum.linear().value() == pytest.approx(np.array([[3.0], [-6.0]]))
    assert isinstance(ξ == 2.0 * ξ, EqualityConstraints)
