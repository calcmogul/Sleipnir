import math

import numpy as np
import pytest
from sleipnir.autodiff import Pose2, Rotation2, Translation2
from sleipnir.optimization import Problem, se2


def test_pose2_default_and_set_value() -> None:
    problem = Problem()
    pose = problem.decision_variable(se2)

    assert isinstance(pose, Pose2)
    assert isinstance(pose.rotation(), Rotation2)
    assert (pose.translation().value() == np.zeros(2)).all()
    assert pose.rotation().value() == 0.0

    pose.translation().set_value(np.array([1.0, -2.0]))
    pose.rotation().set_value(0.5)

    assert pose.translation().x().value() == 1.0
    assert pose.translation().y().value() == -2.0
    assert pose.rotation().value() == pytest.approx(0.5)


def test_pose2_operations() -> None:
    problem = Problem()
    pose1 = problem.decision_variable(se2)
    pose1.translation().set_value(np.array([1.0, 2.0]))
    pose1.rotation().set_value(0.7)
    pose2 = Pose2(np.array([-0.5, 0.25]), -2.1)

    def to_numpy(x: float, y: float, θ: float) -> np.ndarray:
        return np.array(
            [
                [math.cos(θ), -math.sin(θ), x],
                [math.sin(θ), math.cos(θ), y],
                [0.0, 0.0, 1.0],
            ]
        )

    T1 = to_numpy(1.0, 2.0, 0.7)
    T2 = to_numpy(-0.5, 0.25, -2.1)

    assert pose1.to_matrix().value() == pytest.approx(T1)
    assert (pose1 * pose2).to_matrix().value() == pytest.approx(T1 @ T2)
    assert pose1.inverse().to_matrix().value() == pytest.approx(np.linalg.inv(T1))
    assert (pose1 * pose1.inverse()).to_matrix().value() == pytest.approx(
        np.identity(3)
    )

    # Poses can also be made from a Translation2 and a Rotation2
    pose2_from_parts = Pose2(Translation2(-0.5, 0.25), Rotation2(-2.1))
    assert pose2_from_parts.to_matrix().value() == pytest.approx(T2)

    p = np.array([0.3, -1.2])
    expected = (T1 @ np.append(p, 1.0))[:2]
    assert Translation2(p).transform_by(pose1).value() == pytest.approx(expected)
