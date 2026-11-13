import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sleipnir.autodiff import Pose2, Pose3, Rotation2, Rotation3, Translation2


def test_relative_to_and_distances() -> None:
    # The relative angle wraps only when the rotations are opposite
    z1 = Rotation2(3.0)
    z2 = Rotation2(-3.0)
    assert z1.relative_to(z2).value() == pytest.approx(6.0 - 2 * math.pi)

    # other * this.relative_to(other) = this
    qa_axis = np.array([1.0, 2.0, 0.0])
    qa = Rotation.from_rotvec(0.5 * qa_axis / np.linalg.norm(qa_axis))
    qb_axis = np.array([0.0, -1.0, 3.0])
    qb = Rotation.from_rotvec(-1.2 * qb_axis / np.linalg.norm(qb_axis))
    q1 = Rotation3(*qa.as_quat(scalar_first=True))
    q2 = Rotation3(*qb.as_quat(scalar_first=True))
    assert (q2 * q1.relative_to(q2)).value() == pytest.approx(
        qa.as_quat(scalar_first=True)
    )
    assert q1.relative_to(Rotation3(1.0, 0.0, 0.0, 0.0)).value() == pytest.approx(
        qa.as_quat(scalar_first=True)
    )

    pose2a = Pose2(np.array([1.0, 2.0]), 0.5)
    pose2b = Pose2(np.array([-3.0, 1.0]), 2.5)
    assert (pose2b * pose2a.relative_to(pose2b)).to_matrix().value() == pytest.approx(
        pose2a.to_matrix().value()
    )

    pose3a = Pose3(np.array([1.0, 2.0, 3.0]), q1)
    pose3b = Pose3(np.array([-3.0, 1.0, 0.5]), q2)
    assert (pose3b * pose3a.relative_to(pose3b)).to_matrix().value() == pytest.approx(
        pose3a.to_matrix().value()
    )

    t = Translation2(3.0, 4.0)
    origin = Translation2(0.0, 0.0)
    assert t.distance(origin).value() == pytest.approx(5.0)
    assert t.squared_distance(origin).value() == pytest.approx(25.0)
