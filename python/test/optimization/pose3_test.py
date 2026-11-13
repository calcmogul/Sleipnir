import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sleipnir.autodiff import Pose3, Rotation3, Translation3
from sleipnir.optimization import Problem, se3


def test_pose3_default_and_set_value() -> None:
    problem = Problem()
    pose = problem.decision_variable(se3)

    assert isinstance(pose, Pose3)
    assert isinstance(pose.rotation(), Rotation3)
    assert (pose.translation().value() == np.zeros(3)).all()
    assert (pose.rotation().value() == [1.0, 0.0, 0.0, 0.0]).all()

    pose.translation().set_value(np.array([1.0, -2.0, 3.0]))
    pose.rotation().set_value(2.0, 0.0, 0.0, 0.0)

    assert pose.translation().z().value() == 3.0
    assert pose.rotation().value()[0] == 1.0


def test_pose3_operations() -> None:
    q1_axis = np.array([1.0, 2.0, -1.0])
    q2_axis = np.array([0.0, -1.0, 3.0])
    q1 = Rotation.from_rotvec(0.7 * q1_axis / np.linalg.norm(q1_axis))
    q2 = Rotation.from_rotvec(-2.1 * q2_axis / np.linalg.norm(q2_axis))
    t1 = np.array([1.0, 2.0, 3.0])
    t2 = np.array([-0.5, 0.25, 1.5])

    problem = Problem()
    pose1 = problem.decision_variable(se3)
    pose1.translation().set_value(t1)
    pose1.rotation().set_value(*q1.as_quat(scalar_first=True))
    pose2 = Pose3(t2, Rotation3(*q2.as_quat(scalar_first=True)))

    def to_numpy(t: np.ndarray, q: Rotation) -> np.ndarray:
        result = np.identity(4)
        result[:3, :3] = q.as_matrix()
        result[:3, 3] = t
        return result

    T1 = to_numpy(t1, q1)
    T2 = to_numpy(t2, q2)

    assert pose1.to_matrix().value() == pytest.approx(T1)
    assert (pose1 * pose2).to_matrix().value() == pytest.approx(T1 @ T2)
    assert pose1.inverse().to_matrix().value() == pytest.approx(np.linalg.inv(T1))
    assert (pose1 * pose1.inverse()).to_matrix().value() == pytest.approx(
        np.identity(4)
    )

    p = np.array([0.3, -1.2, 0.8])
    expected = (T1 @ np.append(p, 1.0))[:3]
    assert Translation3(p).transform_by(pose1).value() == pytest.approx(expected)
