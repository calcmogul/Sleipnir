import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sleipnir.autodiff import ExpressionType, Rotation3, Similarity3, Translation3
from sleipnir.optimization import ExitStatus, Problem


def test_similarity3_default_and_set_value() -> None:
    problem = Problem()
    similarity = problem.similarity3_decision_variable()

    assert isinstance(similarity, Similarity3)
    assert isinstance(similarity.rotation(), Rotation3)
    assert (similarity.translation().value() == np.zeros(3)).all()
    assert (similarity.rotation().value() == [1.0, 0.0, 0.0, 0.0]).all()
    assert similarity.log_scale().value() == 0.0
    assert similarity.scale().value() == 1.0

    similarity.translation().set_value(np.array([1.0, -2.0, 3.0]))
    similarity.rotation().set_value(2.0, 0.0, 0.0, 0.0)
    similarity.log_scale().set_value(math.log(3.0))

    assert similarity.translation().z().value() == 3.0
    assert similarity.rotation().value()[0] == 1.0
    assert similarity.scale().value() == pytest.approx(3.0)


def test_similarity3_parameter() -> None:
    problem = Problem()
    similarity = problem.similarity3_parameter()

    assert isinstance(similarity, Similarity3)
    assert (similarity.translation().value() == np.zeros(3)).all()
    assert (similarity.rotation().value() == np.array([1.0, 0.0, 0.0, 0.0])).all()
    assert similarity.log_scale().value() == 0.0
    assert similarity.translation().x().type() == ExpressionType.PARAMETER
    assert similarity.rotation().w().type() == ExpressionType.PARAMETER
    assert similarity.log_scale().type() == ExpressionType.PARAMETER


def test_similarity3_operations() -> None:
    q1_axis = np.array([1.0, 2.0, -1.0])
    q2_axis = np.array([0.0, -1.0, 3.0])
    q1 = Rotation.from_rotvec(0.7 * q1_axis / np.linalg.norm(q1_axis))
    q2 = Rotation.from_rotvec(-2.1 * q2_axis / np.linalg.norm(q2_axis))
    t1 = np.array([1.0, 2.0, 3.0])
    t2 = np.array([-0.5, 0.25, 1.5])
    s1 = 1.5
    s2 = 0.4

    problem = Problem()
    similarity1 = problem.similarity3_decision_variable()
    similarity1.translation().set_value(t1)
    similarity1.rotation().set_value(*q1.as_quat(scalar_first=True))
    similarity1.log_scale().set_value(math.log(s1))
    similarity2 = Similarity3(
        t2, Rotation3(*q2.as_quat(scalar_first=True)), math.log(s2)
    )

    def to_numpy(t: np.ndarray, q: Rotation, s: float) -> np.ndarray:
        result = np.identity(4)
        result[:3, :3] = s * q.as_matrix()
        result[:3, 3] = t
        return result

    T1 = to_numpy(t1, q1, s1)
    T2 = to_numpy(t2, q2, s2)

    assert similarity1.to_matrix().value() == pytest.approx(T1)
    assert (similarity1 * similarity2).to_matrix().value() == pytest.approx(T1 @ T2)
    assert similarity1.inverse().to_matrix().value() == pytest.approx(np.linalg.inv(T1))
    assert (similarity1 * similarity1.inverse()).to_matrix().value() == pytest.approx(
        np.identity(4)
    )
    assert similarity1.relative_to(similarity2).to_matrix().value() == pytest.approx(
        np.linalg.inv(T2) @ T1
    )

    p = np.array([0.3, -1.2, 0.8])
    expected = (T1 @ np.append(p, 1.0))[:3]
    assert Translation3(p).transform_by(similarity1).value() == pytest.approx(expected)


def test_similarity3_equality_constraint() -> None:
    q_axis = np.array([0.2, 1.0, -0.4])
    q = Rotation.from_rotvec(2.5 * q_axis / np.linalg.norm(q_axis))
    t = np.array([1.0, -0.5, 2.0])

    problem = Problem()
    similarity = problem.similarity3_decision_variable()
    target = Similarity3(t, Rotation3(*q.as_quat(scalar_first=True)), math.log(2.0))

    problem.subject_to(similarity == target)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert similarity.translation().value() == pytest.approx(t)
    assert abs(
        similarity.rotation().value() @ q.as_quat(scalar_first=True)
    ) == pytest.approx(1.0)
    assert similarity.scale().value() == pytest.approx(2.0)
