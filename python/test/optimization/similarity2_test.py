import math

import numpy as np
import pytest
from sleipnir.autodiff import ExpressionType, Rotation2, Similarity2, Translation2
from sleipnir.optimization import ExitStatus, Problem


def test_similarity2_default_and_set_value() -> None:
    problem = Problem()
    similarity = problem.similarity2_decision_variable()

    assert isinstance(similarity, Similarity2)
    assert isinstance(similarity.rotation(), Rotation2)
    assert (similarity.translation().value() == np.zeros(2)).all()
    assert similarity.rotation().value() == 0.0
    assert similarity.log_scale().value() == 0.0
    assert similarity.scale().value() == 1.0

    similarity.translation().set_value(np.array([1.0, -2.0]))
    similarity.rotation().set_value(0.5)
    similarity.log_scale().set_value(math.log(3.0))

    assert similarity.translation().x().value() == 1.0
    assert similarity.translation().y().value() == -2.0
    assert similarity.rotation().value() == pytest.approx(0.5)
    assert similarity.scale().value() == pytest.approx(3.0)


def test_similarity2_parameter() -> None:
    problem = Problem()
    similarity = problem.similarity2_parameter()

    assert isinstance(similarity, Similarity2)
    assert (similarity.translation().value() == np.zeros(2)).all()
    assert similarity.rotation().value() == 0.0
    assert similarity.log_scale().value() == 0.0
    assert similarity.translation().x().type() == ExpressionType.PARAMETER
    assert similarity.rotation().sin().type() == ExpressionType.PARAMETER
    assert similarity.log_scale().type() == ExpressionType.PARAMETER


def test_similarity2_operations() -> None:
    problem = Problem()
    similarity1 = problem.similarity2_decision_variable()
    similarity1.translation().set_value(np.array([1.0, 2.0]))
    similarity1.rotation().set_value(0.7)
    similarity1.log_scale().set_value(math.log(1.5))
    similarity2 = Similarity2(np.array([-0.5, 0.25]), -2.1, 0.4)

    def to_numpy(x: float, y: float, θ: float, s: float) -> np.ndarray:
        return np.array(
            [
                [s * math.cos(θ), -s * math.sin(θ), x],
                [s * math.sin(θ), s * math.cos(θ), y],
                [0.0, 0.0, 1.0],
            ]
        )

    T1 = to_numpy(1.0, 2.0, 0.7, 1.5)
    T2 = to_numpy(-0.5, 0.25, -2.1, 0.4)

    assert similarity1.to_matrix().value() == pytest.approx(T1)
    assert (similarity1 * similarity2).to_matrix().value() == pytest.approx(T1 @ T2)
    assert similarity1.inverse().to_matrix().value() == pytest.approx(np.linalg.inv(T1))
    assert (similarity1 * similarity1.inverse()).to_matrix().value() == pytest.approx(
        np.identity(3)
    )
    assert similarity1.relative_to(similarity2).to_matrix().value() == pytest.approx(
        np.linalg.inv(T2) @ T1
    )

    # Similarity transforms can also be made from a Translation2, a Rotation2,
    # and a log scale
    similarity2_from_parts = Similarity2(
        Translation2(-0.5, 0.25), Rotation2(-2.1), math.log(0.4)
    )
    assert similarity2_from_parts.to_matrix().value() == pytest.approx(T2)

    p = np.array([0.3, -1.2])
    expected = (T1 @ np.append(p, 1.0))[:2]
    assert Translation2(p).transform_by(similarity1).value() == pytest.approx(expected)


def test_similarity2_equality_constraint() -> None:
    problem = Problem()
    similarity = problem.similarity2_decision_variable()
    target = Similarity2(np.array([2.0, -1.0]), 3.0, 0.5)

    problem.subject_to(similarity == target)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert similarity.translation().value() == pytest.approx(np.array([2.0, -1.0]))
    assert similarity.rotation().value() == pytest.approx(3.0)
    assert similarity.scale().value() == pytest.approx(0.5)
