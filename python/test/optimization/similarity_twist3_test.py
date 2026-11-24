import numpy as np
import pytest
import scipy.linalg
from scipy.spatial.transform import Rotation
from sleipnir.autodiff import Rotation3, Similarity3, SimilarityTwist3
from sleipnir.optimization import ExitStatus, Problem
from tolerance import rounding_tolerance, solution_tolerance

AXIS = np.array([1.0, -2.0, 0.5]) / np.linalg.norm([1.0, -2.0, 0.5])


@pytest.mark.parametrize(
    "σ, θ", [(0.0, 0.0), (1e-9, 1e-9), (0.3, 0.0), (0.0, 2.0), (0.6, 0.8), (2.0, 3.0)]
)
def test_similarity_twist3_exp_and_log(σ: float, θ: float) -> None:
    v = np.array([1.0, -2.0, 0.5])
    ω = θ * AXIS
    twist = SimilarityTwist3(v, ω, σ)

    # exp() is the matrix exponential of the generator. scipy.linalg.expm()'s
    # error is up to a few hundred ε times the result's largest entry.
    G = np.zeros((4, 4))
    G[:3, :3] = σ * np.identity(3) + np.array(
        [[0.0, -ω[2], ω[1]], [ω[2], 0.0, -ω[0]], [-ω[1], ω[0], 0.0]]
    )
    G[:3, 3] = v
    expected = scipy.linalg.expm(G)
    assert twist.exp().to_matrix().value() == pytest.approx(
        expected, abs=rounding_tolerance(1000) * np.abs(expected).max()
    )

    log = twist.exp().log()
    assert log.linear().value().ravel() == pytest.approx(v, abs=rounding_tolerance(100))
    assert log.angular().value().ravel() == pytest.approx(
        ω, abs=rounding_tolerance(100)
    )
    assert log.log_scale().value() == pytest.approx(σ, abs=rounding_tolerance(10))


def test_similarity_twist3_group_properties() -> None:
    ξ = SimilarityTwist3(np.array([1.0, -2.0, 0.5]), np.array([0.3, -1.2, 2.0]), 0.7)
    half = (ξ * 0.5).exp()
    assert (half * half).to_matrix().value() == pytest.approx(
        ξ.exp().to_matrix().value()
    )

    q = Rotation.from_rotvec(2.9 * np.ones(3) / np.sqrt(3.0))
    similarity = Similarity3(
        np.array([3.0, 1.0, -2.0]), Rotation3(*q.as_quat(scalar_first=True)), 0.2
    )
    assert similarity.log().exp().to_matrix().value() == pytest.approx(
        similarity.to_matrix().value()
    )


def test_similarity_twist3_between_two_similarity_transforms() -> None:
    v_true = np.array([1.0, -2.0, 0.5])
    ω_true = np.array([0.3, -1.2, 2.0])
    σ_true = -0.4

    q = Rotation.from_rotvec(np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0))
    start = Similarity3(
        np.array([-1.0, 0.5, 2.0]),
        Rotation3(*q.as_quat(scalar_first=True)),
        np.log(1.5),
    )
    end = start * SimilarityTwist3(v_true, ω_true, σ_true).exp()

    problem = Problem()
    twist = SimilarityTwist3(
        problem.real_decision_variable(3),
        problem.real_decision_variable(3),
        problem.real_decision_variable(),
    )
    problem.subject_to(start * twist.exp() == end)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert twist.linear().value().ravel() == pytest.approx(
        v_true, abs=solution_tolerance(1)
    )
    assert twist.angular().value().ravel() == pytest.approx(
        ω_true, abs=solution_tolerance(1)
    )
    assert twist.log_scale().value() == pytest.approx(σ_true, abs=solution_tolerance(1))
