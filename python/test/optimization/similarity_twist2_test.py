import numpy as np
import pytest
import scipy.linalg
from sleipnir.autodiff import Similarity2, SimilarityTwist2
from sleipnir.optimization import ExitStatus, Problem
from tolerance import rounding_tolerance, solution_tolerance


@pytest.mark.parametrize(
    "σ, θ", [(0.0, 0.0), (1e-9, -1e-9), (0.3, 0.0), (0.0, 2.0), (0.6, -0.8), (2.0, 3.0)]
)
def test_similarity_twist2_exp_and_log(σ: float, θ: float) -> None:
    v = np.array([1.0, -2.0])
    twist = SimilarityTwist2(v, θ, σ)

    # exp() is the matrix exponential of the generator. scipy.linalg.expm()'s
    # error is up to a few hundred ε times the result's largest entry.
    G = np.array([[σ, -θ, v[0]], [θ, σ, v[1]], [0.0, 0.0, 0.0]])
    expected = scipy.linalg.expm(G)
    assert twist.exp().to_matrix().value() == pytest.approx(
        expected, abs=rounding_tolerance(1000) * np.abs(expected).max()
    )

    log = twist.exp().log()
    assert log.linear().value().ravel() == pytest.approx(v, abs=rounding_tolerance(100))
    assert log.angular().value() == pytest.approx(θ, abs=rounding_tolerance(10))
    assert log.log_scale().value() == pytest.approx(σ, abs=rounding_tolerance(10))


def test_similarity_twist2_group_properties() -> None:
    ξ = SimilarityTwist2(np.array([1.0, -2.0]), 2.5, 0.7)
    half = (ξ * 0.5).exp()
    assert (half * half).to_matrix().value() == pytest.approx(
        ξ.exp().to_matrix().value()
    )

    similarity = Similarity2(np.array([3.0, 1.0]), -2.9, 0.2)
    assert similarity.log().exp().to_matrix().value() == pytest.approx(
        similarity.to_matrix().value()
    )

    total = ξ + ξ - (-ξ)
    assert total.log_scale().value() == pytest.approx(2.1)


def test_similarity_twist2_between_two_similarity_transforms() -> None:
    v_true = np.array([1.0, -2.0])
    θ_true = 2.0
    σ_true = -0.4

    start = Similarity2(np.array([-1.0, 0.5]), 1.0, 1.5)
    end = start * SimilarityTwist2(v_true, θ_true, σ_true).exp()

    problem = Problem()
    twist = SimilarityTwist2(
        problem.real_decision_variable(2),
        problem.real_decision_variable(),
        problem.real_decision_variable(),
    )
    problem.subject_to(start * twist.exp() == end)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert twist.linear().value().ravel() == pytest.approx(
        v_true, abs=solution_tolerance(1)
    )
    assert twist.angular().value() == pytest.approx(θ_true, abs=solution_tolerance(1))
    assert twist.log_scale().value() == pytest.approx(σ_true, abs=solution_tolerance(1))
