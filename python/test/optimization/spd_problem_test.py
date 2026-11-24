import numpy as np
import pytest
import scipy.linalg
from sleipnir import autodiff
from sleipnir.autodiff import ExpressionType, Real, RealMatrix, SPDMatrix
from sleipnir.optimization import ExitStatus, IterationInfo, Problem
from tolerance import solution_tolerance


def check_iterates_positive_definite(problem: Problem, offset: int, n: int) -> None:
    """Checks the n×n SPD block at offset stays positive definite."""

    def callback(info: IterationInfo) -> None:
        P = np.zeros((n, n))
        index = offset
        for i in range(n):
            for j in range(i + 1):
                P[i, j] = P[j, i] = info.x[index]
                index += 1
        np.linalg.cholesky(P)

    problem.add_callback(callback)


def det3(P: SPDMatrix) -> Real:
    """Returns the determinant of a 3×3 SPDMatrix."""
    return (
        P[0, 0] * (P[1, 1] * P[2, 2] - P[1, 2] * P[2, 1])
        - P[0, 1] * (P[1, 0] * P[2, 2] - P[1, 2] * P[2, 0])
        + P[0, 2] * (P[1, 0] * P[2, 1] - P[1, 1] * P[2, 0])
    )


def test_spd_log_det_maximum_likelihood() -> None:
    # The Gaussian maximum likelihood precision matrix for sample covariance S
    # minimizes tr(SP) − log det P, and it's P = S⁻¹
    S = np.array([[4.0, 1.0, 0.5], [1.0, 2.0, -0.3], [0.5, -0.3, 0.5]])

    problem = Problem()
    P = problem.spd_decision_variable(3)
    problem.minimize((S @ P).trace() - autodiff.log(det3(P)))
    check_iterates_positive_definite(problem, 0, 3)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert np.abs(P.value() - np.linalg.inv(S)).max() < solution_tolerance(10)


def test_spd_parameter() -> None:
    problem = Problem()
    S = problem.spd_parameter(2)

    assert isinstance(S, SPDMatrix)
    assert (S.value() == np.eye(2)).all()
    assert S[1, 0].type() == ExpressionType.PARAMETER

    S.set_value(np.array([[2.0, 1.0], [1.0, 3.0]]))
    assert (S.value() == np.array([[2.0, 1.0], [1.0, 3.0]])).all()
    assert S[0, 1].value() == 1.0


def test_spd_structured_lqr() -> None:
    # See the C++ test for the problem statement
    A = np.array(
        [
            [1.0, 0.1, 0.00426122638850534],
            [0.0, 1.0, 0.07869386805747332],
            [0.0, 0.0, 0.6065306597126334],
        ]
    )
    B = np.array([[0.00073877361149466], [0.02130613194252668], [0.3934693402873665]])
    Q = np.diag([1.0, 0.1, 0.0])
    R = 0.1

    # F starts at the full LQR gains with the last column dropped, and P starts
    # at the solution of the Lyapunov equation for those gains
    S = scipy.linalg.solve_discrete_are(A, B, Q, np.array([[R]]))
    K_0 = np.linalg.solve(B.T @ S @ B + R, B.T @ S @ A)
    K_0[0, 2] = 0.0
    A_cl_0 = A - B @ K_0
    P_0 = scipy.linalg.solve_discrete_lyapunov(A_cl_0.T, Q + K_0.T @ K_0 * R)

    problem = Problem()
    P = problem.spd_decision_variable(3)
    F = problem.real_decision_variable(1, 2)
    P.set_value(P_0)
    F.set_value(K_0[:, :2])

    K = RealMatrix([[F[0], F[1], Real(0.0)]])
    A_cl = A - B @ K
    residual = Q + K.T @ K * R + A_cl.T @ P @ A_cl - P
    problem.subject_to(residual.symmetric_view() == 0)

    problem.minimize(P.trace())
    check_iterates_positive_definite(problem, 0, 3)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert abs(F.value(0) - 1.9140313971107417) < solution_tolerance(10)
    assert abs(F.value(1) - 2.2849602346968894) < solution_tolerance(10)
    assert abs(np.trace(P.value()) - 15.71201670662753) < solution_tolerance(10)


@pytest.mark.parametrize("P_0", [0.01, 1.0, 100.0])
def test_spd_discrete_algebraic_riccati_equation(P_0: float) -> None:
    # See the C++ test for the problem statement
    A = np.array(
        [
            [1.0, 0.1, 0.00426122638850534],
            [0.0, 1.0, 0.07869386805747332],
            [0.0, 0.0, 0.6065306597126334],
        ]
    )
    B = np.array(
        [
            [0.00073877361149466, 0.0],
            [0.02130613194252668, 0.1],
            [0.3934693402873665, 0.0],
        ]
    )
    Q = np.diag([1.0, 0.1, 0.0])
    R = np.diag([0.1, 0.2])
    P_expected = np.array(
        [
            [9.576584400466913, 3.614691854994743, 0.44953971039753926],
            [3.614691854994743, 2.9744755586284817, 0.4238989679269632],
            [0.44953971039753926, 0.4238989679269632, 0.06596987193320486],
        ]
    )

    problem = Problem()
    P = problem.spd_decision_variable(3)
    P.set_value(P_0 * np.eye(3))

    residual = (
        A.T @ P @ A - P - A.T @ P @ B @ autodiff.solve(R + B.T @ P @ B, B.T @ P @ A) + Q
    )
    problem.subject_to(residual.symmetric_view() == 0)
    check_iterates_positive_definite(problem, 0, 3)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert np.abs(P.value() - P_expected).max() < solution_tolerance(10)


def test_spd_matrix_equality_constrains_lower_triangle() -> None:
    problem = Problem()
    P = problem.spd_decision_variable(3)
    Q = problem.spd_decision_variable(3)
    A = problem.real_decision_variable(3, 3)
    G = A.T @ A
    S = np.eye(3)

    assert isinstance(P, SPDMatrix)
    assert isinstance(P + Q, SPDMatrix)
    assert isinstance(P - Q, RealMatrix)

    # One equation per lower-triangle element, whichever side P is on
    for constraint in [
        P == G,
        G == P,
        P == S,
        S == P,
        P == Q,
        P == G.symmetric_view(),
        G.symmetric_view() == P,
        G.symmetric_view() == 0,
        0 == G.symmetric_view(),
        G == G.symmetric_view(),
    ]:
        problem.subject_to(constraint)
    assert problem.equality_constraint_type() != ExpressionType.NONE


def test_spd_matrix_rejects_ambiguous_comparisons() -> None:
    problem = Problem()
    P = problem.spd_decision_variable(2)
    G = problem.real_decision_variable(2, 2)

    # Inequalities are ambiguous between elementwise bounds and the
    # semidefinite ordering
    for compare in [
        lambda: P <= G,  # type: ignore[operator]
        lambda: G <= P,  # type: ignore[operator]
        lambda: P >= np.eye(2),  # type: ignore[operator]
        lambda: np.eye(2) <= P,  # type: ignore[operator]
        lambda: P >= 0,  # type: ignore[operator]
        lambda: 0 <= P,  # type: ignore[operator]
        lambda: G[0, 0] <= P,  # type: ignore[operator]
    ]:
        with pytest.raises(TypeError):
            compare()

    # An SPD matrix can't equal a scalar
    for compare_scalar in [lambda: P == 0, lambda: 0 == P, lambda: P == G[0, 0]]:
        with pytest.raises(TypeError):
            compare_scalar()

    # Operands that need a conversion to a matrix still work
    problem.subject_to(P == np.eye(2, dtype=int))

    # Element constraints are fine
    problem.subject_to(P[0, 0] <= 3)
    problem.subject_to(P[1, 0] == 0)


def test_spd_matrix_elements() -> None:
    problem = Problem()
    P = problem.spd_decision_variable(2)
    assert P.shape == (2, 2)
    assert (P.value() == np.eye(2)).all()

    P.set_value(np.array([[2.0, 1.0], [1.0, 3.0]]))
    assert P.value(0, 1) == P.value(1, 0) == 1.0
    assert P[-1, -1].value() == 3.0
    assert P.T.value(0, 1) == 1.0
    assert P.trace().value() == 5.0

    # Constant
    C = SPDMatrix(np.array([[2.0, 1.0], [1.0, 3.0]]))
    assert ((np.eye(2) @ C).value() == C.value()).all()

    # The conversion to RealMatrix is explicit
    assert (RealMatrix(C).value() == C.value()).all()
    assert (autodiff.solve(RealMatrix(C), np.eye(2)).value() @ C.value()).round(
        12
    ).tolist() == np.eye(2).tolist()
