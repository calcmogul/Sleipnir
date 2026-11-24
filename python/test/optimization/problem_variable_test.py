import numpy as np
import pytest
from sleipnir.autodiff import ExpressionType
from sleipnir.optimization import ExitStatus, IterationInfo, Problem
from tolerance import solution_tolerance


def test_scalar_init_assign() -> None:
    problem = Problem()

    # Scalar zero init
    x = problem.real_decision_variable()
    assert x.value() == 0.0

    # Scalar assignment
    x.set_value(1.0)
    assert x.value() == 1.0
    x.set_value(2.0)
    assert x.value() == 2.0


def test_vector_init_assign() -> None:
    problem = Problem()

    # Vector zero init
    y = problem.real_decision_variable(2)
    assert y.value(0) == 0.0
    assert y.value(1) == 0.0

    # Vector assignment
    y[0].set_value(1.0)
    y[1].set_value(2.0)
    assert y.value(0) == 1.0
    assert y.value(1) == 2.0
    y[0].set_value(3.0)
    y[1].set_value(4.0)
    assert y.value(0) == 3.0
    assert y.value(1) == 4.0


def test_matrix_init_assign() -> None:
    problem = Problem()

    # RealMatrix zero init
    z = problem.real_decision_variable(3, 2)
    assert z.value(0, 0) == 0.0
    assert z.value(0, 1) == 0.0
    assert z.value(1, 0) == 0.0
    assert z.value(1, 1) == 0.0
    assert z.value(2, 0) == 0.0
    assert z.value(2, 1) == 0.0

    # RealMatrix assignment; element comparison
    z.set_value(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert z.value(0, 0) == 1.0
    assert z.value(0, 1) == 2.0
    assert z.value(1, 0) == 3.0
    assert z.value(1, 1) == 4.0
    assert z.value(2, 0) == 5.0
    assert z.value(2, 1) == 6.0

    # RealMatrix assignment; matrix comparison
    expected = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    z.set_value(expected)
    assert (z.value() == expected).all()

    # Block assignment
    z[:2, :1].set_value(np.array([[1.0], [1.0]]))
    assert (z.value() == np.array([[1.0, 8.0], [1.0, 10.0], [11.0, 12.0]])).all()

    # Segment assignment
    z[:3, :1][:2, :].set_value(np.array([[2.0], [2.0]]))
    assert (z.value() == np.array([[2.0, 8.0], [2.0, 10.0], [11.0, 12.0]])).all()


def test_parameter_init_assign() -> None:
    problem = Problem()

    p = problem.parameter()
    assert p.type() == ExpressionType.PARAMETER
    assert p.value() == 0.0
    p.set_value(2.0)
    assert p.value() == 2.0

    P = problem.parameter(2, 3)
    assert P.shape == (2, 3)
    for elem in P:
        assert elem.type() == ExpressionType.PARAMETER


def test_parameter_not_folded() -> None:
    problem = Problem()

    # Parameters start at zero and one here, which constants would fold
    p = problem.parameter()
    q = problem.parameter()
    q.set_value(1.0)
    x = problem.real_decision_variable()

    product = p * x
    quotient = x / q

    assert product.type() == ExpressionType.LINEAR
    assert quotient.type() == ExpressionType.LINEAR

    p.set_value(2.0)
    q.set_value(4.0)
    x.set_value(3.0)

    assert product.value() == 6.0
    assert quotient.value() == 0.75


def test_parameter_resolve_with_new_values() -> None:
    problem = Problem()

    target = problem.parameter()
    upper_bound = problem.parameter()
    x = problem.real_decision_variable()

    problem.minimize((x - target) * (x - target))
    problem.subject_to(x <= upper_bound)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.NONE
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    # Unconstrained minimum is feasible
    target.set_value(1.0)
    upper_bound.set_value(5.0)
    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS
    assert x.value() == pytest.approx(1.0, abs=solution_tolerance(1))

    # Unconstrained minimum is infeasible, so the bound is active
    target.set_value(10.0)
    upper_bound.set_value(3.0)
    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS
    assert x.value() == pytest.approx(3.0, abs=solution_tolerance(1))


def test_parameter_manifold_types() -> None:
    problem = Problem()

    n = problem.unit_vector_parameter(3)
    pose = problem.pose3_parameter()
    similarity = problem.similarity2_parameter()
    P = problem.spd_parameter(2)
    x = problem.real_decision_variable()

    r = x - n[1] - pose.translation().x() - similarity.rotation().sin() - P[1, 0]
    problem.minimize(r * r)

    # The iterate has only x
    def callback(info: IterationInfo) -> None:
        assert info.x.size == 1

    problem.add_callback(callback)

    n.set_value(np.array([0.0, 1.0, 0.0]))
    pose.translation().set_value(np.array([2.0, 0.0, 0.0]))
    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS
    assert x.value() == pytest.approx(3.0, abs=solution_tolerance(1))
