import numpy as np
import pytest
from sleipnir.optimization import ExitStatus, Problem
from tolerance import solution_tolerance


def test_dual_equality_constraint() -> None:
    problem = Problem()

    x, y = problem.decision_variable(2)

    problem.minimize(x * x + y * y)

    constraint = x + y == 2
    problem.subject_to(constraint)

    # The dual variable is zero before the first solve
    assert problem.dual(constraint) == pytest.approx([0.0])

    assert problem.solve() == ExitStatus.SUCCESS

    assert x.value() == pytest.approx(1.0, abs=solution_tolerance(1))
    assert y.value() == pytest.approx(1.0, abs=solution_tolerance(1))

    # ∇f = [2x, 2y] = [2, 2] and ∇cₑ = [1, 1], so ∇f − y∇cₑ = 0 gives y = 2
    assert problem.dual(constraint) == pytest.approx([2.0], abs=solution_tolerance(10))


def test_dual_inequality_constraints() -> None:
    problem = Problem()

    x = problem.decision_variable()

    problem.minimize((x - 2) * (x - 2))

    upper_bound = x <= 1
    lower_bound = x >= -5
    problem.subject_to(lower_bound)
    problem.subject_to(upper_bound)

    assert problem.solve() == ExitStatus.SUCCESS

    assert x.value() == pytest.approx(1.0, abs=solution_tolerance(1))

    # The upper bound is active. ∇f = 2(x − 2) = −2 and ∇cᵢ = ∇(1 − x) = −1,
    # so ∇f − z∇cᵢ = 0 gives z = 2.
    assert problem.dual(upper_bound) == pytest.approx([2.0], abs=solution_tolerance(10))

    # The lower bound is inactive
    assert problem.dual(lower_bound) == pytest.approx([0.0], abs=solution_tolerance(1))


def test_dual_matrix_constraints() -> None:
    problem = Problem()

    x = problem.decision_variable(3)

    # Minimize distance to [2, 2, 2]
    problem.minimize(sum((x[i] - 2) * (x[i] - 2) for i in range(3)))

    # Constraints the queried ones are interleaved with
    problem.subject_to(x[0] >= -10)

    total = x[1] + x[2] == 5
    problem.subject_to(total)

    upper_bounds = x <= np.array([[1.0], [4.0], [2.25]])
    problem.subject_to(upper_bounds)

    assert problem.solve() == ExitStatus.SUCCESS

    # x₀ is at its upper bound. x₁ and x₂ split the sum evenly until x₂ hits its
    # upper bound.
    assert x.value(0) == pytest.approx(1.0, abs=solution_tolerance(1))
    assert x.value(1) == pytest.approx(2.75, abs=solution_tolerance(1))
    assert x.value(2) == pytest.approx(2.25, abs=solution_tolerance(1))

    # Stationarity for x₁ gives 2(x₁ − 2) − y = 0, so y = 1.5
    assert problem.dual(total) == pytest.approx([1.5], abs=solution_tolerance(10))

    # Stationarity for x₀ gives 2(x₀ − 2) + z₀ = 0, so z₀ = 2. The upper bound
    # on x₁ is inactive. Stationarity for x₂ gives 2(x₂ − 2) − y + z₂ = 0, so
    # z₂ = 1.
    assert problem.dual(upper_bounds) == pytest.approx(
        [2.0, 0.0, 1.0], abs=solution_tolerance(10)
    )


def test_set_dual_equality_constraint() -> None:
    problem = Problem()

    x, y = problem.decision_variable(2)

    problem.minimize(x * x + y * y)

    constraint = x + y == 2
    problem.subject_to(constraint)

    problem.set_dual(constraint, np.array([2.0]))
    assert problem.dual(constraint) == pytest.approx([2.0])

    iterations = 0

    def callback(info) -> None:
        nonlocal iterations
        iterations += 1

    problem.add_callback(callback)

    # The primal-dual solution satisfies the KKT conditions, so the solver
    # starts there and stops immediately
    x.set_value(1.0)
    y.set_value(1.0)
    assert problem.solve() == ExitStatus.SUCCESS
    assert iterations == 0

    # With zero dual variables, the solver has to take a step
    problem.set_dual(constraint, np.array([0.0]))
    x.set_value(1.0)
    y.set_value(1.0)
    iterations = 0
    assert problem.solve() == ExitStatus.SUCCESS
    assert iterations > 0

    assert problem.dual(constraint) == pytest.approx([2.0], abs=solution_tolerance(10))


def test_set_dual_matrix_constraints() -> None:
    problem = Problem()

    x = problem.decision_variable(3)

    problem.minimize(sum((x[i] - 2) * (x[i] - 2) for i in range(3)))

    problem.subject_to(x[0] >= -10)

    total = x[1] + x[2] == 5
    problem.subject_to(total)

    upper_bounds = x <= np.array([[1.0], [4.0], [2.25]])
    problem.subject_to(upper_bounds)

    y = np.array([-3.0])
    z = np.array([4.0, 5.0, 6.0])
    problem.set_dual(total, y)
    problem.set_dual(upper_bounds, z)
    assert problem.dual(total) == pytest.approx(y)
    assert problem.dual(upper_bounds) == pytest.approx(z)

    iterations = 0

    def callback(info) -> None:
        nonlocal iterations
        iterations += 1

    problem.add_callback(callback)

    def solve() -> int:
        """Returns the iteration count of solving from the solution's primal
        variables."""
        nonlocal iterations
        x.set_value(np.array([[1.0], [2.75], [2.25]]))
        iterations = 0

        assert problem.solve() == ExitStatus.SUCCESS

        assert x.value(0) == pytest.approx(1.0, abs=solution_tolerance(1))
        assert x.value(1) == pytest.approx(2.75, abs=solution_tolerance(1))
        assert x.value(2) == pytest.approx(2.25, abs=solution_tolerance(1))

        return iterations

    solve()

    # Starting from the previous solve's dual variables takes fewer iterations
    # than starting from wrong ones
    warm_iterations = solve()
    problem.set_dual(total, y)
    problem.set_dual(upper_bounds, z)
    wrong_dual_iterations = solve()
    assert warm_iterations < wrong_dual_iterations


def test_set_dual_warm_start_new_problem() -> None:
    def solve(warm: bool) -> int:
        """Returns the iteration count of solving from the solution's primal
        variables, with the solution's dual variables if warm is true."""
        problem = Problem()

        x = problem.decision_variable(3)

        problem.minimize(sum((x[i] - 2) * (x[i] - 2) for i in range(3)))

        problem.subject_to(x[0] >= -10)

        total = x[1] + x[2] == 5
        problem.subject_to(total)

        upper_bounds = x <= np.array([[1.0], [4.0], [2.25]])
        problem.subject_to(upper_bounds)

        x.set_value(np.array([[1.0], [2.75], [2.25]]))
        if warm:
            problem.set_dual(total, np.array([1.5]))
            problem.set_dual(upper_bounds, np.array([2.0, 0.0, 1.0]))

        iterations = 0

        def callback(info) -> None:
            nonlocal iterations
            iterations += 1

        problem.add_callback(callback)

        assert problem.solve() == ExitStatus.SUCCESS

        assert x.value(0) == pytest.approx(1.0, abs=solution_tolerance(1))
        assert x.value(1) == pytest.approx(2.75, abs=solution_tolerance(1))
        assert x.value(2) == pytest.approx(2.25, abs=solution_tolerance(1))

        return iterations

    assert solve(True) < solve(False)
