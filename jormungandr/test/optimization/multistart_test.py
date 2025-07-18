import pytest

import jormungandr.autodiff as autodiff
from jormungandr.optimization import ExitStatus, Problem, multistart


class DecisionVariables:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def mishras_bird_function_solve(input: DecisionVariables):
    problem = Problem()

    x = problem.decision_variable()
    x.set_value(input.x)
    y = problem.decision_variable()
    y.set_value(input.y)

    # https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    J = (
        autodiff.sin(y) * autodiff.exp((1 - autodiff.cos(x)) ** 2)
        + autodiff.cos(x) * autodiff.exp((1 - autodiff.sin(y)) ** 2)
        + (x - y) ** 2
    )
    problem.minimize(J)

    problem.subject_to((x + 5) ** 2 + (y + 5) ** 2 < 25)

    return problem.solve(), J.value(), DecisionVariables(x.value(), y.value())


def test_mishras_bird_function():
    status, cost, variables = multistart(
        mishras_bird_function_solve,
        [DecisionVariables(-3, -8), DecisionVariables(-3, -1.5)],
    )

    assert status == ExitStatus.SUCCESS

    assert variables.x == pytest.approx(-3.130246803458174, abs=1e-11)
    assert variables.y == pytest.approx(-1.5821421769364057, abs=1e-11)
