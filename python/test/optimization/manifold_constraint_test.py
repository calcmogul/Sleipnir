import numpy as np
import pytest
from sleipnir.autodiff import (
    Translation2,
)
from sleipnir.optimization import (
    ExitStatus,
    Problem,
)
from tolerance import solution_tolerance


def test_distance_tolerances_at_least() -> None:
    problem = Problem()
    p = problem.translation2_decision_variable()
    p.set_value(np.array([0.5, 0.1]))
    problem.minimize(p.squared_distance(Translation2(1.0, 0.0)))
    problem.subject_to(p.squared_distance(Translation2(0.0, 0.0)) >= 4.0)

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    assert p.x().value() == pytest.approx(2.0, abs=solution_tolerance(1))
    assert p.y().value() == pytest.approx(0.0, abs=solution_tolerance(1))
