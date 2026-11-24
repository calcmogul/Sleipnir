import numpy as np
from sleipnir import autodiff
from sleipnir.autodiff import Real, Rotation3, Translation2
from sleipnir.optimization import LintKind, Problem


def test_lint_clean_problem() -> None:
    problem = Problem()

    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    x.set_value(0.5)
    problem.minimize((x - 1) * (x - 1) + (y - 2) * (y - 2))
    problem.subject_to(x + y <= 3)
    problem.subject_to(x >= 0)
    problem.subject_to(x * x + y * y == 4 * x)

    assert problem.lint() == []

    # Linting doesn't change the problem
    assert x.value() == 0.5
    assert y.value() == 0.0


def test_lint_unused_decision_variables() -> None:
    problem = Problem()

    x = problem.real_decision_variable(3)
    problem.rotation2_decision_variable()
    used_rotation = problem.rotation2_decision_variable()
    problem.minimize(x[0] * x[0] + used_rotation.cos())
    problem.subject_to(x[2] >= 1)

    findings = problem.lint()
    assert len(findings) == 1
    assert findings[0].kind == LintKind.UNUSED_DECISION_VARIABLE
    assert findings[0].decision_variables == [1, 3, 4]
    assert findings[0].equality_constraints == []
    assert findings[0].inequality_constraints == []
    assert (
        findings[0].message
        == "Decision variables x[1], x[3], and x[4] don't appear in the cost "
        "function or any constraint. Remove them, or use them in the problem."
    )
    assert str(findings[0]) == findings[0].message
    assert repr(findings[0]).startswith(
        "LintFinding(kind=LintKind.UNUSED_DECISION_VARIABLE, message="
    )


def test_lint_constant_constraints() -> None:
    problem = Problem()

    x = problem.real_decision_variable()
    p = problem.parameter()
    p.set_value(1.0)
    problem.minimize(x * x)
    problem.subject_to(p == 2)
    problem.subject_to(p >= 0)

    findings = problem.lint()
    assert len(findings) == 2

    assert findings[0].kind == LintKind.CONSTANT_CONSTRAINT
    assert findings[0].equality_constraints == [0]
    assert findings[0].inequality_constraints == []
    assert "can't be satisfied" in findings[0].message
    assert "With the parameters' current values" in findings[0].message

    assert findings[1].kind == LintKind.CONSTANT_CONSTRAINT
    assert findings[1].equality_constraints == []
    assert findings[1].inequality_constraints == [0]
    assert "always satisfied" in findings[1].message


def test_lint_sphere_constraints_unit_sphere() -> None:
    problem = Problem()

    v = problem.real_decision_variable(3)
    v[0].set_value(0.3)
    v[1].set_value(0.4)
    v[2].set_value(5.0)
    problem.minimize(v[0] + v[1] + v[2])
    problem.subject_to(v.T @ v == 1)

    findings = problem.lint()
    assert len(findings) == 1
    assert findings[0].kind == LintKind.SPHERE_CONSTRAINT
    assert findings[0].decision_variables == [0, 1, 2]
    assert findings[0].equality_constraints == [0]
    assert (
        findings[0].message
        == "Equality constraint 0 keeps x[0], x[1], and x[2] on the unit sphere. "
        "Creating them with unit_vector_decision_variable(3) instead removes the "
        "constraint and keeps every iterate on the sphere."
    )


def test_lint_sphere_constraints_radius_2_circle() -> None:
    problem = Problem()

    # Radius 2 circle written as 4 = x² + y²
    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x + y)
    problem.subject_to(Real(4.0) == x * x + y * y)

    findings = problem.lint()
    assert len(findings) == 1
    assert findings[0].kind == LintKind.SPHERE_CONSTRAINT
    assert "on a sphere of radius 2" in findings[0].message
    assert "rotation2_decision_variable()" in findings[0].message


def test_lint_sphere_constraints_quaternion() -> None:
    problem = Problem()

    q = problem.real_decision_variable(4)
    problem.minimize(q[0])
    problem.subject_to(q.T @ q == 1)

    findings = problem.lint()
    assert len(findings) == 1
    assert "rotation3_decision_variable()" in findings[0].message


def test_lint_sphere_constraints_not_spheres() -> None:
    problem = Problem()

    x = problem.real_decision_variable(7)
    problem.minimize(x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6])

    # Ellipse
    problem.subject_to(x[0] * x[0] + 2 * x[1] * x[1] == 1)

    # Shifted circle
    problem.subject_to(x[2] * x[2] + x[3] * x[3] + x[2] == 1)

    # Empty set
    problem.subject_to(x[4] * x[4] + x[5] * x[5] == -1)

    # One decision variable
    problem.subject_to(x[6] * x[6] == 1)

    assert problem.lint() == []


def test_lint_sphere_constraints_overlapping_spheres() -> None:
    problem = Problem()

    x = problem.real_decision_variable(3)
    problem.minimize(x[0] + x[1] + x[2])
    problem.subject_to(x[0] * x[0] + x[1] * x[1] == 1)
    problem.subject_to(x[1] * x[1] + x[2] * x[2] == 1)

    assert problem.lint() == []


def test_lint_sphere_constraints_manifold_decision_variables() -> None:
    problem = Problem()

    u = problem.unit_vector_decision_variable(2)
    problem.minimize(u[0])
    problem.subject_to(u[0] * u[0] + u[1] * u[1] == 1)

    assert problem.lint() == []


def test_lint_linear_equality_constraints_redundant() -> None:
    problem = Problem()

    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x * x + y * y)
    problem.subject_to(x + 2 * y == 1)
    problem.subject_to(2 * x + 4 * y == 2)

    findings = problem.lint()
    assert len(findings) == 1
    assert findings[0].kind == LintKind.REDUNDANT_CONSTRAINTS
    assert findings[0].decision_variables == [0, 1]
    assert findings[0].equality_constraints == [0, 1]
    assert (
        findings[0].message
        == "Equality constraints 0 and 1 set x[0] + 2·x[1] = 1. Keep one of them; "
        "duplicate equality constraints make the constraint Jacobian "
        "rank-deficient."
    )


def test_lint_linear_equality_constraints_conflicting() -> None:
    problem = Problem()

    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x * x + y * y)
    problem.subject_to(x - y == 1)
    problem.subject_to(y - x == 1)

    findings = problem.lint()
    assert len(findings) == 1
    assert findings[0].kind == LintKind.CONFLICTING_CONSTRAINTS
    assert findings[0].equality_constraints == [0, 1]
    assert (
        findings[0].message
        == "Equality constraints 0 and 1 set x[0] − x[1] to different values "
        "(1 and -1), so the problem is infeasible."
    )


def test_lint_linear_equality_constraints_inequality_constraints_implied_or_violated() -> (
    None
):
    problem = Problem()

    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x * x + y * y)
    problem.subject_to(x + y == 1)
    problem.subject_to(x + y >= 0)
    problem.subject_to(x + y <= 0)
    problem.subject_to(x + y <= 1)

    findings = problem.lint()
    assert len(findings) == 2

    assert findings[0].kind == LintKind.REDUNDANT_CONSTRAINTS
    assert findings[0].equality_constraints == [0]
    assert findings[0].inequality_constraints == [0, 2]
    assert (
        findings[0].message
        == "Equality constraint 0 sets x[0] + x[1] = 1, which satisfies "
        "inequality constraints 0 and 2. Remove them."
    )

    assert findings[1].kind == LintKind.CONFLICTING_CONSTRAINTS
    assert findings[1].equality_constraints == [0]
    assert findings[1].inequality_constraints == [1]


def test_lint_linear_inequality_constraints_redundant_bounds() -> None:
    problem = Problem()

    x = problem.real_decision_variable()
    problem.minimize(x * x)
    problem.subject_to(x >= 0)
    problem.subject_to(x >= 1)
    problem.subject_to(x <= 5)
    problem.subject_to(-x >= -3)

    findings = problem.lint()
    assert len(findings) == 2

    assert findings[0].kind == LintKind.REDUNDANT_CONSTRAINTS
    assert findings[0].inequality_constraints == [0, 1]
    assert (
        findings[0].message
        == "Inequality constraint 1 requires x[0] ≥ 1, which implies inequality "
        "constraint 0. Remove it."
    )

    assert findings[1].kind == LintKind.REDUNDANT_CONSTRAINTS
    assert findings[1].inequality_constraints == [2, 3]
    assert (
        findings[1].message
        == "Inequality constraint 3 requires x[0] ≤ 3, which implies inequality "
        "constraint 2. Remove it."
    )


def test_lint_linear_inequality_constraints_equality_as_inequalities() -> None:
    problem = Problem()

    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x * x + y * y)
    problem.subject_to(2 * x - 2 * y >= 2)
    problem.subject_to(x - y <= 1)

    findings = problem.lint()
    assert len(findings) == 1
    assert findings[0].kind == LintKind.EQUALITY_AS_INEQUALITIES
    assert findings[0].inequality_constraints == [0, 1]
    assert (
        findings[0].message
        == "Inequality constraints 0 and 1 require x[0] − x[1] ≥ 1 and "
        "x[0] − x[1] ≤ 1. Replace them with the equality constraint "
        "x[0] − x[1] = 1, which the solver handles directly instead of through "
        "two inequality constraints with no strict interior."
    )


def test_lint_linear_inequality_constraints_conflicting_bounds() -> None:
    problem = Problem()

    x = problem.real_decision_variable()
    problem.minimize(x * x)
    problem.subject_to(x >= 2)
    problem.subject_to(x <= 1)

    findings = problem.lint()
    assert len(findings) == 1
    assert findings[0].kind == LintKind.CONFLICTING_CONSTRAINTS
    assert findings[0].inequality_constraints == [0, 1]


def test_lint_linear_inequality_constraints_different_linear_combinations() -> None:
    problem = Problem()

    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x * x + y * y)
    problem.subject_to(x + y >= 0)
    problem.subject_to(x + 2 * y >= 1)
    problem.subject_to(x - y <= 1)

    assert problem.lint() == []


def test_lint_nonsmooth_functions_abs() -> None:
    problem = Problem()

    # abs(a) ≤ b
    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x * x + y * y)
    problem.subject_to(autodiff.abs(y - 1) <= 2)
    problem.subject_to(autodiff.abs(x) <= y)

    findings = problem.lint()
    assert len(findings) == 1
    assert findings[0].kind == LintKind.NONSMOOTH_FUNCTION
    assert not findings[0].cost
    assert findings[0].equality_constraints == []
    assert findings[0].inequality_constraints == [0, 1]
    assert (
        findings[0].message
        == "Inequality constraints 0 and 1 have the form abs(a) ≤ b, where a and b "
        "are expressions. abs(a) isn't differentiable where a = 0. Write them as "
        "−b ≤ a ≤ b instead."
    )


def test_lint_nonsmooth_functions_max_and_min() -> None:
    problem = Problem()

    # max(a, b) ≤ c and min(a, b) ≥ c
    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x * x + y * y)
    problem.subject_to(autodiff.max(x, y) <= 1)
    problem.subject_to(-1 <= autodiff.min(x, y))

    findings = problem.lint()
    assert len(findings) == 2
    assert findings[0].inequality_constraints == [0]
    assert (
        findings[0].message
        == "Inequality constraint 0 has the form max(a, b) ≤ c, where a, b, and c "
        "are expressions. max(a, b) isn't differentiable where a = b. Write it as "
        "a ≤ c and b ≤ c instead."
    )
    assert findings[1].inequality_constraints == [1]
    assert "Write it as a ≥ c and b ≥ c instead." in findings[1].message


def test_lint_nonsmooth_functions_norms() -> None:
    problem = Problem()

    t = problem.translation2_decision_variable()
    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    r = problem.parameter()
    r.set_value(2.0)
    target = Translation2(1.0, 2.0)
    problem.minimize(autodiff.sqrt(x * x + y * y))
    problem.subject_to(t.distance(target) <= 1)
    problem.subject_to(autodiff.hypot(x, y) >= r)
    problem.subject_to(t.distance(target) == 1)

    findings = problem.lint()
    assert len(findings) == 2
    assert findings[0].cost
    assert findings[0].equality_constraints == []
    assert findings[0].inequality_constraints == []
    assert (
        findings[0].message
        == "The cost function is a norm, like distance() or hypot(), which isn't "
        "differentiable where it's 0. Minimize the squared norm, like "
        "squared_distance(), instead, which has the same minimizer."
    )
    assert not findings[1].cost
    assert findings[1].equality_constraints == [0]
    assert findings[1].inequality_constraints == [0, 1]
    assert (
        findings[1].message
        == "Equality constraint 0 and inequality constraints 0 and 1 compare a "
        "norm, like distance() or hypot(), to a value that doesn't depend on "
        "decision variables. The norm isn't differentiable where it's 0. Compare "
        "the squared norm, like squared_distance(), to the value's square "
        "instead."
    )


def test_lint_nonsmooth_functions_forms_without_a_differentiable_replacement() -> None:
    problem = Problem()

    a = problem.translation2_decision_variable()
    b = problem.translation2_decision_variable()
    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    start = Translation2(0.0, 0.0)
    end = Translation2(3.0, 4.0)

    # Path length and L1 cost
    problem.minimize(
        a.distance(start)
        + b.distance(a)
        + end.distance(b)
        + autodiff.abs(x)
        + autodiff.max(x, y)
    )

    # Bounds of the other direction, equalities, and norms compared to decision
    # variables
    problem.subject_to(autodiff.abs(x) >= 1)
    problem.subject_to(autodiff.abs(y) == 1)
    problem.subject_to(autodiff.max(x, y) >= 1)
    problem.subject_to(autodiff.min(x, y) <= 1)
    problem.subject_to(autodiff.sign(x) == 1)
    problem.subject_to(a.distance(b) <= x)

    assert problem.lint() == []


def test_lint_nonsmooth_functions_smooth_uses() -> None:
    problem = Problem()

    t = problem.translation2_decision_variable()
    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    target = Translation2(1.0, 2.0)
    problem.minimize(autodiff.sqrt(x * x + y * y + 1) + autodiff.sqrt(x + 2))
    problem.subject_to(t.squared_distance(target) <= 1)
    problem.subject_to(autodiff.hypot(x, 1.0) <= 2)
    p = problem.parameter()
    p.set_value(-3.0)
    problem.subject_to(autodiff.abs(p) * y >= 0)

    assert problem.lint() == []


def test_lint_nonsmooth_functions_manifold_types() -> None:
    problem = Problem()

    q = problem.rotation3_decision_variable()
    u = problem.unit_vector_decision_variable(3)
    p = problem.pose3_decision_variable()
    problem.minimize(
        (q.to_vector().T @ q.to_vector())[0]
        + (p.log().angular().T @ p.log().angular())[0]
        + (p.log().linear().T @ p.log().linear())[0]
    )
    problem.subject_to(q == Rotation3(0.5, 0.5, 0.5, 0.5))
    problem.subject_to(u == np.array([0.0, 0.0, 1.0]))

    assert problem.lint() == []


def test_lint_badly_scaled_coefficients_within_a_linear_constraint() -> None:
    problem = Problem()

    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x * x)
    problem.subject_to(x + 1e9 * y >= 1)
    problem.subject_to(x - 2e9 * y <= 3)

    findings = problem.lint()
    assert len(findings) == 1
    assert findings[0].kind == LintKind.BADLY_SCALED
    assert not findings[0].cost
    assert findings[0].decision_variables == [1]
    assert findings[0].inequality_constraints == [0, 1]
    assert (
        findings[0].message
        == "Coefficients in inequality constraints 0 and 1 span a ratio of up to "
        "2.0e+09, which costs the solver about half its precision. Substituting "
        "x[1] = 1e-09·y[1] for a new decision variable y[1], like by changing "
        "units, reduces it to 2.0e+00."
    )


def test_lint_badly_scaled_coefficients_within_a_quadratic_cost_function() -> None:
    problem = Problem()

    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(1e-5 * x * x + 1e5 * y * y)

    findings = problem.lint()
    assert len(findings) == 1
    assert findings[0].kind == LintKind.BADLY_SCALED
    assert findings[0].cost
    assert findings[0].decision_variables == [1]
    assert "Coefficients in the cost function" in findings[0].message
    assert "x[1] = 1e-05·y[1]" in findings[0].message


def test_lint_badly_scaled_coefficients_not_fixable_by_rescaling_decision_variables() -> (
    None
):
    problem = Problem()

    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x * x + y * y)
    problem.subject_to(x + 1e9 * y >= 1)
    problem.subject_to(1e9 * x + y >= 1)

    assert problem.lint() == []


def test_lint_badly_scaled_coefficients_small_constraint_gradient() -> None:
    problem = Problem()

    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x * x + y * y)
    problem.subject_to(1e-9 * x >= -1)
    problem.subject_to(y >= 1)

    findings = problem.lint()
    assert len(findings) == 1
    assert findings[0].kind == LintKind.BADLY_SCALED
    assert findings[0].inequality_constraints == [0]
    assert (
        findings[0].message
        == "Inequality constraint 0 has a gradient 1.0e+09 times smaller than "
        "inequality constraint 1's, even after the solver scales down gradients "
        "larger than 100. Multiply it by 1e+09 so its gradient is closer to 1."
    )


def test_lint_badly_scaled_coefficients_large_gradients_the_solver_scales_down() -> (
    None
):
    problem = Problem()

    x = problem.real_decision_variable()
    y = problem.real_decision_variable()
    problem.minimize(x * x + y * y)
    problem.subject_to(1e12 * x >= -1)
    problem.subject_to(y >= 1e-3)
    problem.subject_to(1e-3 * x + y >= 0)

    assert problem.lint() == []
