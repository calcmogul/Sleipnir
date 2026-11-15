// Copyright (c) Sleipnir contributors

use sleipnir::{ExpressionType, Gradient, Options, Problem, SolveStatus, constrain};

#[test]
fn parameters_are_not_decision_variables() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let p = problem.parameter(2.0);
    let m = problem.parameter_matrix(2, 3, 1.5);

    assert_eq!(problem.decision_variables().len(), 1);
    assert_eq!(p.value(), 2.0);
    assert_eq!(m.values(), [1.5; 6]);

    // Like the C++ library, parameters count as linear
    assert_eq!(p.expression_type(), ExpressionType::Linear);
    assert_eq!((p * x).expression_type(), ExpressionType::Quadratic);
}

#[test]
fn resolve_with_new_parameter_values() {
    // min (x − a)² + (y − b)² s.t. x + y ≤ c, with parameters a, b, c
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    let params = problem.parameter_matrix(3, 1, 0.0);
    let (a, b, c) = (params[0], params[1], params[2]);
    problem.minimize((x - a) * (x - a) + (y - b) * (y - b));
    problem.subject_to(constrain!(x + y <= c));

    // Unconstrained optimum is feasible
    problem.set_value_matrix(&params, [[1.0], [2.0], [10.0]]);
    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert!((x.value() - 1.0).abs() < 1e-6 && (y.value() - 2.0).abs() < 1e-6);

    // The constraint becomes active: projection of (3, 3) onto x + y ≤ 4
    problem.set_value(a, 3.0);
    problem.set_value(b, 3.0);
    problem.set_value(c, 4.0);
    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert!((x.value() - 2.0).abs() < 1e-6 && (y.value() - 2.0).abs() < 1e-6);
}

#[test]
fn parameters_in_values_and_derivatives() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let p = problem.parameter(3.0);
    problem.set_initial(x, 2.0);

    let f = p * x * x;
    assert_eq!(f.value(), 12.0);
    problem.set_value(p, 5.0);
    assert_eq!(f.value(), 20.0);

    // ∂f/∂x = 2px and ∂f/∂p = x²
    assert_eq!(Gradient::new(f, x).get().values(), [20.0]);
    assert_eq!(Gradient::new(f, p).get().values(), [4.0]);
}

#[test]
fn set_value_on_decision_variable() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.set_value(x, 4.0);
    assert_eq!(x.value(), 4.0);
}

#[test]
#[should_panic(expected = "parameters aren't decision variables; use set_value()")]
fn set_initial_on_parameter_panics() {
    let problem: Problem = Problem::new();
    let p = problem.parameter(1.0);
    problem.set_initial(p, 2.0);
}

#[test]
#[should_panic(expected = "constants don't have settable values")]
fn set_value_on_constant_panics() {
    let problem: Problem = Problem::new();
    problem.set_value(sleipnir::Variable::from(1.0), 2.0);
}
