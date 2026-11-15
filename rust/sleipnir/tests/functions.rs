// Copyright (c) Sleipnir contributors

use sleipnir::{
    Condition, ExpressionType, Options, Problem, SolveStatus, Variable, constrain, if_else,
};

#[test]
fn new_functions_evaluate_like_cpp() {
    // Pin x and y, then check each function's Rust value against C++'s
    // through an equality constraint on a fresh variable
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    problem.set_initial(x, -0.3);
    problem.set_initial(y, 2.0);

    let exprs = [
        x.erf(),
        y.erf(),
        x.sign(),
        y.sign(),
        (x - x).sign(),
        if_else(Condition::Less, x, y, x * 3.0, y * 5.0),
        if_else(Condition::Greater, x, y, x * 3.0, y * 5.0),
        if_else(Condition::LessEqual, x, x, 1.0, 2.0 * y),
        if_else(Condition::GreaterEqual, x, y, 1.0, 2.0 * y),
        if_else(Condition::Equal, x, x, y, 0.0),
        if_else(Condition::NotEqual, x, x, y, -y),
    ];
    let outputs = problem.decision_variable_matrix(exprs.len(), 1);
    problem.subject_to(constrain!(x == -0.3));
    problem.subject_to(constrain!(y == 2.0));
    for (i, &expr) in exprs.iter().enumerate() {
        problem.subject_to(constrain!(outputs[i] == expr));
    }

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    let expected = [
        -0.328_626_759_459_127_4,
        0.995_322_265_018_952_7,
        -1.0,
        1.0,
        0.0,
        -0.9,
        10.0,
        1.0,
        4.0,
        2.0,
        -2.0,
    ];
    for (i, expr) in exprs.iter().enumerate() {
        assert!(
            (expr.value() - expected[i]).abs() < 1e-12,
            "expr {i}: {}",
            expr.value()
        );
        assert!(
            (outputs[i].value() - expected[i]).abs() < 1e-10,
            "output {i}"
        );
    }
}

#[test]
fn if_else_reevaluates_condition() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let relu = if_else(Condition::Greater, x, 0.0, x, 0.0);

    problem.set_initial(x, -2.0);
    assert_eq!(relu.value(), 0.0);
    problem.set_initial(x, 3.0);
    assert_eq!(relu.value(), 3.0);

    // Constant conditions and equal constant branches fold
    assert!(if_else(Condition::Less, 1.0, 2.0, x, 0.0).expression_type() == ExpressionType::Linear);
    assert!(if_else(Condition::Less, x, 2.0, 5.0, 5.0).is_constant());
}

#[test]
fn expression_types_follow_cpp_rules() {
    use ExpressionType::*;

    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();

    let cases = [
        (Variable::from(2.0), Constant),
        (x, Linear),
        (-x, Linear),
        (x + 1.0, Linear),
        (x * 3.0, Linear),
        (x / 3.0, Linear),
        (3.0 / x, Nonlinear),
        (x * y, Quadratic),
        (x * x + y, Quadratic),
        (x * x * x, Nonlinear),
        (x.powf(2.0), Quadratic),
        (x.powf(3.0), Nonlinear),
        (x.powf(1.0), Linear),
        (x.powf(0.0), Constant),
        (x.powf(0.5), Nonlinear),
        (x.sin(), Nonlinear),
        (x.abs(), Nonlinear),
        (x * 0.0, Constant),
        (x * 1.0, Linear),
        (0.0 - x, Linear),
        (x.hypot(y), Nonlinear),
    ];
    for (i, (expr, expected)) in cases.iter().enumerate() {
        assert_eq!(expr.expression_type(), *expected, "case {i}");
    }

    assert_eq!(problem.cost_function_type(), None);
    assert_eq!(problem.equality_constraint_type(), None);
    problem.minimize(x * y);
    problem.subject_to(constrain!(x + y == 1.0));
    problem.subject_to(constrain!(x * x <= 4.0));
    problem.subject_to(constrain!(x.sin() >= -1.0));
    assert_eq!(problem.cost_function_type(), Quadratic);
    assert_eq!(problem.equality_constraint_type(), Linear);
    assert_eq!(problem.inequality_constraint_type(), Nonlinear);
    assert_eq!(Nonlinear.to_string(), "nonlinear");
}

#[test]
fn pruning() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.set_initial(x, 5.0);

    assert!((x * 0.0).is_constant());
    assert!((0.0 * x).is_constant());
    assert_eq!((x * 0.0).value(), 0.0);
    assert_eq!((x / 1.0).value(), 5.0);
    assert_eq!((x + 0.0).value(), 5.0);
    assert_eq!((0.0 - x).value(), -5.0);
    assert_eq!(x.powf(2.0).value(), 25.0);
    assert_eq!(x.powf(0.5).value(), 5.0_f64.sqrt());
    assert_eq!(Variable::from(0.0).pow(x).value(), 0.0);
    assert_eq!(Variable::from(1.0).pow(x).value(), 1.0);
}

#[test]
fn values_track_decision_variables() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    let f = x * y + x.exp();

    problem.set_initial(x, 0.0);
    problem.set_initial(y, 2.0);
    assert_eq!(f.value(), 1.0);

    // Changing a leaf invalidates cached values that depend on it
    problem.set_initial(y, 3.0);
    assert_eq!(f.value(), 1.0);
    problem.set_initial(x, 1.0);
    assert_eq!(f.value(), 3.0 + 1.0_f64.exp());

    // Expressions recorded after a value change see the new value
    let g = f * 2.0;
    assert_eq!(g.value(), 2.0 * (3.0 + 1.0_f64.exp()));
}
