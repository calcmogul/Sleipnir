// Copyright (c) Sleipnir contributors

use std::ops::ControlFlow;

use sleipnir::{Options, Problem, SolveError, SolveStatus, Variable, VariableMatrix, constrain};

fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn unconstrained_quadratic() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.minimize((x - 3.0) * (x - 3.0) + (y + 1.0) * (y + 1.0));

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert_near(x.value(), 3.0, 1e-8);
    assert_near(y.value(), -1.0, 1e-8);
}

#[test]
fn rosenbrock() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    problem.set_initial(x, -1.2);
    problem.set_initial(y, 1.0);

    problem.minimize(100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x));

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert_near(x.value(), 1.0, 1e-6);
    assert_near(y.value(), 1.0, 1e-6);
}

#[test]
fn equality_constrained() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.minimize(x * x + y * y);
    problem.subject_to(constrain!(x + y == 1.0));

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert_near(x.value(), 0.5, 1e-8);
    assert_near(y.value(), 0.5, 1e-8);
}

#[test]
fn inequality_constrained() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.minimize((x - 3.0) * (x - 3.0) + (y - 3.0) * (y - 3.0));
    problem.subject_to(constrain!(x <= 1.0));
    problem.subject_to(constrain!(-2.0 <= y <= 2.0));

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert_near(x.value(), 1.0, 1e-6);
    assert_near(y.value(), 2.0, 1e-6);
}

#[test]
fn math_functions() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    problem.set_initial(x, 0.1);
    problem.set_initial(y, 1.0);

    // sin(x) = 1/2 on [0, 1.5] and exp(ln(y)) = √2 at the optimum
    let e = x.sin() - 0.5;
    let f = y.ln().exp() - 2.0_f64.sqrt();
    problem.minimize(e * e + f * f);
    problem.subject_to(constrain!(0.0 <= x <= 1.5));
    problem.subject_to(constrain!(y >= 0.5));

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert_near(x.value(), std::f64::consts::FRAC_PI_6, 1e-6);
    assert_near(y.value(), 2.0_f64.sqrt(), 1e-6);
}

#[test]
fn rust_and_cpp_evaluate_functions_identically() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    problem.set_initial(x, -0.3);
    problem.set_initial(y, 2.0);

    // Pin x and y, then check every function's value against the Rust
    // evaluation through an equality constraint on a fresh variable
    let exprs = [
        x.abs(),
        x.acos(),
        x.asin(),
        x.atan(),
        x.cbrt(),
        x.cos(),
        x.cosh(),
        x.exp(),
        y.ln(),
        y.log10(),
        x.sin(),
        x.sinh(),
        y.sqrt(),
        x.tan(),
        x.tanh(),
        y.pow(x),
        y.powf(1.5),
        x.atan2(y),
        x.hypot(y),
        x.max(y),
        x.min(y),
    ];
    let outputs = problem.decision_variable_matrix(exprs.len(), 1);
    problem.subject_to(constrain!(x == -0.3));
    problem.subject_to(constrain!(y == 2.0));
    for (i, &expr) in exprs.iter().enumerate() {
        problem.subject_to(constrain!(outputs[i] == expr));
    }

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    for (i, expr) in exprs.iter().enumerate() {
        assert_near(outputs[i].value(), expr.value(), 1e-10);
    }
}

#[test]
fn maximize() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();

    problem.maximize(-(x - 2.0) * (x - 2.0));

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert_near(x.value(), 2.0, 1e-8);
}

#[test]
fn matrix_least_squares() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(2, 1);

    // A x = b with A = [[1, 2], [3, 4]], b = [5, 6] → x = [−4, 4.5]
    let a = VariableMatrix::from([[1.0, 2.0], [3.0, 4.0]]);
    let b = VariableMatrix::from([[5.0], [6.0]]);
    let r = &(&a * &x) - &b;
    problem.minimize(
        r.iter()
            .fold(0.0.into(), |sum: Variable<'_, f64>, &e| sum + e * e),
    );

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert_near(x[0].value(), -4.0, 1e-6);
    assert_near(x[1].value(), 4.5, 1e-6);
}

#[test]
fn constant_problem() {
    let problem: Problem = Problem::new();
    problem.minimize(Variable::from(2.0));
    problem.subject_to(constrain!(Variable::from(1.0) >= 0.0));

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert_eq!(problem.cost().unwrap().value(), 2.0);
}

#[test]
fn f32_problem() {
    let problem = Problem::<f32>::new();
    let x = problem.decision_variable();

    problem.minimize((x - 3.0) * (x - 3.0));
    problem.subject_to(constrain!(x <= 1.0));

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert!((x.value() - 1.0).abs() <= 1e-3, "x = {}", x.value());
}

#[test]
fn callback_sees_iterates() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    problem.set_initial(x, -1.2);
    problem.set_initial(y, 1.0);
    let cost = 100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x);
    problem.minimize(cost);

    let mut iterations = Vec::new();
    let mut costs = Vec::new();
    let status = problem.solve_with_callback(&Options::default(), |info| {
        iterations.push(info.iteration);
        assert_eq!(info.x.len(), 2);
        costs.push(info.value(cost));
        ControlFlow::Continue(())
    });

    assert_eq!(status, Ok(SolveStatus::Success));
    assert_eq!(iterations, (0..iterations.len() as u32).collect::<Vec<_>>());
    assert_near(costs[0], 24.2, 1e-12);
    assert!(costs.last().unwrap() < &1e-6);
}

#[test]
fn callback_stops_solve() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.set_initial(x, 10.0);
    problem.minimize(x.powf(4.0));

    let status = problem.solve_with_callback(&Options::default(), |info| {
        if info.iteration == 2 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });

    assert_eq!(status, Ok(SolveStatus::CallbackRequestedStop));
    assert!(x.value() < 10.0 && x.value() > 0.0, "x = {}", x.value());
}

#[test]
fn callback_can_record_expressions() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.minimize((x - 1.0) * (x - 1.0));

    let status = problem.solve_with_callback(&Options::default(), |info| {
        let _ = info.value(x * 2.0);
        ControlFlow::Continue(())
    });
    assert_eq!(status, Ok(SolveStatus::Success));
}

#[test]
#[should_panic(expected = "callback panic")]
fn callback_panic_propagates() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.minimize((x - 1.0) * (x - 1.0));

    let _ = problem.solve_with_callback(&Options::default(), |_| panic!("callback panic"));
}

#[test]
fn too_few_dofs() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();

    problem.minimize(x * x);
    problem.subject_to(constrain!(x == 1.0));
    problem.subject_to(constrain!(x * 2.0 == 3.0));

    assert_eq!(
        problem.solve(&Options::default()),
        Err(SolveError::TooFewDofs)
    );
}

#[test]
fn infeasible() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();

    problem.minimize(x * x);
    problem.subject_to(constrain!(x >= 1.0));
    problem.subject_to(constrain!(x <= -1.0));

    let error = problem.solve(&Options::default()).unwrap_err();
    assert!(
        matches!(
            error,
            SolveError::LocallyInfeasible | SolveError::GloballyInfeasible
        ),
        "{error:?}"
    );
}

#[test]
fn max_iterations_keeps_last_iterate() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    problem.set_initial(x, -1.2);
    problem.set_initial(y, 1.0);
    problem.minimize(100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x));

    let options = Options {
        max_iterations: 2,
        ..Default::default()
    };
    assert_eq!(
        problem.solve(&options),
        Err(SolveError::MaxIterationsExceeded)
    );
    assert_ne!((x.value(), y.value()), (-1.2, 1.0));
}

#[test]
fn warm_start_from_previous_solve() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.minimize((x - 3.0) * (x - 3.0) * (x - 3.0) * (x - 3.0));

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    let mut iterations = 0;
    let status = problem.solve_with_callback(&Options::default(), |_| {
        iterations += 1;
        ControlFlow::Continue(())
    });
    assert_eq!(status, Ok(SolveStatus::Success));
    assert!(iterations <= 1, "{iterations} iterations");
}
