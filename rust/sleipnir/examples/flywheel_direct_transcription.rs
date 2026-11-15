// Copyright (c) Sleipnir contributors

//! Flywheel velocity control with hand-written direct transcription.

use sleipnir::{Options, Problem, Variable, constrain};

fn main() {
    let total_time = 5.0;
    let dt = 0.005_f64;
    let n = (total_time / dt) as usize;

    // Flywheel model:
    // States: [velocity]
    // Inputs: [voltage]
    let a = [[(-dt).exp()]];
    let b = [[1.0 - (-dt).exp()]];

    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(1, n + 1);
    let u = problem.decision_variable_matrix(1, n);

    // Dynamics constraint
    for k in 0..n {
        problem.subject_to(constrain!(
            x.column(k + 1) == a * x.column(k) + b * u.column(k)
        ));
    }

    // State and input constraints
    problem.subject_to(constrain!(x.column(0) == 0.0));
    problem.subject_to(constrain!(-12.0 <= u <= 12.0));

    // Cost function - minimize error
    let r = 10.0;
    let cost: Variable<'_, f64> = x.iter().map(|&x_k| (r - x_k) * (r - x_k)).sum();
    problem.minimize(cost);

    problem.solve(&Options::default()).unwrap();

    // The first state
    println!("x₀ = {}", x[0].value());

    // The first input
    println!("u₀ = {}", u[0].value());
}
