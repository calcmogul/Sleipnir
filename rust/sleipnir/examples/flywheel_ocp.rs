// Copyright (c) Sleipnir contributors

//! Flywheel velocity control with the OCP API.

use std::time::Duration;

use sleipnir::{DynamicsType, Ocp, OcpConfig, Options, Problem, TranscriptionMethod};

fn main() {
    let total_time = Duration::from_secs(5);
    let dt = Duration::from_millis(5);
    let n = (total_time.as_secs_f64() / dt.as_secs_f64()) as usize;

    // Flywheel model:
    // States: [velocity]
    // Inputs: [voltage]
    let a = -1.0_f64;
    let b = 1.0;
    let a_discrete = (a * dt.as_secs_f64()).exp();
    let b_discrete = (1.0 - a_discrete) * b;

    let r = 10.0;

    let problem: Problem = Problem::new();
    let ocp = Ocp::new(
        &problem,
        OcpConfig {
            dynamics_type: DynamicsType::Discrete,
            transcription_method: TranscriptionMethod::DirectTranscription,
            ..OcpConfig::new(1, 1, dt, n)
        },
        |x, u| x * a_discrete + u * b_discrete,
    );
    ocp.constrain_initial_state(0.0);
    ocp.set_upper_input_bound(12.0);
    ocp.set_lower_input_bound(-12.0);

    // Set up cost
    let error = ocp.x().map(|x| r - x);
    problem.minimize(error.dot(&error));

    problem.solve(&Options::default()).unwrap();

    // The first state
    println!("x₀ = {}", ocp.x()[0].value());

    // The first input
    println!("u₀ = {}", ocp.u()[0].value());
}
