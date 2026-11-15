// Copyright (c) Sleipnir contributors

use std::time::Duration;

use sleipnir::{
    DynamicsType, ExpressionType, Ocp, OcpConfig, Options, Problem, SolveStatus, TimestepMethod,
    TranscriptionMethod, VariableMatrix, constrain,
};

/// Solver tolerance multiples, like the C++ tests' solution_tolerance().
fn tolerance(multiple: f64) -> f64 {
    multiple * Options::<f64>::default().tolerance
}

fn degenerate_tolerance(multiple: f64) -> f64 {
    multiple * Options::<f64>::default().tolerance.sqrt()
}

/// Optimal trajectory for xₖ₊₁ = A xₖ + B uₖ tracking r from 0 with
/// |u| ≤ u_max: saturate until one step can land on r, then hold it.
fn optimal_flywheel_trajectory(
    a: f64,
    b: f64,
    r: f64,
    u_max: f64,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n + 1];
    let mut u = vec![0.0; n];
    for k in 0..n {
        if a * x[k] + b * u_max < r {
            u[k] = u_max;
            x[k + 1] = a * x[k] + b * u_max;
        } else {
            u[k] = (r - a * x[k]) / b;
            x[k + 1] = r;
        }
    }
    (x, u)
}

/// Port of the C++ library's OCP flywheel test.
fn flywheel_test(
    dynamics: impl Fn(f64, f64) -> (f64, f64),
    dynamics_type: DynamicsType,
    transcription_method: TranscriptionMethod,
) {
    const N: usize = 1000;
    let dt = Duration::from_millis(5);
    let (a, b) = (-1.0_f64, 1.0);
    let a_discrete = (a * dt.as_secs_f64()).exp();
    let b_discrete = (1.0 - a_discrete) * b;
    let r = 10.0;
    let (coeff_x, coeff_u) = dynamics(a, b);

    let problem: Problem = Problem::new();
    let ocp = Ocp::new(
        &problem,
        OcpConfig {
            dynamics_type,
            transcription_method,
            ..OcpConfig::new(1, 1, dt, N)
        },
        |x, u| x * coeff_x + u * coeff_u,
    );
    ocp.constrain_initial_state(0.0);
    ocp.set_upper_input_bound(12.0);
    ocp.set_lower_input_bound(-12.0);

    let error = ocp.x().map(|x| r - x);
    problem.minimize(error.dot(&error));

    assert_eq!(problem.cost_function_type(), ExpressionType::Quadratic);
    if transcription_method == TranscriptionMethod::SingleShooting {
        // The initial state is set directly instead of constrained
        assert_eq!(problem.equality_constraint_type(), ExpressionType::None);
    } else {
        assert_eq!(problem.equality_constraint_type(), ExpressionType::Linear);
    }
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::Linear);

    assert_eq!(
        problem.solve(&Options::default()),
        Ok(SolveStatus::Success),
        "{transcription_method:?} {dynamics_type:?}"
    );

    let (x_ref, u_ref) = optimal_flywheel_trajectory(a_discrete, b_discrete, r, 12.0, N);
    let x = ocp.x().values();
    let u = ocp.u().values();
    assert!(x[0].abs() <= tolerance(1.0));
    for k in 0..=N {
        if transcription_method == TranscriptionMethod::DirectCollocation {
            // Spline inputs chatter after leaving saturation, so the solution
            // is degenerate
            assert!(
                (x[k] - x_ref[k]).abs() <= degenerate_tolerance(5.0),
                "k = {k}: {} vs {}",
                x[k],
                x_ref[k]
            );
            if k < N {
                assert!((u[k] - u_ref[k]).abs() <= 20.0, "k = {k}");
            }
        } else {
            assert!(
                (x[k] - x_ref[k]).abs() <= tolerance(1000.0),
                "{transcription_method:?} k = {k}: {} vs {}",
                x[k],
                x_ref[k]
            );
            if k < N {
                // The cost depends on uₖ only through Buₖ
                assert!(
                    (b_discrete * (u[k] - u_ref[k])).abs() <= tolerance(1000.0),
                    "{transcription_method:?} k = {k}"
                );
            }
        }
    }
}

#[test]
fn flywheel_explicit() {
    for method in [
        TranscriptionMethod::DirectCollocation,
        TranscriptionMethod::DirectTranscription,
        TranscriptionMethod::SingleShooting,
    ] {
        flywheel_test(|a, b| (a, b), DynamicsType::ExplicitOde, method);
    }
}

#[test]
fn flywheel_discrete() {
    let dt = 0.005_f64;
    let a_discrete = (-dt).exp();
    let b_discrete = 1.0 - a_discrete;
    for method in [
        TranscriptionMethod::DirectTranscription,
        TranscriptionMethod::SingleShooting,
    ] {
        flywheel_test(
            |_, _| (a_discrete, b_discrete),
            DynamicsType::Discrete,
            method,
        );
    }
}

/// Minimum-time double integrator rest-to-rest move of 1 m with |u| ≤ 1,
/// whose optimum is bang-bang with total time 2 s. RK4 is exact for it.
fn minimum_time(timestep_method: TimestepMethod, transcription_method: TranscriptionMethod) {
    const N: usize = 20;
    let problem: Problem = Problem::new();
    let ocp = Ocp::new(
        &problem,
        OcpConfig {
            timestep_method,
            transcription_method,
            ..OcpConfig::new(2, 1, Duration::from_millis(100), N)
        },
        |x, u| VariableMatrix::from([[x[1], u[0]]]).transpose(),
    );
    ocp.constrain_initial_state([[0.0], [0.0]]);
    ocp.constrain_final_state([[1.0], [0.0]]);
    ocp.set_lower_input_bound(-1.0);
    ocp.set_upper_input_bound(1.0);
    ocp.set_min_timestep(Duration::from_millis(1));
    ocp.set_max_timestep(Duration::from_millis(500));

    let total_time = ocp.dt().view(0, ..N).sum();
    problem.minimize(total_time);

    assert_eq!(
        problem.solve(&Options::default()),
        Ok(SolveStatus::Success),
        "{timestep_method:?} {transcription_method:?}"
    );
    assert!(
        (total_time.value() - 2.0).abs() < 1e-4,
        "{timestep_method:?} {transcription_method:?}: T = {}",
        total_time.value()
    );
    let x = ocp.final_state().values();
    assert!((x[0] - 1.0).abs() < 1e-6 && x[1].abs() < 1e-6, "{x:?}");
}

#[test]
fn variable_timesteps() {
    minimum_time(
        TimestepMethod::VariableSingle,
        TranscriptionMethod::DirectTranscription,
    );
    minimum_time(
        TimestepMethod::VariableSingle,
        TranscriptionMethod::SingleShooting,
    );
    minimum_time(
        TimestepMethod::Variable,
        TranscriptionMethod::DirectTranscription,
    );
}

#[test]
fn variable_single_timestep_shares_one_variable() {
    let problem: Problem = Problem::new();
    let ocp = Ocp::new(
        &problem,
        OcpConfig {
            timestep_method: TimestepMethod::VariableSingle,
            ..OcpConfig::new(1, 1, Duration::from_millis(50), 4)
        },
        |x, u| x + u,
    );
    // 1 timestep, 5 inputs, 5 states
    assert_eq!(problem.decision_variables().len(), 11);
    assert!(ocp.dt().values().iter().all(|&dt| dt == 0.05));
}

#[test]
fn for_each_step() {
    let problem: Problem = Problem::new();
    let ocp = Ocp::new(
        &problem,
        OcpConfig::new(1, 1, Duration::from_millis(250), 4),
        |_, u| u.clone(),
    );

    let mut steps = 0;
    ocp.for_each_step(|x, u| {
        assert_eq!((x.rows(), u.rows()), (1, 1));
        problem.subject_to(constrain!(x <= 0.5));
        steps += 1;
    });
    assert_eq!(steps, 5);

    let mut times = Vec::new();
    ocp.for_each_step_with_time(|t, _, _, dt| {
        assert_eq!(dt.value(), 0.25);
        times.push(t.value());
    });
    assert_eq!(times, [0.0, 0.25, 0.5, 0.75, 1.0]);

    // Maximize the final state subject to the per-step bound
    ocp.constrain_initial_state(0.0);
    problem.maximize(ocp.final_state()[0]);
    ocp.set_upper_input_bound(1.0);
    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert!((ocp.final_state()[0].value() - 0.5).abs() < 1e-6);
}

#[test]
fn single_shooting_initial_state() {
    let problem: Problem = Problem::new();
    let ocp = Ocp::new(
        &problem,
        OcpConfig {
            transcription_method: TranscriptionMethod::SingleShooting,
            dynamics_type: DynamicsType::Discrete,
            ..OcpConfig::new(2, 1, Duration::from_millis(100), 3)
        },
        |x, u| x + &VariableMatrix::from([[u[0]], [u[0] * 2.0]]),
    );

    // Constant initial states are set, not constrained
    ocp.constrain_initial_state([[1.0], [2.0]]);
    assert!(problem.equality_constraints().is_empty());
    assert_eq!(ocp.initial_state().values(), [1.0, 2.0]);

    problem.set_initial_matrix(ocp.u(), [[1.0, 1.0, 1.0, 0.0]]);
    assert_eq!(ocp.final_state().values(), [4.0, 8.0]);

    // The initial state isn't a decision variable
    assert_eq!(problem.decision_variables().len(), 4);

    // Changing the initial state changes the trajectory
    ocp.constrain_initial_state(0.0);
    assert_eq!(ocp.final_state().values(), [3.0, 6.0]);
}

#[test]
fn single_shooting_variable_initial_state_is_constrained() {
    let problem: Problem = Problem::new();
    let start = problem.decision_variable();
    let ocp = Ocp::new(
        &problem,
        OcpConfig {
            transcription_method: TranscriptionMethod::SingleShooting,
            ..OcpConfig::new(1, 1, Duration::from_millis(100), 3)
        },
        |_, u| u.clone(),
    );
    ocp.constrain_initial_state(start);
    assert_eq!(problem.equality_constraints().len(), 1);
}

#[test]
#[should_panic(expected = "direct collocation requires an explicit ODE")]
fn collocation_requires_explicit_ode() {
    let problem: Problem = Problem::new();
    let _ = Ocp::new(
        &problem,
        OcpConfig {
            dynamics_type: DynamicsType::Discrete,
            transcription_method: TranscriptionMethod::DirectCollocation,
            ..OcpConfig::new(1, 1, Duration::from_millis(5), 10)
        },
        |x, _| x.clone(),
    );
}

#[test]
#[should_panic(expected = "dynamics must return a 2×1 matrix, got 1×1")]
fn dynamics_shape_mismatch_panics() {
    let problem: Problem = Problem::new();
    let _ = Ocp::new(
        &problem,
        OcpConfig::new(2, 1, Duration::from_millis(5), 10),
        |_, u| u.clone(),
    );
}

#[test]
#[should_panic(expected = "initial state must be a scalar or a 2×1 matrix, got 1×2")]
fn initial_state_shape_mismatch_panics() {
    let problem: Problem = Problem::new();
    let ocp = Ocp::new(
        &problem,
        OcpConfig::new(2, 1, Duration::from_millis(5), 10),
        |x, _| x.clone(),
    );
    ocp.constrain_initial_state([[1.0, 2.0]]);
}
