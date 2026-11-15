// Copyright (c) Sleipnir contributors

use sleipnir::{MultistartResult, Options, Problem, SolveError, SolveStatus, multistart};

/// Returns a solve function that ignores the problem and reports the given
/// (status, cost) for each guess.
fn canned(
    results: &[(Result<SolveStatus, SolveError>, f64)],
) -> impl Fn(&usize) -> MultistartResult<f64, usize> + Sync + '_ {
    |&i| MultistartResult {
        status: results[i].0,
        cost: results[i].1,
        variables: i,
    }
}

#[test]
fn prefers_success_over_lower_cost() {
    let results = [
        (Err(SolveError::MaxIterationsExceeded), -100.0),
        (Ok(SolveStatus::Success), 5.0),
        (Err(SolveError::LocallyInfeasible), -50.0),
        (Ok(SolveStatus::Success), 3.0),
        (Ok(SolveStatus::CallbackRequestedStop), -10.0),
    ];
    // Every order of guesses picks the cheapest success
    for order in [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [3, 0, 4, 2, 1]] {
        let best = multistart(canned(&results), &order).unwrap();
        assert_eq!(best.variables, 3, "order {order:?}");
    }
}

#[test]
fn lowest_cost_without_success_and_nan_last() {
    let results = [
        (Err(SolveError::Timeout), f64::NAN),
        (Err(SolveError::Timeout), 2.0),
        (Err(SolveError::DivergingIterates), 1.0),
    ];
    assert_eq!(
        multistart(canned(&results), &[0, 1, 2]).unwrap().variables,
        2
    );

    let results = [
        (Ok(SolveStatus::Success), f64::NAN),
        (Ok(SolveStatus::Success), 7.0),
    ];
    assert_eq!(multistart(canned(&results), &[0, 1]).unwrap().variables, 1);
}

#[test]
fn no_guesses() {
    assert!(multistart(canned(&[]), &[]).is_none());
}

#[test]
#[should_panic(expected = "solve panicked")]
fn solve_panics_propagate() {
    let _ = multistart(
        |&x: &f64| -> MultistartResult<f64, f64> {
            if x > 0.0 {
                panic!("solve panicked");
            }
            MultistartResult {
                status: Ok(SolveStatus::Success),
                cost: 0.0,
                variables: x,
            }
        },
        &[-1.0, 1.0],
    );
}

#[test]
fn finds_global_minimum() {
    // f(x) = x⁴ − 3x² + x has a local minimum near 1.13 and the global
    // minimum near −1.30
    let best = multistart(
        |&x0: &f64| {
            let problem: Problem = Problem::new();
            let x = problem.decision_variable();
            problem.set_initial(x, x0);
            let cost = x.powf(4.0) - 3.0 * x * x + x;
            problem.minimize(cost);
            let status = problem.solve(&Options::default());
            MultistartResult {
                status,
                cost: cost.value(),
                variables: x.value(),
            }
        },
        &[2.0, 1.0, -2.0, 0.5],
    )
    .unwrap();
    assert_eq!(best.status, Ok(SolveStatus::Success));
    assert!(
        (best.variables + 1.300_839_565_6).abs() < 1e-6,
        "{}",
        best.variables
    );
}
