// Copyright (c) Sleipnir contributors

use std::cmp::Ordering;
use std::panic;

use crate::{Scalar, SolveError, SolveStatus};

/// Result of one solve in [`multistart`].
#[derive(Clone, Debug, PartialEq)]
pub struct MultistartResult<S, T> {
    /// Solver status.
    pub status: Result<SolveStatus, SolveError>,
    /// Solution's cost.
    pub cost: S,
    /// Decision variable values, in the same form as the initial guesses.
    pub variables: T,
}

/// Solves an optimization problem from several initial guesses in parallel
/// and returns the best solution, or `None` without initial guesses.
///
/// `solve` builds and solves a problem from one initial guess. Each call runs
/// on its own thread, so each builds its own [`Problem`](crate::Problem).
/// Solutions with [`SolveStatus::Success`] are always preferred over others,
/// and lower cost breaks ties (NaN costs rank last).
///
/// ```
/// use sleipnir::{MultistartResult, Options, Problem, multistart};
///
/// // x⁴ − 2x² has minima at x = ±1; start from both sides
/// let best = multistart(
///     |&x0: &f64| {
///         let problem: Problem = Problem::new();
///         let x = problem.decision_variable();
///         problem.set_initial(x, x0);
///         let cost = x.powf(4.0) - 2.0 * x * x;
///         problem.minimize(cost);
///         let status = problem.solve(&Options::default());
///         MultistartResult { status, cost: cost.value(), variables: x.value() }
///     },
///     &[-2.0, 0.5, 3.0],
/// )
/// .unwrap();
/// assert!((best.cost + 1.0).abs() < 1e-8);
/// ```
///
/// # Panics
///
/// Resumes a panic from any `solve` call after all threads finish.
pub fn multistart<S, T, F>(solve: F, initial_guesses: &[T]) -> Option<MultistartResult<S, T>>
where
    S: Scalar,
    T: Send + Sync,
    F: Fn(&T) -> MultistartResult<S, T> + Sync,
{
    let results: Vec<_> = std::thread::scope(|scope| {
        let threads: Vec<_> = initial_guesses
            .iter()
            .map(|guess| scope.spawn(|| solve(guess)))
            .collect();
        threads.into_iter().map(|thread| thread.join()).collect()
    });

    let mut best: Option<MultistartResult<S, T>> = None;
    for result in results {
        let result = result.unwrap_or_else(|payload| panic::resume_unwind(payload));
        if best
            .as_ref()
            .is_none_or(|best| rank(&result, best) == Ordering::Less)
        {
            best = Some(result);
        }
    }
    best
}

/// Orders results best first: successful before unsuccessful, then by cost
/// with NaN last.
fn rank<S: Scalar, T>(a: &MultistartResult<S, T>, b: &MultistartResult<S, T>) -> Ordering {
    let success = |r: &MultistartResult<S, T>| r.status == Ok(SolveStatus::Success);
    let is_nan = |cost: S| cost.partial_cmp(&cost).is_none();
    success(b)
        .cmp(&success(a))
        .then_with(|| match a.cost.partial_cmp(&b.cost) {
            Some(ordering) => ordering,
            None => is_nan(a.cost).cmp(&is_nan(b.cost)),
        })
}
