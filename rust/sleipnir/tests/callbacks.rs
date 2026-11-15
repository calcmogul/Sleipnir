// Copyright (c) Sleipnir contributors

use std::ops::ControlFlow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use sleipnir::{Options, Problem, SolveStatus, constrain};

/// Rosenbrock problem, which takes several iterations.
fn rosenbrock() -> Problem {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    problem.set_initial(x, -1.2);
    problem.set_initial(y, 1.0);
    problem.minimize(100.0 * (y - x * x).powf(2.0) + (1.0 - x).powf(2.0));
    problem
}

#[test]
fn stored_callbacks() {
    let problem = rosenbrock();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    problem.add_callback(move |_| {
        counter.fetch_add(1, Ordering::Relaxed);
        ControlFlow::Continue(())
    });

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    let first_solve = calls.load(Ordering::Relaxed);
    assert!(first_solve > 1);

    // Callbacks persist across solves until cleared
    problem.set_initial(problem.decision_variables()[0], -1.2);
    problem.solve(&Options::default()).unwrap();
    assert!(calls.load(Ordering::Relaxed) > first_solve);

    problem.clear_callbacks();
    let before = calls.load(Ordering::Relaxed);
    problem.solve(&Options::default()).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), before);
}

#[test]
fn callback_order_and_persistence() {
    let problem = rosenbrock();
    let log = Arc::new(Mutex::new(Vec::new()));

    let l = log.clone();
    problem.add_persistent_callback(move |info| {
        l.lock().unwrap().push(("persistent", info.iteration));
        ControlFlow::Continue(())
    });
    let l = log.clone();
    problem.add_callback(move |info| {
        l.lock().unwrap().push(("stored", info.iteration));
        ControlFlow::Continue(())
    });

    let l = log.clone();
    let status = problem.solve_with_callback(&Options::default(), |info| {
        l.lock().unwrap().push(("solve", info.iteration));
        if info.iteration == 1 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(status, Ok(SolveStatus::CallbackRequestedStop));
    assert_eq!(
        *log.lock().unwrap(),
        [
            ("stored", 0),
            ("solve", 0),
            ("persistent", 0),
            ("stored", 1),
            ("solve", 1),
        ]
    );

    // clear_callbacks() keeps persistent callbacks
    problem.clear_callbacks();
    log.lock().unwrap().clear();
    problem.set_initial(problem.decision_variables()[0], -1.2);
    problem.solve(&Options::default()).unwrap();
    let log = log.lock().unwrap();
    assert!(!log.is_empty() && log.iter().all(|&(name, _)| name == "persistent"));
}

#[test]
fn stored_callback_can_stop_solve() {
    let problem = rosenbrock();
    problem.add_callback(|info| {
        if info.iteration == 2 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(
        problem.solve(&Options::default()),
        Ok(SolveStatus::CallbackRequestedStop)
    );
}

#[test]
fn iteration_info_matrices() {
    // 3 variables, 1 equality constraint, 2 inequality constraints
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(3, 1);
    problem.minimize(x[0] * x[0] + x[1] * x[2] + x[2]);
    problem.subject_to(constrain!(x[0] + 2.0 * x[1] == 1.0));
    problem.subject_to(constrain!(x[2] >= 0.0));
    problem.subject_to(constrain!(x[1] * x[1] <= 4.0));

    let mut checked = false;
    problem
        .solve_with_callback(&Options::default(), |info| {
            assert_eq!(info.x.len(), 3);
            assert_eq!((info.y.len(), info.s.len(), info.z.len()), (1, 2, 2));
            assert_eq!((info.g.nrows(), info.g.ncols()), (3, 1));
            assert_eq!((info.h.nrows(), info.h.ncols()), (3, 3));
            assert_eq!((info.a_e.nrows(), info.a_e.ncols()), (1, 3));
            assert_eq!((info.a_i.nrows(), info.a_i.ncols()), (2, 3));

            // The Hessian is the lower triangle
            assert!(info.h.triplet_iter().all(|(row, col, _)| col <= row));

            // Linear equality constraint Jacobian is its coefficients
            let a_e: Vec<_> = info
                .a_e
                .triplet_iter()
                .map(|(_, col, &v)| (col, v))
                .collect();
            assert_eq!(a_e, [(0, 1.0), (1, 2.0)]);

            // ∂f/∂x₂ = x₁ + 1
            let g = info.g.get_entry(2, 0).unwrap().into_value();
            assert!((g - (info.x[1] + 1.0)).abs() < 1e-12);

            checked = true;
            ControlFlow::Break(())
        })
        .unwrap();
    assert!(checked);
}

#[test]
fn rebuilds_after_structure_changes() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.minimize((x - 3.0) * (x - 3.0));
    problem.solve(&Options::default()).unwrap();
    assert!((x.value() - 3.0).abs() < 1e-8);

    // A constraint added after solving is enforced
    problem.subject_to(constrain!(x <= 1.0));
    problem.solve(&Options::default()).unwrap();
    assert!((x.value() - 1.0).abs() < 1e-6);

    // So is a new cost and decision variable
    let y = problem.decision_variable();
    problem.minimize((x - 3.0) * (x - 3.0) + (y + 2.0) * (y + 2.0));
    problem.solve(&Options::default()).unwrap();
    assert!((x.value() - 1.0).abs() < 1e-6 && (y.value() + 2.0).abs() < 1e-6);
}

#[test]
fn reused_problem_sees_new_initial_values() {
    // x⁴ − 2x² has minima at ±1, so the starting side picks the minimum
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.minimize(x.powf(4.0) - 2.0 * x * x);

    for start in [3.0, -3.0, 0.5, -0.5] {
        problem.set_initial(x, start);
        problem.solve(&Options::default()).unwrap();
        assert!(
            (x.value() - start.signum()).abs() < 1e-6,
            "start {start}: {}",
            x.value()
        );
    }
}
