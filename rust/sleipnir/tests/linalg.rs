// Copyright (c) Sleipnir contributors

use sleipnir::{Options, Problem, SolveStatus, VariableMatrix, constrain};

fn assert_near(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= tolerance,
            "expected {expected:?}, got {actual:?}"
        );
    }
}

/// Well-conditioned n×n test matrix: diagonally dominant with mixed signs.
fn test_matrix(n: usize) -> VariableMatrix<'static, f64> {
    VariableMatrix::from_fn(n, n, |row, col| {
        if row == col {
            (n as f64 + 1.0 + row as f64).into()
        } else {
            (((row * 7 + col * 3) % 5) as f64 - 2.0).into()
        }
    })
}

/// Returns A X − B as a flat vector of values.
fn residual(
    a: &VariableMatrix<'_, f64>,
    x: &VariableMatrix<'_, f64>,
    b: &VariableMatrix<'_, f64>,
) -> Vec<f64> {
    (a * x - b).values()
}

#[test]
fn square_constant_systems() {
    // 1–4 use the closed-form inverse, 5–7 use Householder QR
    for n in 1..=7 {
        let a = test_matrix(n);
        let b = VariableMatrix::from_fn(n, 2, |row, col| {
            ((row + 1) as f64 * (col as f64 - 0.5)).into()
        });
        let x = a.solve(&b);
        assert_eq!((x.rows(), x.cols()), (n, 2));
        assert!(x.iter().all(|x| x.is_constant()), "n = {n}");
        assert_near(&residual(&a, &x, &b), &vec![0.0; 2 * n], 1e-12);
    }
}

#[test]
fn negative_pivots() {
    // Leading entries of both signs exercise both Householder sign choices
    let a = VariableMatrix::<f64>::from([
        [-4.0, 1.0, 0.0, 2.0, 1.0],
        [1.0, 5.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, -6.0, 1.0, 2.0],
        [2.0, 0.0, 1.0, 7.0, -1.0],
        [1.0, 0.0, 2.0, -1.0, -8.0],
    ]);
    let b = VariableMatrix::from([[1.0], [2.0], [3.0], [4.0], [5.0]]);
    let x = a.solve(&b);
    assert_near(&residual(&a, &x, &b), &[0.0; 5], 1e-12);
}

#[test]
fn least_squares() {
    // Overdetermined: the residual is orthogonal to A's columns
    let a = VariableMatrix::<f64>::from([
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 1.0],
        [1.0, 3.0, 0.0],
        [1.0, 4.0, 1.0],
        [1.0, 5.0, 2.0],
    ]);
    let b = VariableMatrix::from([[1.0], [2.0], [2.0], [4.0], [5.0], [7.0]]);
    let x = a.solve(&b);
    assert_eq!((x.rows(), x.cols()), (3, 1));
    let normal = (a.transpose() * (&a * &x - &b)).values();
    assert_near(&normal, &[0.0; 3], 1e-12);

    // Consistent overdetermined systems are solved exactly
    let exact = VariableMatrix::from([[2.0], [-1.0], [0.5]]);
    let x = a.solve(&a * &exact);
    assert_near(&x.values(), &exact.values(), 1e-12);
}

#[test]
fn right_hand_side_types() {
    let a = VariableMatrix::<f64>::from([[2.0, 0.0], [0.0, 4.0]]);
    assert_eq!(a.solve([[2.0], [2.0]]).values(), [1.0, 0.5]);

    let b = VariableMatrix::from([[2.0, 4.0], [8.0, 8.0]]);
    assert_eq!(a.solve(b.column(1)).values(), [2.0, 2.0]);
    assert_eq!(a.solve(&b).values(), [1.0, 2.0, 2.0, 2.0]);
}

#[test]
fn empty_system() {
    let a = VariableMatrix::<f64>::zeros(0, 0);
    let x = a.solve(VariableMatrix::zeros(0, 3));
    assert_eq!((x.rows(), x.cols()), (0, 3));
}

#[test]
fn solve_with_variable_matrix() {
    // Fit s so solve(s A, b) = A⁻¹b / 2, i.e., s = 2, for every solve path
    for n in [1, 2, 3, 4, 5, 6] {
        let problem: Problem = Problem::new();
        let s = problem.decision_variable();
        problem.set_initial(s, 1.0);

        let a = test_matrix(n);
        let b = VariableMatrix::from_fn(n, 1, |row, _| (row as f64 + 1.0).into());
        let target = &a.solve(&b) * 0.5;

        let error = (&a * s).solve(&b) - &target;
        problem.minimize(error.dot(&error));
        assert_eq!(
            problem.solve(&Options::default()),
            Ok(SolveStatus::Success),
            "n = {n}"
        );
        assert!((s.value() - 2.0).abs() < 1e-8, "n = {n}, s = {}", s.value());
    }
}

#[test]
fn solve_with_variable_right_hand_side() {
    // x = A b with A constant, recovered by constraining solve(A, x) = b
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(5, 1);
    let a = test_matrix(5);
    let b = VariableMatrix::from([[1.0], [-2.0], [3.0], [-4.0], [5.0]]);

    problem.subject_to(constrain!(a.solve(&x) == b));
    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    assert_near(&x.values(), &(&a * &b).values(), 1e-8);
}

#[test]
#[should_panic(expected = "solve needs A and B to have the same number of rows, got 2×2 and 3×1")]
fn row_mismatch_panics() {
    let _ = VariableMatrix::<f64>::identity(2).solve(VariableMatrix::zeros(3, 1));
}

#[test]
#[should_panic(expected = "solve needs A to have at least as many rows as columns, got 2×3")]
fn underdetermined_panics() {
    let _ = VariableMatrix::<f64>::zeros(2, 3).solve(VariableMatrix::zeros(2, 1));
}
