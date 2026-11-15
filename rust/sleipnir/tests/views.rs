// Copyright (c) Sleipnir contributors

// The tests exercise owned and borrowed operands
#![allow(clippy::op_ref)]

use sleipnir::{Options, Problem, Slice, SolveStatus, Variable, VariableMatrix, constrain};

/// 3×4 matrix whose element (i, j) is 10i + j.
fn grid() -> VariableMatrix<'static, f64> {
    VariableMatrix::from_fn(3, 4, |row, col| ((10 * row + col) as f64).into())
}

#[test]
fn ranges() {
    let m = grid();

    let v = m.view(1.., 1..3);
    assert_eq!((v.rows(), v.cols()), (2, 2));
    assert_eq!(v.values(), [11.0, 12.0, 21.0, 22.0]);
    assert_eq!(m.view(..=1, 3).values(), [3.0, 13.0]);
    assert_eq!(m.view(.., ..).values(), m.values());
    assert_eq!(m.view(2, ..2).values(), [20.0, 21.0]);
    assert_eq!(m.row(1).values(), [10.0, 11.0, 12.0, 13.0]);
    assert_eq!(m.column(2).values(), [2.0, 12.0, 22.0]);
    assert!(m.view(1..1, ..).is_empty());
}

#[test]
fn strided_slices() {
    let m = grid();

    assert_eq!(m.view(0, Slice::new(..).step(2)).values(), [0.0, 2.0]);
    assert_eq!(
        m.view(0, Slice::new(..).step(-1)).values(),
        [3.0, 2.0, 1.0, 0.0]
    );
    assert_eq!(m.view(0, Slice::new(..).step(-2)).values(), [3.0, 1.0]);
    assert_eq!(m.view(0, Slice::new(..3).step(-2)).values(), [2.0, 0.0]);
    assert_eq!(m.view(0, Slice::new(1..).step(3)).values(), [1.0]);
    assert_eq!(
        m.view(Slice::new(..).step(-1), Slice::new(..).step(2))
            .values(),
        [20.0, 22.0, 10.0, 12.0, 0.0, 2.0]
    );
}

#[test]
fn views_of_views() {
    let m = grid();
    let v = m.view(Slice::new(..).step(-1), 1..);

    // Rows 2, 1, 0 and columns 1, 2, 3; then rows 1.. and every other column
    let w = v.view(1.., Slice::new(..).step(2));
    assert_eq!(w.values(), [11.0, 13.0, 1.0, 3.0]);
    assert_eq!(w.row(1).values(), [1.0, 3.0]);
    assert_eq!(w.column(0).values(), [11.0, 1.0]);
    assert_eq!(w[(1, 1)].value(), 3.0);
    assert_eq!(w[2].value(), 1.0);
}

#[test]
fn view_methods() {
    let m = grid();
    let v = m.view(..2, ..2);

    assert_eq!(v.transpose().values(), [0.0, 10.0, 1.0, 11.0]);
    assert_eq!(v.sum().value(), 22.0);
    assert_eq!(v.map(|x| x * 2.0).values(), [0.0, 2.0, 20.0, 22.0]);
    assert_eq!(v.to_matrix().values(), [0.0, 1.0, 10.0, 11.0]);
    assert_eq!(VariableMatrix::from(v).values(), [0.0, 1.0, 10.0, 11.0]);
    assert_eq!(v.iter().count(), 4);
    assert_eq!((-v).values(), [-0.0, -1.0, -10.0, -11.0]);
    assert_eq!(m.as_view().values(), m.values());
}

#[test]
fn mutable_views() {
    let mut m = VariableMatrix::<f64>::zeros(2, 3);

    m.column_mut(1).copy_from([[1.0], [2.0]]);
    m.row_mut(1).view_mut(.., Slice::new(..).step(2)).fill(5.0);
    assert_eq!(m.values(), [0.0, 1.0, 0.0, 5.0, 2.0, 5.0]);

    let mut v = m.view_mut(.., 1..);
    v[(0, 1)] = 7.0.into();
    v[3] = 8.0.into();
    assert_eq!(v.values(), [1.0, 7.0, 2.0, 8.0]);
    assert_eq!(v.to_matrix().values(), [1.0, 7.0, 2.0, 8.0]);
    assert_eq!(v.as_view().sum().value(), 18.0);
    assert_eq!(m.values(), [0.0, 1.0, 7.0, 5.0, 2.0, 8.0]);

    // Copy from another matrix's view
    let source = grid();
    m.view_mut(.., ..2).copy_from(source.view(1.., ..2));
    assert_eq!(m.values(), [10.0, 11.0, 7.0, 20.0, 21.0, 8.0]);
}

#[test]
fn view_operators() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(2, 2);
    problem.set_initial_matrix(&x, [[1.0, 2.0], [3.0, 4.0]]);
    let s = problem.decision_variable();
    problem.set_initial(s, 2.0);

    let c0 = x.column(0);
    let c1 = x.column(1);
    assert_eq!((c0 + c1).values(), [3.0, 7.0]);
    assert_eq!((&c0 - &x.column(1)).values(), [-1.0, -1.0]);
    assert_eq!((&x * c1).values(), [10.0, 22.0]);
    assert_eq!((x.row(0) * &x).values(), [7.0, 10.0]);
    assert_eq!((c0 + [[1.0], [1.0]]).values(), [2.0, 4.0]);
    assert_eq!(([[1.0, 1.0]] * c0).values(), [4.0]);
    assert_eq!((c0 * 2.0).values(), [2.0, 6.0]);
    assert_eq!((2.0 * c0).values(), [2.0, 6.0]);
    assert_eq!((&c0 * s).values(), [2.0, 6.0]);
    assert_eq!((s * c0).values(), [2.0, 6.0]);
    assert_eq!((c0 / 2.0).values(), [0.5, 1.5]);

    let mut m = x.column(0).to_matrix();
    m += x.column(1);
    m -= &x.column(0);
    assert_eq!(m.values(), [2.0, 4.0]);
}

#[test]
fn view_constraints() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(2, 3);
    problem.set_initial_matrix(&x, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let c = constrain!(x.column(0) == x.column(1));
    let r: Vec<_> = c.residuals().iter().map(Variable::value).collect();
    assert_eq!(r, [-1.0, -1.0]);

    let c = constrain!(0.0 <= x.view(.., 1..) <= &x.view(.., ..2));
    assert_eq!(c.residuals().len(), 8);
}

#[test]
fn set_initial_through_view() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(2, 2);
    problem.set_initial_matrix(x.row(1), [[3.0, 4.0]]);
    assert_eq!(x.values(), [0.0, 0.0, 3.0, 4.0]);
}

#[test]
fn direct_transcription_with_views() {
    // Flywheel: x' = −x + u, discretized with forward Euler; drive x from 0
    // to r = 10 while minimizing tracking error, with |u| ≤ 12
    const N: usize = 50;
    let (dt, r) = (0.05, 10.0);
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(1, N + 1);
    let u = problem.decision_variable_matrix(1, N);

    let a = [[1.0 - dt]];
    let b = [[dt]];
    for k in 0..N {
        problem.subject_to(constrain!(
            x.column(k + 1) == a * x.column(k) + b * u.column(k)
        ));
    }
    problem.subject_to(constrain!(x.column(0) == 0.0));
    problem.subject_to(constrain!(-12.0 <= u <= 12.0));
    let error = x.view(.., 1..) - [[r; N]];
    problem.minimize(error.row(0).iter().map(|&e| e * e).sum());

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));

    // Saturated input until near the reference, then holding u = r
    assert!((u[0].value() - 12.0).abs() < 1e-6, "u₀ = {}", u[0].value());
    assert!((x[N].value() - r).abs() < 0.1, "x_N = {}", x[N].value());
}

#[test]
#[should_panic(expected = "column range 2..5 out of bounds for 4 columns")]
fn range_out_of_bounds_panics() {
    let _ = grid().view(.., 2..5);
}

#[test]
#[should_panic(expected = "row 3 out of bounds for 3 rows")]
fn index_out_of_bounds_panics() {
    let _ = grid().view(3, ..);
}

#[test]
#[should_panic(expected = "slice step can't be zero")]
fn zero_step_panics() {
    let _ = Slice::new(..).step(0);
}

#[test]
#[should_panic(expected = "matrix shape mismatch: 2×1 vs 1×2")]
fn copy_from_shape_mismatch_panics() {
    let mut m = VariableMatrix::<f64>::zeros(2, 2);
    m.column_mut(0).copy_from([[1.0, 2.0]]);
}
