// Copyright (c) Sleipnir contributors

// The tests exercise every owned and borrowed operand combination
#![allow(clippy::op_ref)]

use sleipnir::{Options, Problem, SolveStatus, Variable, VariableMatrix, constrain};

fn assert_near(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= 1e-6,
            "expected {expected:?}, got {actual:?}"
        );
    }
}

/// Returns a problem's 2×1 decision variable with initial value (1, 2).
fn vector(problem: &Problem) -> VariableMatrix<'_, f64> {
    let x = problem.decision_variable_matrix(2, 1);
    problem.set_initial_matrix(&x, [[1.0], [2.0]]);
    x
}

#[test]
fn matrix_matrix_ops() {
    let problem: Problem = Problem::new();
    let x = vector(&problem);
    let a = VariableMatrix::from([[1.0, 2.0], [3.0, 4.0]]);

    assert_eq!((&a * &x).values(), [5.0, 11.0]);
    assert_eq!((&x + &x).values(), [2.0, 4.0]);
    assert_eq!((&x - &x).values(), [0.0, 0.0]);
    assert_eq!((a.clone() * x.clone()).values(), [5.0, 11.0]);
    assert_eq!((&a * x.clone()).values(), [5.0, 11.0]);
    assert_eq!((a * &x).values(), [5.0, 11.0]);
}

#[test]
fn array_operands() {
    let problem: Problem = Problem::new();
    let x = vector(&problem);

    assert_eq!((&x + [[10.0], [20.0]]).values(), [11.0, 22.0]);
    assert_eq!((&x - &[[10.0], [20.0]]).values(), [-9.0, -18.0]);
    assert_eq!(([[10.0], [20.0]] - &x).values(), [9.0, 18.0]);
    assert_eq!(([[1.0, 2.0], [3.0, 4.0]] * &x).values(), [5.0, 11.0]);
    assert_eq!((x.transpose() * [[1.0], [1.0]]).values(), [3.0]);
    assert_eq!((&[[2.0, 0.0], [0.0, 2.0]] * x.clone()).values(), [2.0, 4.0]);

    // Arrays of variables
    let y = [[x[0], x[1]]];
    assert_eq!((y * &x).values(), [5.0]);
}

#[test]
fn scalar_and_variable_factors() {
    let problem: Problem = Problem::new();
    let x = vector(&problem);
    let s = problem.decision_variable();
    problem.set_initial(s, 3.0);

    assert_eq!((&x * 2.0).values(), [2.0, 4.0]);
    assert_eq!((2.0 * &x).values(), [2.0, 4.0]);
    assert_eq!((x.clone() * 2.0).values(), [2.0, 4.0]);
    assert_eq!((2.0 * x.clone()).values(), [2.0, 4.0]);
    assert_eq!((&x / 2.0).values(), [0.5, 1.0]);
    assert_eq!((&x * s).values(), [3.0, 6.0]);
    assert_eq!((s * &x).values(), [3.0, 6.0]);
    assert_eq!((&x / s).values(), [1.0 / 3.0, 2.0 / 3.0]);

    let f = VariableMatrix::<f32>::ones(1, 2);
    assert_eq!((2.0_f32 * &f).values(), [2.0, 2.0]);
}

#[test]
fn assignment_ops() {
    let problem: Problem = Problem::new();
    let x = vector(&problem);
    let s = problem.decision_variable();
    problem.set_initial(s, 4.0);

    let mut m = x.clone();
    m *= 3.0;
    assert_eq!(m.values(), [3.0, 6.0]);
    m /= 3.0;
    assert_eq!(m.values(), [1.0, 2.0]);
    m *= s;
    assert_eq!(m.values(), [4.0, 8.0]);
    m /= s;
    m += &x;
    assert_eq!(m.values(), [2.0, 4.0]);
    m -= [[1.0], [1.0]];
    assert_eq!(m.values(), [1.0, 3.0]);
    m += x.clone();
    assert_eq!(m.values(), [2.0, 5.0]);

    let mut a = VariableMatrix::from([[1.0, 2.0], [3.0, 4.0]]);
    a *= &x;
    assert_eq!(a.values(), [5.0, 11.0]);
}

#[test]
#[should_panic(expected = "matrix shape mismatch: 2×1 vs 1×2")]
fn elementwise_shape_mismatch_panics() {
    let problem: Problem = Problem::new();
    let x = vector(&problem);
    let _ = &x + [[1.0, 2.0]];
}

#[test]
#[should_panic(expected = "matrix shape mismatch: 2×1 * 2×1")]
fn product_shape_mismatch_panics() {
    let problem: Problem = Problem::new();
    let x = vector(&problem);
    let _ = &x * &x;
}

#[test]
fn matrix_methods() {
    let problem: Problem = Problem::new();
    let x = vector(&problem);
    let m = VariableMatrix::<f64>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let t = m.transpose();
    assert_eq!((t.rows(), t.cols()), (3, 2));
    assert_eq!(t.values(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(m.row(1).values(), [4.0, 5.0, 6.0]);
    assert_eq!(m.column(2).values(), [3.0, 6.0]);
    assert_eq!((m.len(), m.is_empty()), (6, false));
    assert!(VariableMatrix::<f64>::zeros(0, 3).is_empty());

    assert_eq!(m.sum().value(), 21.0);
    assert_eq!(VariableMatrix::<f64>::zeros(0, 0).sum().value(), 0.0);
    assert_eq!(x.dot(&x).value(), 5.0);
    assert_eq!(m.map(|e| e * e).values(), [1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    assert_eq!(
        x.map(Variable::exp).values(),
        [1.0_f64.exp(), 2.0_f64.exp()]
    );
    assert_eq!(x.component_mul(&x).values(), [1.0, 4.0]);
    assert_eq!(x.component_div(&x).values(), [1.0, 1.0]);
    assert_eq!(
        m.zip_map(&m, |a, b| a.max(b) - 1.0).values(),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn iteration() {
    let problem: Problem = Problem::new();
    let x = vector(&problem);

    let mut total = 0.0;
    for e in &x {
        total += e.value();
    }
    assert_eq!(total, 3.0);

    let product: Variable<'_, f64> = x.iter().product();
    assert_eq!(product.value(), 2.0);
    let sum: Variable<'_, f64> = x.clone().into_iter().sum();
    assert_eq!(sum.value(), 3.0);
}

#[test]
#[should_panic(expected = "row 2 out of bounds for 2 rows")]
fn row_out_of_bounds_panics() {
    let _ = VariableMatrix::<f64>::zeros(2, 2).row(2);
}

#[test]
fn constraints_with_arrays() {
    let problem: Problem = Problem::new();
    let x = vector(&problem);
    let a = [[1.0, 2.0], [3.0, 4.0]];

    let c = constrain!(&x == [[0.0], [5.0]]);
    let r: Vec<_> = c.residuals().iter().map(Variable::value).collect();
    assert_eq!(r, [1.0, -3.0]);

    let c = constrain!([[0.0], [0.0]] <= x <= [[2.0], [2.0]]);
    let r: Vec<_> = c.residuals().iter().map(Variable::value).collect();
    assert_eq!(r, [1.0, 2.0, 1.0, 0.0]);

    let c = constrain!(a * &x >= [[5.0], [10.0]]);
    let r: Vec<_> = c.residuals().iter().map(Variable::value).collect();
    assert_eq!(r, [0.0, 1.0]);
}

#[test]
fn set_initial_matrix() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(1, 2);
    problem.set_initial_matrix(&x, [[3.0, 4.0]]);
    assert_eq!(x.values(), [3.0, 4.0]);
}

#[test]
#[should_panic(expected = "matrix shape mismatch: 1×2 vs 2×1")]
fn set_initial_matrix_shape_mismatch_panics() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(1, 2);
    problem.set_initial_matrix(&x, [[3.0], [4.0]]);
}

#[test]
fn solve_matrix_problem() {
    // min ½xᵀQx + cᵀx  s.t.  Ax = b, x ≥ 0
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(3, 1);
    let q = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
    let c = [[-2.0], [-5.0], [1.0]];

    problem.minimize(0.5 * (x.transpose() * q * &x)[0] + (x.transpose() * c)[0]);
    problem.subject_to(constrain!([[1.0, 1.0, 1.0]] * &x == [[1.0]]));
    problem.subject_to(constrain!(&x >= 0.0));

    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));

    // KKT: x = (0, 1, 0) with ν = −3 and bound multipliers μ₀ = 1, μ₂ = 4
    assert_near(&x.values(), &[0.0, 1.0, 0.0]);
}

mod nalgebra_interop {
    use super::*;
    use nalgebra::{DMatrix, Matrix2, Vector2};

    #[test]
    fn operators() {
        let problem: Problem = Problem::new();
        let x = super::vector(&problem);
        let a = Matrix2::new(1.0, 2.0, 3.0, 4.0);

        assert_eq!((a * &x).values(), [5.0, 11.0]);
        assert_eq!((&a * x.clone()).values(), [5.0, 11.0]);
        assert_eq!((&x + Vector2::new(10.0, 20.0)).values(), [11.0, 22.0]);
        assert_eq!((Vector2::new(10.0, 20.0) - &x).values(), [9.0, 18.0]);
        assert_eq!((x.transpose() * &a).values(), [7.0, 10.0]);

        let d = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        assert_eq!((&d * &x).values(), [3.0]);

        let mut m = x.clone();
        m += Vector2::new(1.0, 1.0);
        assert_eq!(m.values(), [2.0, 3.0]);
    }

    #[test]
    fn conversions() {
        let m = VariableMatrix::<f64>::from(&Matrix2::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!((m.rows(), m.cols()), (2, 2));
        assert_eq!(
            m.value_matrix(),
            DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0])
        );

        let problem: Problem = Problem::new();
        let x = problem.decision_variable_matrix(2, 1);
        problem.set_initial_matrix(&x, &Vector2::new(5.0, 6.0));
        assert_eq!(x.values(), [5.0, 6.0]);
    }

    #[test]
    fn constraints_and_solve() {
        let problem: Problem = Problem::new();
        let x = problem.decision_variable_matrix(2, 1);
        let a = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        let b = Vector2::new(5.0, 6.0);

        problem.minimize(x.dot(&x));
        problem.subject_to(constrain!(a * &x == b));

        assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
        let expected = a.lu().solve(&b).unwrap();
        assert_near(&x.values(), expected.as_slice());
        assert!(
            (x.value_matrix() - DMatrix::from_column_slice(2, 1, expected.as_slice())).norm()
                < 1e-6
        );
    }
}
