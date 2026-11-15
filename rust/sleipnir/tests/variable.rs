// Copyright (c) Sleipnir contributors

use sleipnir::{Problem, Variable, VariableMatrix, constrain};

#[test]
fn constants_need_no_problem() {
    let two = Variable::<f64>::from(2.0);
    let expr = (two * 3.0 + 1.0).sqrt() - two.powf(2.0) / 4.0;

    assert!(expr.is_constant());
    assert_eq!(expr.value(), 7.0_f64.sqrt() - 1.0);
    assert_eq!(format!("{two:?}"), "Variable::Constant(2.0)");
}

#[test]
fn constants_join_any_problem() {
    let c = Variable::from(2.0);
    let a: Problem = Problem::new();
    let b: Problem = Problem::new();
    let x = a.decision_variable();
    let y = b.decision_variable();
    a.set_initial(x, 1.0);
    b.set_initial(y, 5.0);

    let sum_a = x + c;
    let sum_b = c * y;
    assert!(!sum_a.is_constant());
    assert_eq!(sum_a.value(), 3.0);
    assert_eq!(sum_b.value(), 10.0);
}

#[test]
fn constant_constraints_and_cost() {
    let problem: Problem = Problem::new();
    problem.minimize(Variable::from(1.0));
    problem.subject_to(constrain!(Variable::from(3.0) >= 2.0));

    assert_eq!(problem.cost().unwrap().value(), 1.0);
    let constraints = problem.inequality_constraints();
    assert_eq!(constraints.len(), 1);
    assert!(!constraints[0].is_constant());
    assert_eq!(constraints[0].value(), 1.0);
}

#[test]
#[should_panic(expected = "constants aren't decision variables")]
fn set_initial_on_constant_panics() {
    let problem: Problem = Problem::new();
    problem.set_initial(Variable::from(1.0), 2.0);
}

#[test]
#[should_panic(expected = "only decision variables and parameters have settable values")]
fn set_initial_on_expression_panics() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.set_initial(x * 2.0, 2.0);
}

#[test]
fn matrix_factories() {
    let m = VariableMatrix::<f64>::zeros(2, 3);
    assert_eq!((m.rows(), m.cols()), (2, 3));
    assert_eq!(m.values(), [0.0; 6]);
    assert!(m.iter().all(Variable::is_constant));

    assert_eq!(VariableMatrix::<f64>::ones(1, 2).values(), [1.0, 1.0]);
    assert_eq!(
        VariableMatrix::<f64>::identity(2).values(),
        [1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(
        VariableMatrix::<f64>::from_element(2, 1, 4.0).values(),
        [4.0, 4.0]
    );
    let m = VariableMatrix::<f64>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(m[(1, 0)].value(), 3.0);
}

#[test]
fn matrix_conversions() {
    let m = VariableMatrix::<f64>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!((m.rows(), m.cols()), (2, 3));
    assert_eq!(m[(1, 2)].value(), 6.0);

    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.set_initial(x, 7.0);

    // Mixed scalars and variables
    let m = VariableMatrix::from([[x, Variable::from(1.0)]]);
    assert_eq!(m.values(), [7.0, 1.0]);

    let v: VariableMatrix<'_, f64> = [1.0, 2.0, 3.0].into_iter().collect();
    assert_eq!((v.rows(), v.cols()), (3, 1));

    let s = VariableMatrix::from(x);
    assert_eq!((s.rows(), s.cols()), (1, 1));
}

#[test]
fn constant_matrix_arithmetic() {
    let a = VariableMatrix::<f64>::from([[1.0, 2.0], [3.0, 4.0]]);
    let b = VariableMatrix::from([[5.0], [6.0]]);
    let ab = &a * &b;
    assert_eq!(ab.values(), [17.0, 39.0]);
    assert!(ab.iter().all(Variable::is_constant));

    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(2, 1);
    problem.set_initial(x[0], 1.0);
    problem.set_initial(x[1], -1.0);
    assert_eq!((&a * &x - &b).values(), [-6.0, -7.0]);
}

#[test]
fn empty_inner_dimension_product_is_zero() {
    let a = VariableMatrix::<f64>::zeros(2, 0);
    let b = VariableMatrix::zeros(0, 3);
    assert_eq!((&a * &b).values(), [0.0; 6]);
}

#[test]
fn block() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(2, 1);
    let m = VariableMatrix::block(&[
        &[&VariableMatrix::identity(2), &x],
        &[&VariableMatrix::zeros(1, 2), &VariableMatrix::ones(1, 1)],
    ]);
    assert_eq!((m.rows(), m.cols()), (3, 3));
    assert_eq!(m.values(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(!m[(1, 2)].is_constant());

    assert_eq!(VariableMatrix::<f64>::block(&[]).values(), []);
}

#[test]
#[should_panic(expected = "blocks in block row 0 have different row counts")]
fn block_row_count_mismatch_panics() {
    let _ = VariableMatrix::<f64>::block(&[&[
        &VariableMatrix::zeros(1, 1),
        &VariableMatrix::zeros(2, 1),
    ]]);
}

#[test]
#[should_panic(expected = "block row 1 has a different column count than block row 0")]
fn block_column_count_mismatch_panics() {
    let _ = VariableMatrix::<f64>::block(&[
        &[&VariableMatrix::zeros(1, 2)],
        &[&VariableMatrix::zeros(1, 3)],
    ]);
}

#[test]
fn constraints_against_constant_matrices() {
    let problem: Problem = Problem::new();
    let y = problem.decision_variable_matrix(2, 1);
    problem.set_initial(y[0], 1.0);
    problem.set_initial(y[1], 2.0);

    let c = constrain!(VariableMatrix::from([[0.5], [3.0]]) <= y);
    let residuals: Vec<_> = c.residuals().iter().map(Variable::value).collect();
    assert_eq!(residuals, [0.5, -1.0]);
}
