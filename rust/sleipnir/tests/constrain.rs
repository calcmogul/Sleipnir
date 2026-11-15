// Copyright (c) Sleipnir contributors

use sleipnir::{
    Compare, Constraints, EqualityConstraints, InequalityConstraints, Problem, Variable,
    VariableMatrix, constrain,
};

fn values<S: sleipnir::Scalar>(residuals: &[Variable<'_, S>]) -> Vec<S> {
    residuals.iter().map(Variable::value).collect()
}

#[test]
fn variable_relations() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    problem.set_initial(x, 3.0);
    problem.set_initial(y, 1.0);

    let c: EqualityConstraints<'_, f64> = constrain!(x == y);
    assert_eq!(values(c.residuals()), [2.0]);

    // c(x) ≥ 0 form: lhs − rhs for >=, rhs − lhs for <=
    let c: InequalityConstraints<'_, f64> = constrain!(x >= y);
    assert_eq!(values(c.residuals()), [2.0]);
    let c: InequalityConstraints<'_, f64> = constrain!(x <= y);
    assert_eq!(values(c.residuals()), [-2.0]);
}

#[test]
fn scalar_operands() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.set_initial(x, 3.0);

    assert_eq!(values(constrain!(x >= 1.0).residuals()), [2.0]);
    assert_eq!(values(constrain!(1.0 >= x).residuals()), [-2.0]);
    assert_eq!(values(constrain!(x == 1.0).residuals()), [2.0]);
    assert_eq!(values(constrain!(1.0 == x).residuals()), [-2.0]);

    let bound = 5.0;
    assert_eq!(values(constrain!(x <= bound).residuals()), [2.0]);
}

#[test]
fn precedence() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    problem.set_initial(x, 1.0);
    problem.set_initial(y, 2.0);

    // (x + 2y) − (1 − y) = 5 − (−1) = 6
    let c = constrain!(x + y * 2.0 >= 1.0 - y);
    assert_eq!(values(c.residuals()), [6.0]);
}

#[test]
fn negative_operands_with_and_without_spaces() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.set_initial(x, 3.0);

    assert_eq!(values(constrain!(x <= -1.0).residuals()), [-4.0]);
    #[rustfmt::skip]
    let c = constrain!(x<=-1.0);
    assert_eq!(values(c.residuals()), [-4.0]);
    #[rustfmt::skip]
    let c = constrain!(x>=-x);
    assert_eq!(values(c.residuals()), [6.0]);
    #[rustfmt::skip]
    let c = constrain!(x==-1.0);
    assert_eq!(values(c.residuals()), [4.0]);
}

#[test]
#[allow(clippy::redundant_closure_call, clippy::unnecessary_cast)]
fn generic_and_nested_operands() {
    fn double<T: std::ops::Mul<f64, Output = T>>(x: T) -> T {
        x * 2.0
    }

    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.set_initial(x, 3.0);

    assert_eq!(
        values(constrain!(double::<Variable<'_, f64>>(x) >= 1.0).residuals()),
        [5.0]
    );
    assert_eq!(
        values(constrain!((|v: f64| v * 2.0)(1.0) <= x).residuals()),
        [1.0]
    );
    assert_eq!(
        values(constrain!(x >= if 1 < 2 { 1.0 } else { 2.0 }).residuals()),
        [2.0]
    );
    assert_eq!(
        values(constrain!(x >= [1.0, 4.0][(1 >= 0) as usize]).residuals()),
        [-1.0]
    );
    assert_eq!(values(constrain!(x >= 2.0 as f64).residuals()), [1.0]);
}

#[test]
fn chained_inequalities() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.set_initial(x, 3.0);

    let c = constrain!(-1.0 <= x <= 5.0);
    assert_eq!(values(c.residuals()), [4.0, 2.0]);

    let c = constrain!(5.0 >= x >= -1.0);
    assert_eq!(values(c.residuals()), [2.0, 4.0]);

    let c = constrain!(0.0 <= x <= x * 2.0 <= 10.0);
    assert_eq!(values(c.residuals()), [3.0, 3.0, 4.0]);
}

#[test]
fn chain_evaluates_each_operand_once() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();

    let mut evaluations = 0;
    let _ = constrain!(0.0 <= {
        evaluations += 1;
        x
    } <= 1.0);
    assert_eq!(evaluations, 1);
}

#[test]
fn chain_bindings_are_hygienic() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.set_initial(x, 3.0);

    #[allow(non_upper_case_globals)]
    let __sleipnir_operand_0 = 10.0;
    let c = constrain!(0.0 <= x <= __sleipnir_operand_0);
    assert_eq!(values(c.residuals()), [3.0, 7.0]);
}

#[test]
fn matrix_operands() {
    let problem: Problem = Problem::new();
    let y = problem.decision_variable_matrix(2, 1);
    let z = problem.decision_variable();
    problem.set_initial(y[0], 1.0);
    problem.set_initial(y[1], 2.0);
    problem.set_initial(z, 5.0);

    assert_eq!(values(constrain!(y >= 0.5).residuals()), [0.5, 1.5]);
    assert_eq!(values(constrain!(0.5 >= y).residuals()), [-0.5, -1.5]);
    assert_eq!(values(constrain!(y <= z).residuals()), [4.0, 3.0]);
    assert_eq!(values(constrain!(z == y).residuals()), [4.0, 3.0]);
    assert_eq!(values(constrain!(y == y).residuals()), [0.0, 0.0]);
    assert_eq!(
        values(constrain!(-1.0 <= y <= z).residuals()),
        [2.0, 3.0, 4.0, 3.0]
    );

    // Matrix product: [[1, 2]] * y = 1 + 4 = 5
    let a = VariableMatrix::from_fn(1, 2, |_, col| y[col] / y[col] * (col as f64 + 1.0));
    assert_eq!(values(constrain!(&a * &y == z).residuals()), [0.0]);

    // Matrices aren't moved
    let _ = &y;
}

#[test]
fn empty_matrix_with_scalar() {
    let problem: Problem = Problem::new();
    let y = problem.decision_variable_matrix(0, 1);
    assert!(constrain!(y >= 0.0).residuals().is_empty());
}

#[test]
fn methods_match_macro() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    problem.set_initial(x, 3.0);

    assert_eq!(
        values(x.geq(&1.0).residuals()),
        values(constrain!(x >= 1.0).residuals())
    );
    assert_eq!(
        values(x.leq(&1.0).residuals()),
        values(constrain!(x <= 1.0).residuals())
    );
    assert_eq!(
        values(x.equals(&1.0).residuals()),
        values(constrain!(x == 1.0).residuals())
    );
}

#[test]
fn subject_to_sorts_constraints() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable_matrix(3, 1);

    problem.minimize(x * x);
    problem.subject_to(constrain!(x == 1.0));
    problem.subject_to(constrain!(y >= 0.0));
    problem.subject_to(constrain!(-1.0 <= x <= 1.0));
    problem.subject_to(Constraints::from(x.equals(&y[0])));

    assert!(problem.cost().is_some());
    assert_eq!(problem.decision_variables().len(), 4);
    assert_eq!(problem.equality_constraints().len(), 2);
    assert_eq!(problem.inequality_constraints().len(), 5);
}

#[test]
fn f32_problem() {
    let problem = Problem::<f32>::new();
    let x = problem.decision_variable();
    problem.set_initial(x, 3.0);
    assert_eq!(
        values(constrain!(0.0 <= x <= 1.0).residuals()),
        [3.0_f32, -2.0]
    );
}

#[test]
fn symmetric_decision_variable() {
    let problem: Problem = Problem::new();
    let p = problem.symmetric_decision_variable(3);
    assert_eq!(problem.decision_variables().len(), 6);
    for row in 0..3 {
        for col in 0..3 {
            problem.set_initial(p[(row, col)], (10 * row.max(col) + row.min(col)) as f64);
        }
    }
    assert_eq!(
        p.values(),
        [0.0, 10.0, 20.0, 10.0, 11.0, 21.0, 20.0, 21.0, 22.0]
    );
}

#[test]
#[should_panic(expected = "constraint shape mismatch: 2×1 vs 3×1")]
fn shape_mismatch_panics() {
    let problem: Problem = Problem::new();
    let y = problem.decision_variable_matrix(2, 1);
    let z = problem.decision_variable_matrix(3, 1);
    let _ = constrain!(y == z);
}

#[test]
#[should_panic(expected = "variables belong to different problems")]
fn mixing_problems_panics() {
    let a: Problem = Problem::new();
    let b: Problem = Problem::new();
    let _ = constrain!(a.decision_variable() == b.decision_variable());
}

#[test]
#[should_panic(expected = "variable belongs to a different problem")]
fn subject_to_other_problem_panics() {
    let a: Problem = Problem::new();
    let b: Problem = Problem::new();
    let x = a.decision_variable();
    b.subject_to(constrain!(x >= 0.0));
}
