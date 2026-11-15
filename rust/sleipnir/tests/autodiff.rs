// Copyright (c) Sleipnir contributors

use sleipnir::{
    Condition, ExpressionType, Gradient, Hessian, Jacobian, Options, Problem, SolveStatus,
    Variable, VariableMatrix, constrain, if_else,
};

/// Returns every scalar function of (x, y) the expression graph supports.
fn functions<'p>(
    x: Variable<'p, f64>,
    y: Variable<'p, f64>,
) -> Vec<(&'static str, Variable<'p, f64>)> {
    vec![
        ("neg", -x),
        ("abs", x.abs()),
        ("acos", x.acos()),
        ("asin", x.asin()),
        ("atan", x.atan()),
        ("cbrt", x.cbrt()),
        ("cos", x.cos()),
        ("cosh", x.cosh()),
        ("erf", x.erf()),
        ("exp", x.exp()),
        ("ln", y.ln()),
        ("log10", y.log10()),
        ("sign", x.sign()),
        ("sin", x.sin()),
        ("sinh", x.sinh()),
        ("sqrt", y.sqrt()),
        ("tan", x.tan()),
        ("tanh", x.tanh()),
        ("add", x + y),
        ("sub", x - y),
        ("mul", x * y),
        ("div", x / y),
        ("pow", y.pow(x)),
        ("powf", y.powf(1.5)),
        ("atan2", x.atan2(y)),
        ("hypot", x.hypot(y)),
        ("max", x.max(y)),
        ("min", x.min(y)),
        ("if_else", if_else(Condition::Less, x, y, x * y, x + y)),
        ("composite", (x * y).sin() * (x - y).exp() / y.sqrt()),
    ]
}

#[test]
fn gradients_match_finite_differences() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    let (x0, y0) = (0.3, 1.7);
    let h = 1e-6;
    let wrt = VariableMatrix::from([[x], [y]]);

    for (name, f) in functions(x, y) {
        let gradient = Gradient::new(f, &wrt);
        problem.set_initial(x, x0);
        problem.set_initial(y, y0);
        let value = gradient.value();
        let analytic = [value.get_entry(0, 0), value.get_entry(1, 0)]
            .map(|e| e.map_or(0.0, |e| e.into_value()));

        for (i, var) in [x, y].into_iter().enumerate() {
            let at = |dx: f64| {
                problem.set_initial(x, x0);
                problem.set_initial(y, y0);
                problem.set_initial(var, [x0, y0][i] + dx);
                f.value()
            };
            let numeric = (at(h) - at(-h)) / (2.0 * h);
            assert!(
                (analytic[i] - numeric).abs() <= 1e-6 * (1.0 + numeric.abs()),
                "∂{name}/∂{}: analytic {} vs numeric {numeric}",
                ["x", "y"][i],
                analytic[i]
            );
        }

        // get() evaluates to the same values
        problem.set_initial(x, x0);
        problem.set_initial(y, y0);
        assert_eq!(gradient.get().values(), analytic, "{name}");
    }
}

#[test]
fn gradient_sparsity() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(3, 1);

    // f only depends on x₀ and x₂, and ∂sign/∂x is structurally zero
    let f = x[0] * x[2] + x[1].sign();
    let g = Gradient::new(f, &x);
    assert_eq!(g.value().nnz(), 2);
    assert_eq!(g.get()[1].value(), 0.0);
    assert!(g.get()[1].is_constant());

    // A constant has no gradient
    let g = Gradient::new(Variable::from(3.0), &x);
    assert_eq!(g.value().nnz(), 0);
    assert_eq!((g.get().rows(), g.get().cols()), (3, 1));

    // Single-variable wrt
    let g = Gradient::new(x[0] * x[0], x[0]);
    problem.set_initial(x[0], 3.0);
    assert_eq!(g.get().values(), [6.0]);
}

#[test]
fn jacobian() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(3, 1);
    problem.set_initial_matrix(&x, [[1.0], [2.0], [3.0]]);

    let f = VariableMatrix::from([[x[0] * x[1]], [x[2].powf(2.0)], [x[0] + 2.0 * x[1]]]);
    let j = Jacobian::new(&f, &x);
    assert_eq!(
        j.get().values(),
        [2.0, 1.0, 0.0, 0.0, 0.0, 6.0, 1.0, 2.0, 0.0]
    );
    let value = j.value();
    assert_eq!((value.nrows(), value.ncols(), value.nnz()), (3, 3, 5));
}

#[test]
fn hessian() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(2, 1);
    problem.set_initial_matrix(&x, [[1.0], [2.0]]);

    // Rosenbrock: H = [1200x₀² − 400x₁ + 2, −400x₀; −400x₀, 200]
    let f = 100.0 * (x[1] - x[0] * x[0]).powf(2.0) + (1.0 - x[0]).powf(2.0);
    let expected = [402.0, -400.0, -400.0, 200.0];

    let h = Hessian::new(f, &x);
    assert_eq!(h.get().values(), expected);
    let value = h.value();
    assert_eq!(value.nnz(), 4);
    assert_eq!(value.get_entry(0, 1).unwrap().into_value(), -400.0);

    let lower = Hessian::lower(f, &x).value();
    assert_eq!(lower.nnz(), 3);
    assert!(lower.get_entry(0, 1).unwrap().into_value() == 0.0);
    assert_eq!(lower.get_entry(1, 0).unwrap().into_value(), -400.0);

    // Values follow the decision variables
    problem.set_initial_matrix(&x, [[0.0], [0.0]]);
    assert_eq!(h.get().values(), [2.0, 0.0, 0.0, 200.0]);
}

#[test]
fn derivatives_in_problems() {
    // Find the stationary point of a convex quadratic by constraining ∇f = 0:
    // 2(x₀ − 3) + x₁ = 0 and 4(x₁ + 1) + x₀ = 0, so x = (4, −2)
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(2, 1);
    let f = (x[0] - 3.0).powf(2.0) + 2.0 * (x[1] + 1.0).powf(2.0) + x[0] * x[1];
    let g = Gradient::new(f, &x).get();
    assert!(
        g.iter()
            .all(|g| g.expression_type() == ExpressionType::Linear)
    );

    problem.subject_to(constrain!(g == 0.0));
    assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
    let x = x.values();
    assert!(
        (x[0] - 4.0).abs() < 1e-8 && (x[1] + 2.0).abs() < 1e-8,
        "{x:?}"
    );
}

#[test]
#[should_panic(expected = "wrt must be a column vector, got 1×2")]
fn wrt_must_be_column_vector() {
    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(1, 2);
    let _ = Gradient::new(x[0], &x);
}

#[test]
#[should_panic(expected = "variables belong to different problems")]
fn wrt_from_other_problem_panics() {
    let a: Problem = Problem::new();
    let b: Problem = Problem::new();
    let x = a.decision_variable();
    let y = b.decision_variable();
    let _ = Gradient::new(x * x, y);
}
