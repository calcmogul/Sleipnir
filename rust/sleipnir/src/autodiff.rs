// Copyright (c) Sleipnir contributors

//! Derivatives of expressions: [`Gradient`], [`Jacobian`], and [`Hessian`].
//!
//! Derivatives are built symbolically with reverse mode autodiff over the
//! expression graph, so [`get`](Gradient::get) returns them as expressions
//! (useful in problems with derivatives in them), and
//! [`value`](Gradient::value) evaluates them at the current decision variable
//! values. Entries whose derivative is structurally zero aren't stored.

use nalgebra_sparse::CscMatrix;

use crate::variable::{Arena, Binary, Node, Repr, Unary};
use crate::{Condition, Scalar, Variable, VariableMatrix, if_else};

/// Returns the partial derivatives of node `id` with respect to its operands,
/// skipping partials that are constant zero.
fn partials<'p, S: Scalar>(
    arena: &'p Arena<S>,
    id: usize,
    node: Node<S>,
) -> Vec<(usize, Variable<'p, S>)> {
    let v = |id: usize| Variable(Repr::Node(arena, id));
    let c = |x: f64| Variable::<'p, S>::from(S::from_f64(x));
    let out = v(id);

    let partials = match node {
        Node::Decision(_) | Node::Parameter(_) | Node::Constant(_) => vec![],
        Node::Unary(op, a) => {
            let x = v(a);
            let partial = match op {
                Unary::Neg => c(-1.0),
                Unary::Abs => x.sign(),
                Unary::Acos => c(-1.0) / (c(1.0) - x * x).sqrt(),
                Unary::Asin => c(1.0) / (c(1.0) - x * x).sqrt(),
                Unary::Atan => c(1.0) / (c(1.0) + x * x),
                Unary::Cbrt => c(1.0) / (c(3.0) * out * out),
                Unary::Cos => -x.sin(),
                Unary::Cosh => x.sinh(),
                Unary::Erf => c(std::f64::consts::FRAC_2_SQRT_PI) * (-(x * x)).exp(),
                Unary::Exp => out,
                Unary::Ln => c(1.0) / x,
                Unary::Log10 => c(1.0) / (c(std::f64::consts::LN_10) * x),
                Unary::Sign => c(0.0),
                Unary::Sin => x.cos(),
                Unary::Sinh => x.cosh(),
                Unary::Sqrt => c(0.5) / out,
                Unary::Tan => {
                    let cos = x.cos();
                    c(1.0) / (cos * cos)
                }
                Unary::Tanh => {
                    let cosh = x.cosh();
                    c(1.0) / (cosh * cosh)
                }
            };
            vec![(a, partial)]
        }
        Node::Binary(op, a, b) => {
            let (x, y) = (v(a), v(b));
            match op {
                Binary::Add => vec![(a, c(1.0)), (b, c(1.0))],
                Binary::Sub => vec![(a, c(1.0)), (b, c(-1.0))],
                Binary::Mul => vec![(a, y), (b, x)],
                Binary::Div => vec![(a, c(1.0) / y), (b, -x / (y * y))],
                Binary::Pow => vec![(a, y * x.pow(y - c(1.0))), (b, out * x.ln())],
                Binary::Atan2 => {
                    let r2 = x * x + y * y;
                    vec![(a, y / r2), (b, -x / r2)]
                }
                Binary::Hypot => vec![(a, x / out), (b, y / out)],
                Binary::Max => vec![
                    (a, if_else(Condition::GreaterEqual, x, y, c(1.0), c(0.0))),
                    (b, if_else(Condition::GreaterEqual, x, y, c(0.0), c(1.0))),
                ],
                Binary::Min => vec![
                    (a, if_else(Condition::LessEqual, x, y, c(1.0), c(0.0))),
                    (b, if_else(Condition::LessEqual, x, y, c(0.0), c(1.0))),
                ],
            }
        }
        Node::IfElse(condition, [a, b, t, f]) => {
            let (x, y) = (v(a), v(b));
            vec![
                (t, if_else(condition, x, y, c(1.0), c(0.0))),
                (f, if_else(condition, x, y, c(0.0), c(1.0))),
            ]
        }
    };
    partials
        .into_iter()
        .filter(|(_, partial)| partial.is_constant().then(|| partial.value()) != Some(S::ZERO))
        .collect()
}

/// Returns ∂f/∂wrtᵢ for each element of `wrt`, or `None` where the
/// derivative is structurally zero.
///
/// # Panics
///
/// Panics if `f` and `wrt` belong to different problems.
#[track_caller]
fn gradient_expressions<'p, S: Scalar>(
    f: Variable<'p, S>,
    wrt: &[Variable<'p, S>],
) -> Vec<Option<Variable<'p, S>>> {
    let Repr::Node(arena, root) = f.0 else {
        return vec![None; wrt.len()];
    };

    // Accumulate adjoints from the root toward the leaves. Partials append
    // nodes, so iterate over a snapshot of the root's ancestors.
    let nodes: Vec<Node<S>> = arena.nodes()[..=root].to_vec();
    let mut adjoints: Vec<Option<Variable<'p, S>>> = vec![None; root + 1];
    adjoints[root] = Some(S::ONE.into());
    for id in (0..=root).rev() {
        let Some(adjoint) = adjoints[id] else {
            continue;
        };
        for (operand, partial) in partials(arena, id, nodes[id]) {
            let term = adjoint * partial;
            adjoints[operand] = Some(match adjoints[operand] {
                Some(sum) => sum + term,
                None => term,
            });
        }
    }

    wrt.iter()
        .map(|w| match w.0 {
            Repr::Constant(_) => None,
            Repr::Node(owner, id) => {
                assert!(
                    std::ptr::eq(owner, arena),
                    "variables belong to different problems"
                );
                adjoints.get(id).copied().flatten()
            }
        })
        .collect()
}

/// Rows of derivative expressions with their structural nonzeros.
struct Derivatives<'p, S: Scalar> {
    rows: usize,
    cols: usize,
    /// (row, column, expression) for each structural nonzero.
    entries: Vec<(usize, usize, Variable<'p, S>)>,
}

impl<'p, S: Scalar> Derivatives<'p, S> {
    #[track_caller]
    fn new(functions: &[Variable<'p, S>], wrt: &[Variable<'p, S>]) -> Self {
        let entries = functions
            .iter()
            .enumerate()
            .flat_map(|(row, &f)| {
                gradient_expressions(f, wrt)
                    .into_iter()
                    .enumerate()
                    .filter_map(move |(col, d)| d.map(|d| (row, col, d)))
            })
            .collect();
        Self {
            rows: functions.len(),
            cols: wrt.len(),
            entries,
        }
    }

    fn get(&self) -> VariableMatrix<'p, S> {
        let mut result = VariableMatrix::zeros(self.rows, self.cols);
        for &(row, col, d) in &self.entries {
            result[(row, col)] = d;
        }
        result
    }

    fn value(&self, keep: impl Fn(usize, usize) -> bool) -> CscMatrix<S> {
        let mut entries: Vec<_> = self
            .entries
            .iter()
            .filter(|&&(row, col, _)| keep(row, col))
            .collect();
        entries.sort_by_key(|&&(row, col, _)| (col, row));

        let mut col_offsets = vec![0; self.cols + 1];
        for &&(_, col, _) in &entries {
            col_offsets[col + 1] += 1;
        }
        for col in 0..self.cols {
            col_offsets[col + 1] += col_offsets[col];
        }
        let row_indices = entries.iter().map(|&&(row, _, _)| row).collect();
        let values = entries.iter().map(|&&(_, _, d)| d.value()).collect();
        CscMatrix::try_from_csc_data(self.rows, self.cols, col_offsets, row_indices, values)
            .expect("entries are unique and sorted")
    }
}

/// Returns `wrt`'s elements, which must form a column vector.
#[track_caller]
fn column<'p, S: Scalar>(wrt: VariableMatrix<'p, S>, name: &str) -> Vec<Variable<'p, S>> {
    assert!(
        wrt.cols() == 1,
        "{name} must be a column vector, got {}×{}",
        wrt.rows(),
        wrt.cols()
    );
    wrt.into_iter().collect()
}

/// Gradient of an expression with respect to a vector of variables.
///
/// ```
/// use sleipnir::{Gradient, Problem};
///
/// let problem: Problem = Problem::new();
/// let x = problem.decision_variable_matrix(2, 1);
/// problem.set_initial_matrix(&x, [[1.0], [2.0]]);
///
/// // ∇(x₀² + 3x₀x₁) = (2x₀ + 3x₁, 3x₀)
/// let g = Gradient::new(x[0] * x[0] + 3.0 * x[0] * x[1], &x);
/// assert_eq!(g.get().values(), [8.0, 3.0]);
/// assert_eq!(g.value().nnz(), 2);
/// ```
pub struct Gradient<'p, S: Scalar> {
    derivatives: Derivatives<'p, S>,
}

impl<'p, S: Scalar> Gradient<'p, S> {
    /// Builds the gradient of `f` with respect to `wrt`, a column vector or a
    /// single [`Variable`].
    ///
    /// # Panics
    ///
    /// Panics if `wrt` isn't a column vector, or `f` and `wrt` belong to
    /// different problems.
    #[track_caller]
    pub fn new(f: Variable<'p, S>, wrt: impl Into<VariableMatrix<'p, S>>) -> Self {
        let wrt = column(wrt.into(), "wrt");
        let jacobian = Derivatives::new(&[f], &wrt);
        // Transpose the 1×n Jacobian into an n×1 gradient
        Self {
            derivatives: Derivatives {
                rows: jacobian.cols,
                cols: 1,
                entries: jacobian
                    .entries
                    .into_iter()
                    .map(|(_, col, d)| (col, 0, d))
                    .collect(),
            },
        }
    }

    /// Returns the gradient as an n×1 matrix of expressions.
    pub fn get(&self) -> VariableMatrix<'p, S> {
        self.derivatives.get()
    }

    /// Evaluates the gradient at the current values as a sparse n×1 matrix.
    pub fn value(&self) -> CscMatrix<S> {
        self.derivatives.value(|_, _| true)
    }
}

/// Jacobian of a vector of expressions with respect to a vector of
/// variables.
pub struct Jacobian<'p, S: Scalar> {
    derivatives: Derivatives<'p, S>,
}

impl<'p, S: Scalar> Jacobian<'p, S> {
    /// Builds the Jacobian of `functions` (m×1) with respect to `wrt` (n×1),
    /// an m×n matrix.
    ///
    /// # Panics
    ///
    /// Panics if `functions` or `wrt` isn't a column vector, or they belong
    /// to different problems.
    #[track_caller]
    pub fn new(
        functions: impl Into<VariableMatrix<'p, S>>,
        wrt: impl Into<VariableMatrix<'p, S>>,
    ) -> Self {
        let functions = column(functions.into(), "functions");
        let wrt = column(wrt.into(), "wrt");
        Self {
            derivatives: Derivatives::new(&functions, &wrt),
        }
    }

    /// Returns the Jacobian as a matrix of expressions.
    pub fn get(&self) -> VariableMatrix<'p, S> {
        self.derivatives.get()
    }

    /// Evaluates the Jacobian at the current values as a sparse matrix.
    pub fn value(&self) -> CscMatrix<S> {
        self.derivatives.value(|_, _| true)
    }
}

/// Hessian of an expression with respect to a vector of variables.
pub struct Hessian<'p, S: Scalar> {
    derivatives: Derivatives<'p, S>,
    lower_only: bool,
}

impl<'p, S: Scalar> Hessian<'p, S> {
    /// Builds the Hessian of `f` with respect to `wrt` (n×1), an n×n matrix.
    ///
    /// # Panics
    ///
    /// Panics if `wrt` isn't a column vector, or `f` and `wrt` belong to
    /// different problems.
    #[track_caller]
    pub fn new(f: Variable<'p, S>, wrt: impl Into<VariableMatrix<'p, S>>) -> Self {
        let wrt = wrt.into();
        let gradient = Gradient::new(f, wrt.clone()).get();
        let wrt = column(wrt, "wrt");
        let gradient: Vec<_> = gradient.into_iter().collect();
        Self {
            derivatives: Derivatives::new(&gradient, &wrt),
            lower_only: false,
        }
    }

    /// Like [`new`](Self::new), but [`value`](Self::value) only computes
    /// the lower triangle, like the C++ library's `Hessian<Scalar,
    /// Eigen::Lower>`.
    #[track_caller]
    pub fn lower(f: Variable<'p, S>, wrt: impl Into<VariableMatrix<'p, S>>) -> Self {
        Self {
            lower_only: true,
            ..Self::new(f, wrt)
        }
    }

    /// Returns the full Hessian as a matrix of expressions.
    pub fn get(&self) -> VariableMatrix<'p, S> {
        self.derivatives.get()
    }

    /// Evaluates the Hessian (or its lower triangle) at the current values as
    /// a sparse matrix.
    pub fn value(&self) -> CscMatrix<S> {
        let lower_only = self.lower_only;
        self.derivatives.value(|row, col| !lower_only || col <= row)
    }
}
