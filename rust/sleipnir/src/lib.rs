// Copyright (c) Sleipnir contributors

//! Reverse mode autodiff library and NLP solver DSL, backed by the Sleipnir
//! C++ library.
//!
//! Build a [`Problem`] from decision variables, a cost, and constraints, then
//! solve it. The solver is chosen from the problem's structure.
//!
//! ```
//! use sleipnir::{Options, Problem, SolveStatus, constrain};
//!
//! let problem: Problem = Problem::new();
//!
//! let x = problem.decision_variable();
//! let y = problem.decision_variable_matrix(2, 1);
//!
//! problem.minimize(x * x + (y[0] - 3.0) * (y[0] - 3.0));
//! problem.subject_to(constrain!(x + y[1] == 1.0));
//! problem.subject_to(constrain!(-12.0 <= x <= 12.0));
//! problem.subject_to(constrain!(y >= 0.0));
//!
//! assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
//! assert!((y[0].value() - 3.0).abs() < 1e-6);
//! ```
//!
//! [`Ocp`] builds optimal control problems, [`Gradient`], [`Jacobian`], and
//! [`Hessian`] differentiate expressions, and [`multistart`] solves from
//! several initial guesses in parallel.

extern crate self as sleipnir;

mod autodiff;
mod constraints;
mod ffi;
mod linalg;
mod matrix_ops;
mod multistart;
mod ocp;
mod problem;
mod scalar;
mod solve;
mod variable;
mod variable_matrix;
mod view;

pub use autodiff::{Gradient, Hessian, Jacobian};
pub use constraints::{
    Compare, ConstraintOperand, Constraints, EqualityConstraints, InequalityConstraints,
};
pub use multistart::{MultistartResult, multistart};
pub use nalgebra;
pub use nalgebra_sparse;
pub use ocp::{DynamicsType, Ocp, OcpConfig, TimestepMethod, TranscriptionMethod};
pub use problem::Problem;
pub use scalar::Scalar;
pub use solve::{IterationInfo, Options, SolveError, SolveStatus};
pub use variable::{Condition, ExpressionType, Variable, if_else};
pub use variable_matrix::VariableMatrix;
pub use view::{Slice, VariableBlock, VariableBlockMut, ViewRange};

/// Builds constraints from a comparison.
///
/// `lhs == rhs` returns [`EqualityConstraints`], and `lhs <= rhs` or
/// `lhs >= rhs` returns [`InequalityConstraints`]. Inequalities can be
/// chained, e.g., `lower <= x <= upper`, and each operand is evaluated once.
///
/// Operands are [`ConstraintOperand`]s: floats of the problem's scalar
/// type, [`Variable`]s, [`VariableMatrix`]es, nested arrays, nalgebra matrices
///, or references to any of these. At least one
/// side of each comparison must not be a plain float. Matrices are compared
/// elementwise, and scalars and [`Variable`]s are broadcast against matrices.
/// Scalar literals must be floating-point (`0.0`, not `0`).
///
/// The macro expands to the [`Compare`] methods, which can also be called
/// directly.
///
/// ```
/// use sleipnir::{Problem, constrain};
///
/// let problem: Problem = Problem::new();
/// let x = problem.decision_variable();
/// let y = problem.decision_variable();
///
/// problem.subject_to(constrain!(x + y == 1.0));
/// problem.subject_to(constrain!(x * x <= y));
/// problem.subject_to(constrain!(0.0 <= x <= 1.0));
/// ```
#[doc(inline)]
pub use sleipnir_macros::constrain;

#[doc(hidden)]
pub mod __private {
    pub use crate::constraints::chain;
}

/// Invalid `constrain!` uses.
///
/// Inequalities must be non-strict:
///
/// ```compile_fail
/// # let problem: sleipnir::Problem = sleipnir::Problem::new();
/// # let x = problem.decision_variable();
/// sleipnir::constrain!(x < 1.0);
/// ```
///
/// ```compile_fail
/// # let problem: sleipnir::Problem = sleipnir::Problem::new();
/// # let x = problem.decision_variable();
/// sleipnir::constrain!(x > 1.0);
/// ```
///
/// `!=` isn't a constraint:
///
/// ```compile_fail
/// # let problem: sleipnir::Problem = sleipnir::Problem::new();
/// # let x = problem.decision_variable();
/// sleipnir::constrain!(x != 1.0);
/// ```
///
/// A comparison is required:
///
/// ```compile_fail
/// # let problem: sleipnir::Problem = sleipnir::Problem::new();
/// # let x = problem.decision_variable();
/// sleipnir::constrain!(x + 1.0);
/// ```
///
/// ```compile_fail
/// # let problem: sleipnir::Problem = sleipnir::Problem::new();
/// # let mut x = problem.decision_variable();
/// sleipnir::constrain!(x = 1.0);
/// ```
///
/// ```compile_fail
/// sleipnir::constrain!();
/// ```
///
/// Operands can't be empty:
///
/// ```compile_fail
/// # let problem: sleipnir::Problem = sleipnir::Problem::new();
/// # let x = problem.decision_variable();
/// sleipnir::constrain!(x >=);
/// ```
///
/// Equalities can't be chained:
///
/// ```compile_fail
/// # let problem: sleipnir::Problem = sleipnir::Problem::new();
/// # let x = problem.decision_variable();
/// # let y = problem.decision_variable();
/// sleipnir::constrain!(x == y == 1.0);
/// ```
///
/// Chained inequalities must point the same way:
///
/// ```compile_fail
/// # let problem: sleipnir::Problem = sleipnir::Problem::new();
/// # let x = problem.decision_variable();
/// sleipnir::constrain!(0.0 <= x >= 1.0);
/// ```
///
/// At least one side must be a variable:
///
/// ```compile_fail
/// sleipnir::constrain!(1.0 >= 2.0);
/// ```
///
/// Integer literals aren't scalars:
///
/// ```compile_fail
/// # let problem: sleipnir::Problem = sleipnir::Problem::new();
/// # let x = problem.decision_variable();
/// sleipnir::constrain!(x >= 0);
/// ```
///
/// Scalar types can't be mixed:
///
/// ```compile_fail
/// # let problem: sleipnir::Problem<f32> = sleipnir::Problem::new();
/// # let x = problem.decision_variable();
/// sleipnir::constrain!(x >= 0.0_f64);
/// ```
///
/// Variables can't outlive their problem:
///
/// ```compile_fail
/// let x = {
///     let problem: sleipnir::Problem = sleipnir::Problem::new();
///     problem.decision_variable()
/// };
/// ```
#[cfg(doctest)]
pub struct CompileFailTests;
