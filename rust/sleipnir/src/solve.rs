// Copyright (c) Sleipnir contributors

use std::fmt;
use std::time::Duration;

use nalgebra_sparse::CscMatrix;

use crate::variable::Repr;
use crate::{Scalar, Variable};

/// Solver options.
///
/// ```
/// use std::time::Duration;
///
/// let options = sleipnir::Options::<f64> {
///     timeout: Some(Duration::from_millis(50)),
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options<S: Scalar = f64> {
    /// The solver stops when the error falls below this tolerance.
    pub tolerance: S,
    /// Maximum number of solver iterations before returning a solution.
    pub max_iterations: u32,
    /// Maximum time to solve for, or `None` for no limit.
    pub timeout: Option<Duration>,
    /// Keeps iterates feasible with respect to the inequality constraints
    /// once they become feasible.
    pub feasible_ipm: bool,
    /// Prints solver diagnostics.
    pub diagnostics: bool,
    /// Writes the sparsity patterns of H, Aₑ, and Aᵢ to files named H.spy,
    /// A_e.spy, and A_i.spy in the working directory during solve. Use the
    /// repository's tools/spy.py to plot them.
    pub spy: bool,
}

impl<S: Scalar> Default for Options<S> {
    fn default() -> Self {
        Self {
            tolerance: S::EPSILON.sqrt(),
            max_iterations: 5000,
            timeout: None,
            feasible_ipm: false,
            diagnostics: false,
            spy: false,
        }
    }
}

/// How a successful solve ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveStatus {
    /// The solver found a solution within the tolerance.
    Success,
    /// An iteration callback returned [`ControlFlow::Break`](std::ops::ControlFlow::Break).
    CallbackRequestedStop,
}

/// Why a solve failed. The decision variables hold the last iterate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolveError {
    /// The problem has fewer degrees of freedom than equality constraints.
    TooFewDofs,
    /// The solver converged to a point that locally minimizes the constraint
    /// violation.
    LocallyInfeasible,
    /// The constraints are globally infeasible.
    GloballyInfeasible,
    /// Factorizing the KKT system failed.
    FactorizationFailed,
    /// The backtracking line search failed, and the problem isn't locally
    /// infeasible.
    LineSearchFailed,
    /// Feasibility restoration failed.
    FeasibilityRestorationFailed,
    /// The initial guess or a function evaluated at it isn't finite.
    NonfiniteInitialGuess,
    /// The iterates diverged.
    DivergingIterates,
    /// The solver reached [`Options::max_iterations`].
    MaxIterationsExceeded,
    /// The solver reached [`Options::timeout`].
    Timeout,
}

impl SolveError {
    pub(crate) fn from_exit_status(status: i32) -> Result<SolveStatus, Self> {
        match status {
            0 => Ok(SolveStatus::Success),
            1 => Ok(SolveStatus::CallbackRequestedStop),
            -1 => Err(Self::TooFewDofs),
            -2 => Err(Self::LocallyInfeasible),
            -3 => Err(Self::GloballyInfeasible),
            -4 => Err(Self::FactorizationFailed),
            -5 => Err(Self::LineSearchFailed),
            -6 => Err(Self::FeasibilityRestorationFailed),
            -7 => Err(Self::NonfiniteInitialGuess),
            -8 => Err(Self::DivergingIterates),
            -9 => Err(Self::MaxIterationsExceeded),
            -10 => Err(Self::Timeout),
            _ => panic!("unknown slp::ExitStatus {status}"),
        }
    }
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TooFewDofs => "too few degrees of freedom",
            Self::LocallyInfeasible => "locally infeasible",
            Self::GloballyInfeasible => "globally infeasible",
            Self::FactorizationFailed => "factorization failed",
            Self::LineSearchFailed => "line search failed",
            Self::FeasibilityRestorationFailed => "feasibility restoration failed",
            Self::NonfiniteInitialGuess => "nonfinite initial guess",
            Self::DivergingIterates => "diverging iterates",
            Self::MaxIterationsExceeded => "max iterations exceeded",
            Self::Timeout => "timeout",
        })
    }
}

impl std::error::Error for SolveError {}

/// Solver state passed to iteration callbacks.
///
/// The matrices are the C++ solver's: `h` is the Lagrangian Hessian (lower
/// triangle), and `a_e` and `a_i` are the equality and inequality constraint
/// Jacobians.
#[derive(Debug)]
pub struct IterationInfo<'a, S: Scalar> {
    /// Iteration number.
    pub iteration: u32,
    /// Decision variable values, in creation order.
    pub x: &'a [S],
    /// Inequality constraint slack variables.
    pub s: &'a [S],
    /// Equality constraint dual variables.
    pub y: &'a [S],
    /// Inequality constraint dual variables.
    pub z: &'a [S],
    /// Cost function gradient, as an n×1 matrix.
    pub g: &'a CscMatrix<S>,
    /// Lagrangian Hessian.
    pub h: &'a CscMatrix<S>,
    /// Equality constraint Jacobian.
    pub a_e: &'a CscMatrix<S>,
    /// Inequality constraint Jacobian.
    pub a_i: &'a CscMatrix<S>,
}

impl<S: Scalar> IterationInfo<'_, S> {
    /// Evaluates an expression at this iterate.
    pub fn value(&self, expr: Variable<'_, S>) -> S {
        match expr.0 {
            Repr::Constant(value) => value,
            Repr::Node(arena, id) => arena.value_at(id, self.x),
        }
    }
}
