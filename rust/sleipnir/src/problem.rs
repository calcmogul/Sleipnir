// Copyright (c) Sleipnir contributors

use std::cell::{Cell, RefCell};
use std::fmt;
use std::ops::ControlFlow;
use std::panic::{self, AssertUnwindSafe};

use crate::ffi::{Backend, RawIteration, RawOptions};
use crate::variable::{Arena, Node, Repr};
use crate::{
    Constraints, ExpressionType, IterationInfo, Options, Scalar, SolveError, SolveStatus, Variable,
    VariableMatrix,
};

type IterationCallback<'a, S> = dyn FnMut(&IterationInfo<'_, S>) -> ControlFlow<()> + 'a;

/// Callback stored in a problem.
type StoredCallback<S> = Box<dyn FnMut(&IterationInfo<'_, S>) -> ControlFlow<()> + Send>;

/// C++ problem built from the expression graph, valid while the problem's
/// structure version matches.
struct Built<S: Scalar> {
    version: u64,
    handle: <S as Backend>::Handle,
    /// Parameter node IDs in creation order.
    parameters: Vec<usize>,
}

/// Nonlinear program: a cost function, decision variables, and constraints.
///
/// Every method takes `&self`, since [`Variable`] handles borrow the problem
/// while it's being built.
///
/// The first solve builds the problem in the C++ library, and later solves
/// reuse it until decision variables, the cost, or constraints change.
pub struct Problem<S: Scalar = f64> {
    arena: Arena<S>,
    decision_variables: RefCell<Vec<usize>>,
    cost: Cell<Option<usize>>,
    equality_constraints: RefCell<Vec<usize>>,
    inequality_constraints: RefCell<Vec<usize>>,
    callbacks: RefCell<Vec<StoredCallback<S>>>,
    persistent_callbacks: RefCell<Vec<StoredCallback<S>>>,
    /// Incremented whenever the C++ problem would need rebuilding.
    version: Cell<u64>,
    built: RefCell<Option<Built<S>>>,
}

impl<S: Scalar> fmt::Debug for Problem<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Problem")
            .field(
                "decision_variables",
                &self.decision_variables.borrow().len(),
            )
            .field("cost", &self.cost_function_type())
            .field(
                "equality_constraints",
                &self.equality_constraints.borrow().len(),
            )
            .field(
                "inequality_constraints",
                &self.inequality_constraints.borrow().len(),
            )
            .finish_non_exhaustive()
    }
}

impl<S: Scalar> Default for Problem<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> Problem<S> {
    /// Creates an empty problem.
    pub fn new() -> Self {
        Self {
            arena: Arena::default(),
            decision_variables: RefCell::default(),
            cost: Cell::default(),
            equality_constraints: RefCell::default(),
            inequality_constraints: RefCell::default(),
            callbacks: RefCell::default(),
            persistent_callbacks: RefCell::default(),
            version: Cell::new(0),
            built: RefCell::default(),
        }
    }

    fn structure_changed(&self) {
        self.version.set(self.version.get() + 1);
    }

    /// Creates a decision variable with an initial value of zero.
    pub fn decision_variable(&self) -> Variable<'_, S> {
        let x = Variable::new(&self.arena, Node::Decision(S::ZERO));
        self.decision_variables
            .borrow_mut()
            .push(x.node_in(&self.arena));
        self.structure_changed();
        x
    }

    /// Creates a parameter: a variable with a settable value that the solver
    /// doesn't optimize, like a C++ `Variable` not created by a problem.
    ///
    /// Change it with [`set_value`](Self::set_value) between solves to
    /// re-solve the same problem with new data, without rebuilding it.
    ///
    /// Like in the C++ library, parameters count as linear for
    /// [`expression_type`](Variable::expression_type), so `p * x` is
    /// quadratic.
    ///
    /// ```
    /// use sleipnir::{Options, Problem};
    ///
    /// let problem: Problem = Problem::new();
    /// let x = problem.decision_variable();
    /// let target = problem.parameter(1.0);
    /// problem.minimize((x - target) * (x - target));
    ///
    /// for value in [1.0, 2.0, 3.0] {
    ///     problem.set_value(target, value);
    ///     problem.solve(&Options::default()).unwrap();
    ///     assert!((x.value() - value).abs() < 1e-8);
    /// }
    /// ```
    pub fn parameter(&self, value: S) -> Variable<'_, S> {
        self.structure_changed();
        Variable::new(&self.arena, Node::Parameter(value))
    }

    /// Creates a matrix of parameters, each with the value `value`.
    pub fn parameter_matrix(&self, rows: usize, cols: usize, value: S) -> VariableMatrix<'_, S> {
        VariableMatrix::from_fn(rows, cols, |_, _| self.parameter(value))
    }

    /// Creates a matrix of decision variables.
    pub fn decision_variable_matrix(&self, rows: usize, cols: usize) -> VariableMatrix<'_, S> {
        VariableMatrix::from_fn(rows, cols, |_, _| self.decision_variable())
    }

    /// Creates a symmetric matrix of decision variables. Only the lower
    /// triangle gets its own variables.
    pub fn symmetric_decision_variable(&self, rows: usize) -> VariableMatrix<'_, S> {
        let lower: Vec<_> = (0..rows)
            .flat_map(|row| (0..=row).map(move |_| row))
            .map(|_| self.decision_variable())
            .collect();
        VariableMatrix::from_fn(rows, rows, |row, col| {
            let (row, col) = if col <= row { (row, col) } else { (col, row) };
            lower[row * (row + 1) / 2 + col]
        })
    }

    /// Sets a decision variable's initial value.
    ///
    /// # Panics
    ///
    /// Panics if `x` isn't a decision variable of this problem.
    #[track_caller]
    pub fn set_initial(&self, x: Variable<'_, S>, value: S) {
        assert!(!x.is_constant(), "constants aren't decision variables");
        let id = x.node_in(&self.arena);
        assert!(
            !matches!(self.arena.node(id), Node::Parameter(_)),
            "parameters aren't decision variables; use set_value()"
        );
        self.arena.set_leaf_value(id, value);
    }

    /// Sets a parameter's value, or a decision variable's value like
    /// [`set_initial`](Self::set_initial).
    ///
    /// # Panics
    ///
    /// Panics if `x` isn't a parameter or decision variable of this problem.
    #[track_caller]
    pub fn set_value(&self, x: Variable<'_, S>, value: S) {
        assert!(!x.is_constant(), "constants don't have settable values");
        self.arena.set_leaf_value(x.node_in(&self.arena), value);
    }

    /// Sets a matrix or view of parameters' or decision variables' values
    /// from any matrix convertible to a [`VariableMatrix`].
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ or an element of `x` isn't a parameter or
    /// decision variable of this problem.
    #[track_caller]
    pub fn set_value_matrix<'p, 'q>(
        &'p self,
        x: impl Into<VariableMatrix<'p, S>>,
        values: impl Into<VariableMatrix<'q, S>>,
    ) {
        let x = x.into();
        let values = values.into();
        assert!(
            (x.rows(), x.cols()) == (values.rows(), values.cols()),
            "matrix shape mismatch: {}×{} vs {}×{}",
            x.rows(),
            x.cols(),
            values.rows(),
            values.cols()
        );
        for (&x, value) in x.iter().zip(values.values()) {
            self.set_value(x, value);
        }
    }

    /// Sets a matrix or view of decision variables' initial values from any
    /// matrix convertible to a [`VariableMatrix`], e.g., nested arrays or
    /// nalgebra matrices.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ or an element of `x` isn't a decision
    /// variable of this problem.
    #[track_caller]
    pub fn set_initial_matrix<'p, 'q>(
        &'p self,
        x: impl Into<VariableMatrix<'p, S>>,
        values: impl Into<VariableMatrix<'q, S>>,
    ) {
        let x = x.into();
        let values = values.into();
        assert!(
            (x.rows(), x.cols()) == (values.rows(), values.cols()),
            "matrix shape mismatch: {}×{} vs {}×{}",
            x.rows(),
            x.cols(),
            values.rows(),
            values.cols()
        );
        for (&x, value) in x.iter().zip(values.values()) {
            self.set_initial(x, value);
        }
    }

    /// Sets the cost function to minimize.
    #[track_caller]
    pub fn minimize(&self, cost: Variable<'_, S>) {
        self.cost.set(Some(cost.node_in(&self.arena)));
        self.structure_changed();
    }

    /// Sets the objective function to maximize.
    #[track_caller]
    pub fn maximize(&self, objective: Variable<'_, S>) {
        self.minimize(-objective);
    }

    /// Adds constraints, usually built with [`constrain!`](crate::constrain).
    ///
    /// # Panics
    ///
    /// Panics if the constraints use variables from another problem.
    #[track_caller]
    pub fn subject_to<'p>(&'p self, constraints: impl Into<Constraints<'p, S>>) {
        let (residuals, list) = match &constraints.into() {
            Constraints::Equality(c) => (c.residuals().to_vec(), &self.equality_constraints),
            Constraints::Inequality(c) => (c.residuals().to_vec(), &self.inequality_constraints),
        };
        let ids: Vec<_> = residuals.iter().map(|c| c.node_in(&self.arena)).collect();
        list.borrow_mut().extend(ids);
        self.structure_changed();
    }

    /// Adds a callback called at the start of each solver iteration, until
    /// [`clear_callbacks`](Self::clear_callbacks) removes it. Returning
    /// [`ControlFlow::Break`] stops the solver.
    ///
    /// Stored callbacks must be `'static`, since they live in the problem;
    /// use [`solve_with_callback`](Self::solve_with_callback) for a callback
    /// that borrows. Callbacks run in the order they were added, before the
    /// `solve_with_callback` callback and persistent callbacks, and the first
    /// one to break stops the rest from running.
    pub fn add_callback(
        &self,
        callback: impl FnMut(&IterationInfo<'_, S>) -> ControlFlow<()> + Send + 'static,
    ) {
        self.callbacks.borrow_mut().push(Box::new(callback));
    }

    /// Adds a callback that [`clear_callbacks`](Self::clear_callbacks)
    /// doesn't remove. Persistent callbacks run after all other callbacks.
    pub fn add_persistent_callback(
        &self,
        callback: impl FnMut(&IterationInfo<'_, S>) -> ControlFlow<()> + Send + 'static,
    ) {
        self.persistent_callbacks
            .borrow_mut()
            .push(Box::new(callback));
    }

    /// Removes the callbacks added with [`add_callback`](Self::add_callback).
    pub fn clear_callbacks(&self) {
        self.callbacks.borrow_mut().clear();
    }

    /// Returns the cost function's type, or [`ExpressionType::None`] without
    /// one.
    pub fn cost_function_type(&self) -> ExpressionType {
        self.cost()
            .map_or(ExpressionType::None, |cost| cost.expression_type())
    }

    /// Returns the most general equality constraint type, or
    /// [`ExpressionType::None`] without equality constraints.
    pub fn equality_constraint_type(&self) -> ExpressionType {
        self.constraint_type(&self.equality_constraints.borrow())
    }

    /// Returns the most general inequality constraint type, or
    /// [`ExpressionType::None`] without inequality constraints.
    pub fn inequality_constraint_type(&self) -> ExpressionType {
        self.constraint_type(&self.inequality_constraints.borrow())
    }

    fn constraint_type(&self, ids: &[usize]) -> ExpressionType {
        ids.iter()
            .map(|&id| self.arena.expression_type(id))
            .max()
            .unwrap_or(ExpressionType::None)
    }

    /// Solves the problem. Afterward, the decision variables hold the final
    /// iterate, even if solving failed.
    ///
    /// The solver is chosen from the problem structure: Newton's method
    /// without constraints, SQP with only equality constraints, and an
    /// interior-point method otherwise.
    pub fn solve(&self, options: &Options<S>) -> Result<SolveStatus, SolveError> {
        self.solve_impl(options, None)
    }

    /// Solves the problem, calling `callback` at the start of each iteration.
    /// Returning [`ControlFlow::Break`] stops the solver, which then returns
    /// [`SolveStatus::CallbackRequestedStop`].
    ///
    /// A panic in `callback` stops the solver, and the panic resumes once
    /// the C++ solver has returned.
    pub fn solve_with_callback(
        &self,
        options: &Options<S>,
        mut callback: impl FnMut(&IterationInfo<'_, S>) -> ControlFlow<()>,
    ) -> Result<SolveStatus, SolveError> {
        self.solve_impl(options, Some(&mut callback))
    }

    fn solve_impl(
        &self,
        options: &Options<S>,
        callback: Option<&mut IterationCallback<'_, S>>,
    ) -> Result<SolveStatus, SolveError> {
        // Build the C++ problem if its structure changed
        let mut built = self.built.borrow_mut();
        if built
            .as_ref()
            .is_none_or(|built| built.version != self.version.get())
        {
            let nodes = self.arena.nodes();
            let handle = S::build(
                &nodes,
                self.cost.get(),
                &self.equality_constraints.borrow(),
                &self.inequality_constraints.borrow(),
            )
            .unwrap_or_else(|e| panic!("Sleipnir threw an exception: {}", e.what()));
            let parameters = nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| matches!(node, Node::Parameter(_)))
                .map(|(id, _)| id)
                .collect();
            *built = Some(Built {
                version: self.version.get(),
                handle,
                parameters,
            });
        }
        let built = built.as_mut().unwrap();

        let leaf_value = |id| match self.arena.node(id) {
            Node::Decision(value) | Node::Parameter(value) => value,
            _ => unreachable!("leaf IDs point at leaves"),
        };
        let decision_variables = self.decision_variables.borrow().clone();
        let decision_values: Vec<_> = decision_variables
            .iter()
            .map(|&id| leaf_value(id))
            .collect();
        let parameter_values: Vec<_> = built.parameters.iter().map(|&id| leaf_value(id)).collect();
        let raw_options = RawOptions {
            tolerance: options.tolerance,
            max_iterations: options.max_iterations.min(i32::MAX as u32) as i32,
            timeout: options
                .timeout
                .map_or(f64::INFINITY, |timeout| timeout.as_secs_f64()),
            feasible_ipm: options.feasible_ipm,
            diagnostics: options.diagnostics,
            spy: options.spy,
        };

        // Take the stored callbacks so callbacks can add more without a
        // RefCell conflict
        let mut callbacks = std::mem::take(&mut *self.callbacks.borrow_mut());
        let mut persistent_callbacks = std::mem::take(&mut *self.persistent_callbacks.borrow_mut());
        let has_callback =
            !callbacks.is_empty() || callback.is_some() || !persistent_callbacks.is_empty();

        let mut values = vec![S::ZERO; decision_variables.len()];
        let mut panic_payload = None;
        let mut callback = callback;
        let mut on_iteration = |raw: &RawIteration<'_, S>| {
            let info = IterationInfo {
                iteration: raw.iteration as u32,
                x: raw.x,
                s: raw.s,
                y: raw.y,
                z: raw.z,
                g: &raw.g,
                h: &raw.h,
                a_e: &raw.a_e,
                a_i: &raw.a_i,
            };
            let run = || {
                for callback in callbacks.iter_mut() {
                    if callback(&info).is_break() {
                        return true;
                    }
                }
                if let Some(callback) = callback.as_mut()
                    && callback(&info).is_break()
                {
                    return true;
                }
                persistent_callbacks
                    .iter_mut()
                    .any(|callback| callback(&info).is_break())
            };
            match panic::catch_unwind(AssertUnwindSafe(run)) {
                Ok(stop) => stop,
                Err(payload) => {
                    panic_payload = Some(payload);
                    true
                }
            }
        };
        let status = S::solve(
            &mut built.handle,
            &decision_values,
            &parameter_values,
            &raw_options,
            has_callback
                .then_some(&mut on_iteration as &mut dyn FnMut(&RawIteration<'_, S>) -> bool),
            &mut values,
        )
        .unwrap_or_else(|e| panic!("Sleipnir threw an exception: {}", e.what()));

        // Restore the stored callbacks ahead of any added during the solve
        callbacks.append(&mut self.callbacks.borrow_mut());
        *self.callbacks.borrow_mut() = callbacks;
        persistent_callbacks.append(&mut self.persistent_callbacks.borrow_mut());
        *self.persistent_callbacks.borrow_mut() = persistent_callbacks;

        for (&id, &value) in decision_variables.iter().zip(&values) {
            self.arena.set_leaf_value(id, value);
        }
        if let Some(payload) = panic_payload {
            panic::resume_unwind(payload);
        }

        SolveError::from_exit_status(status)
    }

    /// Decision variables in creation order.
    pub fn decision_variables(&self) -> Vec<Variable<'_, S>> {
        self.handles(&self.decision_variables.borrow())
    }

    /// Cost function, if one was set.
    pub fn cost(&self) -> Option<Variable<'_, S>> {
        self.cost
            .get()
            .map(|id| Variable(Repr::Node(&self.arena, id)))
    }

    /// Equality constraint functions cᵢ(x) = 0.
    pub fn equality_constraints(&self) -> Vec<Variable<'_, S>> {
        self.handles(&self.equality_constraints.borrow())
    }

    /// Inequality constraint functions cᵢ(x) ≥ 0.
    pub fn inequality_constraints(&self) -> Vec<Variable<'_, S>> {
        self.handles(&self.inequality_constraints.borrow())
    }

    #[cfg(test)]
    pub(crate) fn node_count(&self) -> usize {
        self.arena.len()
    }

    fn handles(&self, ids: &[usize]) -> Vec<Variable<'_, S>> {
        ids.iter()
            .map(|&id| Variable(Repr::Node(&self.arena, id)))
            .collect()
    }
}
