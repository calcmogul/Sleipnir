// Copyright (c) Sleipnir contributors

//! Optimal control problems.

use std::time::Duration;

use crate::constraints::Operand;
use crate::{
    ConstraintOperand, Problem, Scalar, Variable, VariableBlock, VariableMatrix, constrain,
};

/// Form of an [`Ocp`]'s dynamics function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DynamicsType {
    /// Explicit ODE, dx/dt = f(t, x, u), integrated with RK4.
    #[default]
    ExplicitOde,
    /// Discrete state transition function, xₖ₊₁ = f(t, xₖ, uₖ).
    Discrete,
}

/// How an [`Ocp`]'s timesteps are chosen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TimestepMethod {
    /// The timestep is a fixed constant.
    #[default]
    Fixed,
    /// The timesteps vary as independent decision variables.
    Variable,
    /// The timesteps are equal but vary as a single decision variable.
    VariableSingle,
}

/// How an [`Ocp`] constrains its trajectory to the dynamics.
///
/// <https://underactuated.mit.edu/trajopt.html> describes each method.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TranscriptionMethod {
    /// Each state is a decision variable constrained to the integrated
    /// dynamics of the previous state.
    #[default]
    DirectTranscription,
    /// The trajectory is a series of cubic polynomials whose centerpoint
    /// slopes are constrained. Requires [`DynamicsType::ExplicitOde`].
    DirectCollocation,
    /// States are functions of the initial state and all previous inputs
    /// instead of decision variables.
    SingleShooting,
}

/// [`Ocp`] settings.
///
/// ```
/// use std::time::Duration;
///
/// use sleipnir::{OcpConfig, TranscriptionMethod};
///
/// let config = OcpConfig {
///     transcription_method: TranscriptionMethod::DirectCollocation,
///     ..OcpConfig::new(2, 1, Duration::from_millis(20), 50)
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OcpConfig {
    /// Number of states.
    pub num_states: usize,
    /// Number of inputs.
    pub num_inputs: usize,
    /// Timestep, or the initial timestep with a variable timestep method.
    pub dt: Duration,
    /// Number of steps. The trajectory has `num_steps + 1` states.
    pub num_steps: usize,
    /// Form of the dynamics function.
    pub dynamics_type: DynamicsType,
    /// How timesteps are chosen.
    pub timestep_method: TimestepMethod,
    /// How the trajectory is constrained to the dynamics.
    pub transcription_method: TranscriptionMethod,
}

impl OcpConfig {
    /// Creates settings with an explicit ODE, fixed timesteps, and direct
    /// transcription.
    pub fn new(num_states: usize, num_inputs: usize, dt: Duration, num_steps: usize) -> Self {
        Self {
            num_states,
            num_inputs,
            dt,
            num_steps,
            dynamics_type: DynamicsType::default(),
            timestep_method: TimestepMethod::default(),
            transcription_method: TranscriptionMethod::default(),
        }
    }
}

/// Optimal control problem (OCP): states, inputs, and timesteps added to a
/// [`Problem`] and constrained to follow system dynamics.
///
/// Add a cost and other constraints to the problem, then solve it.
///
/// ```
/// use std::time::Duration;
///
/// use sleipnir::{Ocp, OcpConfig, Options, Problem, SolveStatus};
///
/// // Flywheel, dx/dt = −x + u, driven from rest toward 10 with |u| ≤ 12
/// let problem: Problem = Problem::new();
/// let ocp = Ocp::new(
///     &problem,
///     OcpConfig::new(1, 1, Duration::from_millis(5), 100),
///     |x, u| -x + u,
/// );
/// ocp.constrain_initial_state(0.0);
/// ocp.set_lower_input_bound(-12.0);
/// ocp.set_upper_input_bound(12.0);
///
/// let error = ocp.x().map(|x| x - 10.0);
/// problem.minimize(error.dot(&error));
/// assert_eq!(problem.solve(&Options::default()), Ok(SolveStatus::Success));
/// assert!((ocp.u()[0].value() - 12.0).abs() < 1e-6);
/// ```
pub struct Ocp<'p, S: Scalar> {
    problem: &'p Problem<S>,
    num_states: usize,
    num_inputs: usize,
    num_steps: usize,
    transcription_method: TranscriptionMethod,
    x: VariableMatrix<'p, S>,
    u: VariableMatrix<'p, S>,
    dt: VariableMatrix<'p, S>,
}

impl<'p, S: Scalar> Ocp<'p, S> {
    /// Adds an OCP to `problem` with dynamics f(x, u): dx/dt for an explicit
    /// ODE or xₖ₊₁ for a discrete state transition function.
    ///
    /// # Panics
    ///
    /// Panics if direct collocation is used without an explicit ODE, or the
    /// dynamics return the wrong shape.
    #[track_caller]
    pub fn new(
        problem: &'p Problem<S>,
        config: OcpConfig,
        mut dynamics: impl FnMut(
            &VariableMatrix<'p, S>,
            &VariableMatrix<'p, S>,
        ) -> VariableMatrix<'p, S>,
    ) -> Self {
        Self::new_with_time(problem, config, |_, x, u, _| dynamics(x, u))
    }

    /// Like [`new`](Self::new), but with dynamics f(t, x, u, dt) that also
    /// take the time and timestep.
    ///
    /// # Panics
    ///
    /// Panics if direct collocation is used without an explicit ODE, or the
    /// dynamics return the wrong shape.
    #[track_caller]
    pub fn new_with_time(
        problem: &'p Problem<S>,
        config: OcpConfig,
        mut dynamics: impl FnMut(
            Variable<'p, S>,
            &VariableMatrix<'p, S>,
            &VariableMatrix<'p, S>,
            Variable<'p, S>,
        ) -> VariableMatrix<'p, S>,
    ) -> Self {
        let OcpConfig {
            num_states,
            num_inputs,
            num_steps,
            ..
        } = config;
        let dt = S::from_f64(config.dt.as_secs_f64());

        // u has num_steps + 1 columns so for_each_step() can use the last
        // state's input
        let u = problem.decision_variable_matrix(num_inputs, num_steps + 1);

        let dt_matrix = match config.timestep_method {
            TimestepMethod::Fixed => VariableMatrix::from_element(1, num_steps + 1, dt),
            TimestepMethod::VariableSingle => {
                let single = problem.decision_variable();
                problem.set_initial(single, dt);
                VariableMatrix::from_element(1, num_steps + 1, single)
            }
            TimestepMethod::Variable => {
                let dts = problem.decision_variable_matrix(1, num_steps + 1);
                for &dt_k in &dts {
                    problem.set_initial(dt_k, dt);
                }
                dts
            }
        };

        let mut ocp = Self {
            problem,
            num_states,
            num_inputs,
            num_steps,
            transcription_method: config.transcription_method,
            x: VariableMatrix::zeros(0, 0),
            u,
            dt: dt_matrix,
        };

        let mut f = |t, x: &VariableMatrix<'p, S>, u: &VariableMatrix<'p, S>, dt| {
            let xdot = dynamics(t, x, u, dt);
            assert!(
                (xdot.rows(), xdot.cols()) == (num_states, 1),
                "dynamics must return a {num_states}×1 matrix, got {}×{}",
                xdot.rows(),
                xdot.cols()
            );
            xdot
        };

        match config.transcription_method {
            TranscriptionMethod::DirectTranscription => {
                ocp.x = problem.decision_variable_matrix(num_states, num_steps + 1);
                ocp.constrain_direct_transcription(config.dynamics_type, &mut f);
            }
            TranscriptionMethod::DirectCollocation => {
                assert!(
                    config.dynamics_type == DynamicsType::ExplicitOde,
                    "direct collocation requires an explicit ODE"
                );
                ocp.x = problem.decision_variable_matrix(num_states, num_steps + 1);
                ocp.constrain_direct_collocation(&mut f);
            }
            TranscriptionMethod::SingleShooting => {
                // The initial state is parameters, and later states are
                // expressions of it and the inputs
                ocp.x = VariableMatrix::zeros(num_states, num_steps + 1);
                for row in 0..num_states {
                    ocp.x[(row, 0)] = problem.parameter(S::ZERO);
                }
                ocp.shoot(config.dynamics_type, &mut f);
            }
        }

        ocp
    }

    /// Problem the OCP was added to.
    pub fn problem(&self) -> &'p Problem<S> {
        self.problem
    }

    /// States, shaped (num_states)×(num_steps + 1). After solving, this is
    /// the optimized trajectory.
    pub fn x(&self) -> &VariableMatrix<'p, S> {
        &self.x
    }

    /// Inputs, shaped (num_inputs)×(num_steps + 1). The last column isn't
    /// used by the dynamics.
    pub fn u(&self) -> &VariableMatrix<'p, S> {
        &self.u
    }

    /// Timesteps, shaped 1×(num_steps + 1). The last column isn't used by
    /// the dynamics.
    pub fn dt(&self) -> &VariableMatrix<'p, S> {
        &self.dt
    }

    /// Initial state.
    pub fn initial_state(&self) -> VariableBlock<'_, 'p, S> {
        self.x.column(0)
    }

    /// Final state.
    pub fn final_state(&self) -> VariableBlock<'_, 'p, S> {
        self.x.column(self.num_steps)
    }

    /// Constrains the initial state to a scalar (broadcast), variable, or
    /// num_states×1 matrix.
    ///
    /// With single shooting, a constant initial state sets the initial
    /// state's values instead of adding constraints.
    ///
    /// # Panics
    ///
    /// Panics if a matrix has the wrong shape.
    #[track_caller]
    pub fn constrain_initial_state(&self, initial_state: impl ConstraintOperand<'p, S>) {
        let value = self.broadcast(initial_state, self.num_states, "initial state");
        if self.transcription_method == TranscriptionMethod::SingleShooting
            && value.iter().all(Variable::is_constant)
        {
            for (&x, &value) in self.initial_state().iter().zip(&value) {
                self.problem.set_value(x, value.value());
            }
        } else {
            self.problem
                .subject_to(constrain!(self.initial_state() == value));
        }
    }

    /// Constrains the final state to a scalar (broadcast), variable, or
    /// num_states×1 matrix.
    ///
    /// # Panics
    ///
    /// Panics if a matrix has the wrong shape.
    #[track_caller]
    pub fn constrain_final_state(&self, final_state: impl ConstraintOperand<'p, S>) {
        let value = self.broadcast(final_state, self.num_states, "final state");
        self.problem
            .subject_to(constrain!(self.final_state() == value));
    }

    /// Calls `f(x, u)` with each of the `num_steps + 1` states and inputs,
    /// e.g., to add constraints at every step.
    pub fn for_each_step(
        &self,
        mut f: impl FnMut(VariableBlock<'_, 'p, S>, VariableBlock<'_, 'p, S>),
    ) {
        for i in 0..=self.num_steps {
            f(self.x.column(i), self.u.column(i));
        }
    }

    /// Calls `f(t, x, u, dt)` with the time, state, input, and timestep of
    /// each of the `num_steps + 1` steps.
    pub fn for_each_step_with_time(
        &self,
        mut f: impl FnMut(
            Variable<'p, S>,
            VariableBlock<'_, 'p, S>,
            VariableBlock<'_, 'p, S>,
            Variable<'p, S>,
        ),
    ) {
        let mut time = Variable::from(S::ZERO);
        for i in 0..=self.num_steps {
            let dt = self.dt[i];
            f(time, self.x.column(i), self.u.column(i), dt);
            time += dt;
        }
    }

    /// Constrains every input to be at least a scalar (broadcast), variable,
    /// or num_inputs×1 matrix.
    ///
    /// # Panics
    ///
    /// Panics if a matrix has the wrong shape.
    #[track_caller]
    pub fn set_lower_input_bound(&self, lower_bound: impl ConstraintOperand<'p, S>) {
        let bound = self.broadcast(lower_bound, self.num_inputs, "input bound");
        for i in 0..=self.num_steps {
            self.problem
                .subject_to(constrain!(self.u.column(i) >= bound));
        }
    }

    /// Constrains every input to be at most a scalar (broadcast), variable,
    /// or num_inputs×1 matrix.
    ///
    /// # Panics
    ///
    /// Panics if a matrix has the wrong shape.
    #[track_caller]
    pub fn set_upper_input_bound(&self, upper_bound: impl ConstraintOperand<'p, S>) {
        let bound = self.broadcast(upper_bound, self.num_inputs, "input bound");
        for i in 0..=self.num_steps {
            self.problem
                .subject_to(constrain!(self.u.column(i) <= bound));
        }
    }

    /// Constrains every timestep to be at least `min_timestep`.
    pub fn set_min_timestep(&self, min_timestep: Duration) {
        let min_timestep = S::from_f64(min_timestep.as_secs_f64());
        self.problem
            .subject_to(constrain!(self.dt >= Variable::from(min_timestep)));
    }

    /// Constrains every timestep to be at most `max_timestep`.
    pub fn set_max_timestep(&self, max_timestep: Duration) {
        let max_timestep = S::from_f64(max_timestep.as_secs_f64());
        self.problem
            .subject_to(constrain!(self.dt <= Variable::from(max_timestep)));
    }

    /// Converts a scalar, variable, or matrix to a rows×1 matrix,
    /// broadcasting scalars and variables.
    #[track_caller]
    fn broadcast(
        &self,
        value: impl ConstraintOperand<'p, S>,
        rows: usize,
        name: &str,
    ) -> VariableMatrix<'p, S> {
        let matrix = match value.__operand() {
            Operand::Variable(x) => return VariableMatrix::from_element(rows, 1, x),
            Operand::Matrix(m) => m.clone(),
            Operand::Owned(m) => m,
        };
        assert!(
            (matrix.rows(), matrix.cols()) == (rows, 1),
            "{name} must be a scalar or a {rows}×1 matrix, got {}×{}",
            matrix.rows(),
            matrix.cols()
        );
        matrix
    }

    /// Integrates dx/dt = f(t, x, u) for `dt` with 4th order Runge-Kutta.
    fn rk4(
        f: &mut impl FnMut(
            Variable<'p, S>,
            &VariableMatrix<'p, S>,
            &VariableMatrix<'p, S>,
            Variable<'p, S>,
        ) -> VariableMatrix<'p, S>,
        x: &VariableMatrix<'p, S>,
        u: &VariableMatrix<'p, S>,
        t0: Variable<'p, S>,
        dt: Variable<'p, S>,
    ) -> VariableMatrix<'p, S> {
        let half_dt = dt * S::from_f64(0.5);
        let k1 = f(t0, x, u, dt);
        let k2 = f(t0 + half_dt, &(x + &k1 * half_dt), u, dt);
        let k3 = f(t0 + half_dt, &(x + &k2 * half_dt), u, dt);
        let k4 = f(t0 + dt, &(x + &k3 * dt), u, dt);
        let two = S::from_f64(2.0);
        x + (k1 + k2 * two + k3 * two + k4) * (dt / S::from_f64(6.0))
    }

    /// Steps the dynamics from state `i` for one timestep.
    fn step(
        &self,
        dynamics_type: DynamicsType,
        f: &mut impl FnMut(
            Variable<'p, S>,
            &VariableMatrix<'p, S>,
            &VariableMatrix<'p, S>,
            Variable<'p, S>,
        ) -> VariableMatrix<'p, S>,
        i: usize,
        time: Variable<'p, S>,
    ) -> VariableMatrix<'p, S> {
        let x = self.x.column(i).to_matrix();
        let u = self.u.column(i).to_matrix();
        let dt = self.dt[i];
        match dynamics_type {
            DynamicsType::ExplicitOde => Self::rk4(f, &x, &u, time, dt),
            DynamicsType::Discrete => f(time, &x, &u, dt),
        }
    }

    fn constrain_direct_transcription(
        &self,
        dynamics_type: DynamicsType,
        f: &mut impl FnMut(
            Variable<'p, S>,
            &VariableMatrix<'p, S>,
            &VariableMatrix<'p, S>,
            Variable<'p, S>,
        ) -> VariableMatrix<'p, S>,
    ) {
        let mut time = Variable::from(S::ZERO);
        for i in 0..self.num_steps {
            let x_end = self.step(dynamics_type, f, i, time);
            self.problem
                .subject_to(constrain!(self.x.column(i + 1) == x_end));
            time += self.dt[i];
        }
    }

    fn shoot(
        &mut self,
        dynamics_type: DynamicsType,
        f: &mut impl FnMut(
            Variable<'p, S>,
            &VariableMatrix<'p, S>,
            &VariableMatrix<'p, S>,
            Variable<'p, S>,
        ) -> VariableMatrix<'p, S>,
    ) {
        let mut time = Variable::from(S::ZERO);
        for i in 0..self.num_steps {
            let x_end = self.step(dynamics_type, f, i, time);
            self.x.column_mut(i + 1).copy_from(x_end);
            time += self.dt[i];
        }
    }

    /// Constrains the slope at each interval's center to the dynamics. See
    /// <https://mec560sbu.github.io/2016/09/30/direct_collocation/>.
    fn constrain_direct_collocation(
        &self,
        f: &mut impl FnMut(
            Variable<'p, S>,
            &VariableMatrix<'p, S>,
            &VariableMatrix<'p, S>,
            Variable<'p, S>,
        ) -> VariableMatrix<'p, S>,
    ) {
        let c = S::from_f64;
        let mut time = Variable::from(S::ZERO);
        for i in 0..self.num_steps {
            let h = self.dt[i];
            let (t_begin, t_end) = (time, time + h);
            let x_begin = self.x.column(i).to_matrix();
            let x_end = self.x.column(i + 1).to_matrix();
            let u_begin = self.u.column(i).to_matrix();
            let u_end = self.u.column(i + 1).to_matrix();

            let xdot_begin = f(t_begin, &x_begin, &u_begin, h);
            let xdot_end = f(t_end, &x_end, &u_end, h);
            let xdot_c = (&x_begin - &x_end) * (Variable::from(c(-3.0)) / (h * c(2.0)))
                - (&xdot_begin + &xdot_end) * c(0.25);

            let t_c = t_begin + h * c(0.5);
            let x_c = (&x_begin + &x_end) * c(0.5) + (&xdot_begin - &xdot_end) * (h / c(8.0));
            let u_c = (&u_begin + &u_end) * c(0.5);

            let xdot_center = f(t_c, &x_c, &u_c, h);
            self.problem.subject_to(constrain!(xdot_c == xdot_center));
            time += h;
        }
    }
}
