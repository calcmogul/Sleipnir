// Copyright (c) Sleipnir contributors

//! Allocates a limited current budget among subsystems so each gets as close
//! to its desired current as its tolerance allows.
//!
//! The desired currents are parameters, so each `calculate()` call re-solves
//! the same problem with new data.

use sleipnir::{Options, Problem, Variable, VariableMatrix, constrain};

/// Allocates current among subsystems.
///
/// The variables borrow the problem, so the manager borrows a problem the
/// caller creates instead of owning one like the C++ version.
struct CurrentManager<'p> {
    problem: &'p Problem,
    desired_currents: VariableMatrix<'p, f64>,
    allocated_currents: VariableMatrix<'p, f64>,
}

impl<'p> CurrentManager<'p> {
    /// Builds the allocation problem for subsystems with the given current
    /// tolerances and a shared current limit.
    fn new(problem: &'p Problem, current_tolerances: &[f64], max_current: f64) -> Self {
        let n = current_tolerances.len();
        let desired_currents = problem.parameter_matrix(n, 1, f64::INFINITY);
        let allocated_currents = problem.decision_variable_matrix(n, 1);

        let mut cost = Variable::from(0.0);
        let mut current_sum = Variable::from(0.0);
        for (i, &tolerance) in current_tolerances.iter().enumerate() {
            let error = desired_currents[i] - allocated_currents[i];
            cost += error * error / (tolerance * tolerance);
            current_sum += allocated_currents[i];

            problem.subject_to(constrain!(allocated_currents[i] >= 0.0));
        }
        problem.minimize(cost);

        problem.subject_to(constrain!(current_sum <= max_current));

        Self {
            problem,
            desired_currents,
            allocated_currents,
        }
    }

    /// Returns the current allocated to each subsystem.
    ///
    /// # Panics
    ///
    /// Panics if the number of desired currents differs from the number of
    /// tolerances passed to `new()`.
    fn calculate(&self, desired_currents: &[f64]) -> Vec<f64> {
        assert_eq!(
            self.desired_currents.rows(),
            desired_currents.len(),
            "Number of desired currents must equal the number of tolerances passed in the \
             constructor."
        );

        for (&parameter, &current) in self.desired_currents.iter().zip(desired_currents) {
            self.problem.set_value(parameter, current);
        }

        // The allocation is still useful if solving didn't fully converge
        let _ = self.problem.solve(&Options::default());

        self.allocated_currents
            .iter()
            .map(|current| current.value().max(0.0))
            .collect()
    }
}

fn main() {
    let problem = Problem::new();
    let manager = CurrentManager::new(&problem, &[1.0, 5.0, 10.0, 5.0], 40.0);

    let currents = manager.calculate(&[25.0, 10.0, 5.0, 0.0]);

    println!("currents = {currents:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(actual: &[f64], expected: &[f64]) {
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() < 1e-3,
                "expected {expected:?}, got {actual:?}"
            );
        }
    }

    #[test]
    fn enough_current() {
        let problem = Problem::new();
        let manager = CurrentManager::new(&problem, &[1.0, 5.0, 10.0, 5.0], 40.0);
        let currents = manager.calculate(&[25.0, 10.0, 5.0, 0.0]);
        assert_near(&currents, &[25.0, 10.0, 5.0, 0.0]);
    }

    #[test]
    fn not_enough_current() {
        let problem = Problem::new();
        let manager = CurrentManager::new(&problem, &[1.0, 5.0, 10.0, 5.0], 40.0);
        let currents = manager.calculate(&[30.0, 10.0, 5.0, 0.0]);

        // Expected values are from the following program:
        //
        // ```python
        // #!/usr/bin/env python
        //
        // from scipy.optimize import minimize
        //
        // r = [30.0, 10.0, 5.0, 0.0]
        // q = [1.0, 5.0, 10.0, 5.0]
        //
        // result = minimize(
        //     lambda x: sum((r[i] - x[i]) ** 2 / q[i] ** 2 for i in range(4)),
        //     [0.0, 0.0, 0.0, 0.0],
        //     constraints=[
        //         {"type": "ineq", "fun": lambda x: x},
        //         {"type": "ineq", "fun": lambda x: 40.0 - sum(x)},
        //     ],
        // )
        // print(result.x)
        // ```
        assert_near(&currents, &[29.960, 9.008, 1.032, 0.0]);
    }

    #[test]
    fn recalculates_with_new_currents() {
        // Both cases on one manager, which reuses the built problem
        let problem = Problem::new();
        let manager = CurrentManager::new(&problem, &[1.0, 5.0, 10.0, 5.0], 40.0);
        assert_near(
            &manager.calculate(&[30.0, 10.0, 5.0, 0.0]),
            &[29.960, 9.008, 1.032, 0.0],
        );
        assert_near(
            &manager.calculate(&[25.0, 10.0, 5.0, 0.0]),
            &[25.0, 10.0, 5.0, 0.0],
        );
    }
}
