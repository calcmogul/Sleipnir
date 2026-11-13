// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>  // IWYU pragma: keep
#include <functional>
#include <limits>
#include <optional>
#include <span>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/control_flow.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/sqp_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/util/all_finite.hpp"
#include "sleipnir/optimization/solver/util/constraint_violation.hpp"
#include "sleipnir/optimization/solver/util/feasibility_restoration.hpp"
#include "sleipnir/optimization/solver/util/feasibility_restoration_cycle_detector.hpp"
#include "sleipnir/optimization/solver/util/filter.hpp"
#include "sleipnir/optimization/solver/util/kkt_error.hpp"
#include "sleipnir/optimization/solver/util/kkt_system.hpp"
#include "sleipnir/optimization/solver/util/tangent_space.hpp"
#include "sleipnir/optimization/solver/util/time_trace.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/print_diagnostics.hpp"
#include "sleipnir/util/scope_exit.hpp"
#include "sleipnir/util/symbol_exports.hpp"
#include "sleipnir/util/time_tracer.hpp"

// See docs/algorithms.md#Works_cited for citation definitions.

namespace slp {

/// Finds the optimal solution to a nonlinear program using Sequential Quadratic
/// Programming (SQP).
///
/// A nonlinear program has the form:
///
/// ```
///      min_x f(x)
/// subject to cₑ(x) = 0
/// ```
///
/// where f(x) is the cost function and cₑ(x) are the equality constraints.
///
/// @tparam Scalar Scalar type.
/// @param[in] matrix_callbacks Matrix callbacks.
/// @param[in] iteration_callbacks The list of callbacks to call at the
///     beginning of each iteration.
/// @param[in] options Solver options.
/// @param[in,out] x The initial guess and output location for the decision
///     variables.
/// @return The exit status.
template <typename Scalar>
ExitStatus sqp(
    const SQPMatrixCallbacks<Scalar>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options<Scalar>& options, Eigen::VectorX<Scalar>& x) {
  using DenseVector = Eigen::VectorX<Scalar>;

  DenseVector y = DenseVector::Zero(matrix_callbacks.num_equality_constraints);

  return sqp(matrix_callbacks, iteration_callbacks, options, x, y);
}

/// Finds the optimal solution to a nonlinear program using Sequential Quadratic
/// Programming (SQP).
///
/// A nonlinear program has the form:
///
/// ```
///      min_x f(x)
/// subject to cₑ(x) = 0
/// ```
///
/// where f(x) is the cost function and cₑ(x) are the equality constraints.
///
/// @tparam Scalar Scalar type.
/// @param[in] matrix_callbacks Matrix callbacks.
/// @param[in] iteration_callbacks The list of callbacks to call at the
///     beginning of each iteration.
/// @param[in] options Solver options.
/// @param[in,out] x The initial guess and output location for the decision
///     variables.
/// @param[in,out] y The initial guess and output location for the equality
///     constraint dual variables.
/// @return The exit status.
template <typename Scalar>
ExitStatus sqp(
    const SQPMatrixCallbacks<Scalar>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options<Scalar>& options, Eigen::VectorX<Scalar>& x,
    Eigen::VectorX<Scalar>& y) {
  using DenseVector = Eigen::VectorX<Scalar>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using SparseVector = Eigen::SparseVector<Scalar>;

  using std::isfinite;

  const auto solve_start_time = std::chrono::steady_clock::now();

  using enum SolveTimeTracerID;

  int iterations = 0;

  // Prints final solver diagnostics when the solver exits
  SolveDiagnostics<sqp_time_trace> diagnostics{options.diagnostics,
                                               options.diagnostics};

  // Set up traced matrix callbacks if diagnostics are enabled
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  std::optional<SQPMatrixCallbacks<Scalar>> traced_matrices;
  if (options.diagnostics) {
    traced_matrices.emplace(
        matrix_callbacks.num_decision_variables,
        matrix_callbacks.num_equality_constraints,
        traced(diagnostics[COST], matrix_callbacks.f),
        traced(diagnostics[GRADIENT], matrix_callbacks.g),
        traced(diagnostics[HESSIAN], matrix_callbacks.H),
        traced(diagnostics[HESSIAN_CONSTRAINTS], matrix_callbacks.H_c),
        traced(diagnostics[EQUALITY_CONSTRAINTS], matrix_callbacks.c_e),
        traced(diagnostics[EQUALITY_JACOBIAN], matrix_callbacks.A_e),
        matrix_callbacks.scaling, matrix_callbacks.manifold);
  }
  const auto& ambient_matrices =
      traced_matrices ? *traced_matrices : matrix_callbacks;
#else
  const auto& ambient_matrices = matrix_callbacks;
#endif

  // Take derivatives in the manifold's tangent space if it isn't Euclidean
  const auto& manifold = matrix_callbacks.manifold;
  TangentSpace<Scalar> tangent_space{manifold};
  std::optional<SQPMatrixCallbacks<Scalar>> tangent_matrices;
  if (!manifold.is_euclidean()) {
    tangent_matrices.emplace(tangent_space.callbacks(ambient_matrices));
  }
  const auto& matrices =
      tangent_matrices ? *tangent_matrices : ambient_matrices;

  ScopedTimeTracer setup_tracer{diagnostics[SETUP]};

  Scalar f = matrices.f(x);
  SparseVector g = matrices.g(x);
  SparseMatrix H = matrices.H(x, y);
  DenseVector c_e = matrices.c_e(x);
  SparseMatrix A_e = matrices.A_e(x);

  // Ensure matrix callback dimensions are consistent
  slp_assert(g.rows() == matrices.num_decision_variables);
  slp_assert(H.rows() == matrices.num_decision_variables);
  slp_assert(H.cols() == matrices.num_decision_variables);
  slp_assert(c_e.rows() == matrices.num_equality_constraints);
  slp_assert(A_e.rows() == matrices.num_equality_constraints);
  slp_assert(A_e.cols() == matrices.num_decision_variables);

  DenseVector trial_x;
  DenseVector trial_y;

  Scalar trial_f;
  DenseVector trial_c_e;

  // Check for overconstrained problem
  if (matrices.num_equality_constraints > matrices.num_decision_variables) {
    if (options.diagnostics) {
      print_too_few_dofs_error(c_e);
    }

    return ExitStatus::TOO_FEW_DOFS;
  }

  // Check whether initial guess has finite cost, constraints, and derivatives
  if (!isfinite(f) || !all_finite(g) || !all_finite(H) || !c_e.allFinite() ||
      !all_finite(A_e)) {
    return ExitStatus::NONFINITE_INITIAL_GUESS;
  }

  Filter<Scalar> filter{constraint_violation(c_e)};

  SQPKKTSystem<Scalar> kkt_system{H, A_e};

  // Step size reduction factor
  constexpr Scalar α_reduction_factor(0.5);

  int full_step_rejected_counter = 0;

  // The smallest KKT error of a step accepted by the KKT error reduction
  // fallback. Each fallback step must improve on it, so finitely many fallback
  // steps are taken before the solver converges.
  Scalar fallback_kkt_error = std::numeric_limits<Scalar>::infinity();

  StallDetector<Scalar> stall_detector;

  FeasibilityRestorationCycleDetector<Scalar>
      feasibility_restoration_cycle_detector;

  // Error
  Scalar E_0 = unscaled_kkt_error_inf_norm<Scalar>(matrices.scaling, g, A_e,
                                                   c_e, y, options.tolerance);

  setup_tracer.stop();

  // Prints initial iterate diagnostics now and closes the iterate diagnostics
  // table when the solver exits
  IterateDiagnosticsTable table{
      options.diagnostics,       E_0,       f,
      constraint_violation(c_e), Scalar(0), Scalar(0)};

  while (E_0 > options.tolerance) {
    ScopedTimeTracer inner_iter_tracer{diagnostics[ITERATION]};

    // Check for diverging iterates
    if (x.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !x.allFinite()) {
      return ExitStatus::DIVERGING_ITERATES;
    }

    PhaseTimeTracer phase_tracer{diagnostics[CALLBACKS]};

    // Call iteration callbacks
    for (const auto& callback : iteration_callbacks) {
      if (callback({iterations, x, {}, y, {}, g, H, A_e, {}}) ==
          ControlFlow::BREAK) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    phase_tracer.next(diagnostics[KKT_MATRIX_BUILD]);

    kkt_system.build(H, A_e);

    phase_tracer.next(diagnostics[KKT_MATRIX_FACTOR]);

    if (!kkt_system.factorize()) [[unlikely]] {
      return ExitStatus::FACTORIZATION_FAILED;
    }

    phase_tracer.next(diagnostics[KKT_SYSTEM_SOLVE]);

    auto step = kkt_system.solve(g, A_e, y, c_e);

    phase_tracer.next(diagnostics[LINE_SEARCH]);

    bool call_feasibility_restoration = false;

    constexpr Scalar α_max(1);
    Scalar α = α_max;

    const FilterEntry<Scalar> current_entry{f, c_e};

    const Scalar D_ϕ = g.transpose() * step.p_x;

    const Scalar α_min = filter.min_step_size(current_entry, D_ϕ);

    // α_min can be zero (e.g., when the constraint violation is zero), so also
    // end the line search when the step becomes negligible relative to the
    // iterate. This bounds the number of step size reductions.
    const Scalar α_tiny = step.tiny_step_size(x, manifold);

    // Loop until a step is accepted
    while (1) {
      manifold.retract(x, α, step.p_x, trial_x);
      trial_y = y + α * step.p_y;

      trial_f = matrices.f(trial_x);
      trial_c_e = matrices.c_e(trial_x);

      // If f(xₖ + αpₖˣ) or cₑ(xₖ + αpₖˣ) aren't finite, reduce step size
      // immediately
      if (!isfinite(trial_f) || !trial_c_e.allFinite()) {
        // Reduce step size
        α *= α_reduction_factor;

        if (α < α_min || α < α_tiny) {
          call_feasibility_restoration = true;
          break;
        }
        continue;
      }

      // Check whether filter accepts trial iterate
      FilterEntry trial_entry{trial_f, trial_c_e};
      if (filter.try_add(current_entry, trial_entry, D_ϕ, α)) {
        // Accept step
        break;
      }

      Scalar prev_constraint_violation = constraint_violation(c_e);
      Scalar next_constraint_violation = constraint_violation(trial_c_e);

      // Second-order corrections
      //
      // If first trial point was rejected and constraint violation stayed the
      // same or went up, apply second-order corrections
      if (α == α_max &&
          next_constraint_violation >= prev_constraint_violation) {
        // Apply second-order corrections. See section 2.4 of [2].
        auto soc_step = step;

        Scalar α_soc = α;
        DenseVector c_e_soc = c_e;

        Scalar soc_constraint_violation = next_constraint_violation;

        bool step_acceptable = false;
        for (int soc_iteration = 0; soc_iteration < 5 && !step_acceptable;
             ++soc_iteration) {
          ScopedTimeTracer soc_tracer{diagnostics[SOC]};

          scope_exit soc_exit{[&] {
            soc_tracer.stop();

            if (options.diagnostics && step_acceptable) {
              print_iterate_diagnostics(
                  iterations, IterationType::SECOND_ORDER_CORRECTION,
                  soc_tracer.current_duration(),
                  kkt_error_inf_norm<Scalar>(g, A_e, trial_c_e, trial_y,
                                             options.tolerance),
                  trial_f, constraint_violation(trial_c_e), Scalar(0),
                  Scalar(0), kkt_system.hessian_regularization(),
                  kkt_system.constraint_jacobian_regularization(),
                  soc_step.primal_inf_norm(), soc_step.dual_inf_norm(), α_soc,
                  Scalar(1), α_reduction_factor, Scalar(1));
            }
          }};

          // Solve the Newton-KKT system with corrected constraint values
          //
          //   cₑˢᵒᶜ = αc(xₖ) + c(xₖ + αpₖˣ)
          c_e_soc = α_soc * c_e_soc + trial_c_e;
          soc_step = kkt_system.solve(g, A_e, y, c_e_soc);

          manifold.retract(x, α_soc, soc_step.p_x, trial_x);
          trial_y = y + α_soc * soc_step.p_y;

          trial_f = matrices.f(trial_x);
          trial_c_e = matrices.c_e(trial_x);

          // Check whether the filter accepts trial iterate
          FilterEntry trial_entry{trial_f, trial_c_e};
          if (filter.try_add(current_entry, trial_entry, D_ϕ, α)) {
            step = soc_step;
            α = α_soc;
            step_acceptable = true;
            break;
          }

          // Constraint violation scale factor for second-order corrections
          constexpr Scalar κ_soc(0.99);

          // If constraint violation hasn't been sufficiently reduced, stop
          // making second-order corrections
          next_constraint_violation = constraint_violation(trial_c_e);
          if (next_constraint_violation > κ_soc * soc_constraint_violation) {
            break;
          }

          soc_constraint_violation = next_constraint_violation;
        }

        if (step_acceptable) {
          // Accept step
          break;
        }
      }

      // If we got here and α is the full step, the full step was rejected.
      // Increment the full-step rejected counter to keep track of how many full
      // steps have been rejected in a row.
      if (α == α_max) {
        ++full_step_rejected_counter;
      }

      // If the full step was rejected enough times in a row, reset the filter
      // if it may be impeding progress
      if (full_step_rejected_counter >= 4 && filter.try_reset(current_entry)) {
        continue;
      }

      // Reduce step size
      α *= α_reduction_factor;

      // If step size hit a minimum or the step became negligible, check if the
      // KKT error was reduced. If it wasn't, invoke feasibility restoration.
      if (α < α_min || α < α_tiny) {
        Scalar current_kkt_error = kkt_error_one_norm<Scalar>(g, A_e, c_e, y);

        manifold.retract(x, α_max, step.p_x, trial_x);
        trial_y = y + α_max * step.p_y;

        trial_f = matrices.f(trial_x);
        trial_c_e = matrices.c_e(trial_x);

        Scalar next_kkt_error = kkt_error_one_norm<Scalar>(
            matrices.g(trial_x), matrices.A_e(trial_x), trial_c_e, trial_y);

        // If the step using αᵐᵃˣ reduced the KKT error below the current
        // iterate's and every previous fallback step's, accept it anyway
        if (next_kkt_error <= Scalar(0.999) * std::min(current_kkt_error,
                                                       fallback_kkt_error) &&
            constraint_violation(trial_c_e) <=
                filter.max_constraint_violation) {
          // Accept step
          fallback_kkt_error = next_kkt_error;
          α = α_max;
          break;
        }

        call_feasibility_restoration = true;
        break;
      }
    }

    phase_tracer.stop();

    // If solver stalled, invoke feasibility restoration
    if (!call_feasibility_restoration) {
      FilterEntry trial_entry{trial_f, trial_c_e};
      if (stall_detector.update(current_entry, trial_entry, α,
                                filter.min_constraint_violation)) {
        call_feasibility_restoration = true;
      }
    }

    if (call_feasibility_restoration) {
      ScopedTimeTracer feasibility_restoration_tracer{
          diagnostics[FEASIBILITY_RESTORATION]};

      FilterEntry initial_entry{matrices.f(x), c_e};

      // Augment the filter with the current iterate so the normal phase can't
      // return to it
      filter.add(initial_entry);

      feasibility_restoration_cycle_detector.record_entry(
          initial_entry.constraint_violation);
      const bool run_feasibility_restoration_to_convergence =
          feasibility_restoration_cycle_detector.is_cycling();

      // Feasibility restoration phase
      gch::small_vector<
          std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
          callbacks;
      for (auto& callback : iteration_callbacks) {
        callbacks.emplace_back(callback);
      }
      callbacks.emplace_back([&](const IterationInfo<Scalar>& info) {
        if (run_feasibility_restoration_to_convergence) {
          return ControlFlow::CONTINUE;
        }

        DenseVector trial_x = info.x.segment(0, x.rows());

        DenseVector trial_c_e = matrices.c_e(trial_x);

        FilterEntry trial_entry{matrices.f(trial_x), trial_c_e};

        // If the current iterate sufficiently reduces constraint violation and
        // is accepted by the normal filter, stop feasibility restoration
        if (trial_entry.constraint_violation <
                Scalar(0.9) * initial_entry.constraint_violation &&
            filter.is_acceptable(trial_entry)) {
          return ControlFlow::BREAK;
        } else {
          return ControlFlow::CONTINUE;
        }
      });
      auto status = feasibility_restoration<Scalar>(ambient_matrices, callbacks,
                                                    options, x, y, iterations);

      if (status != ExitStatus::SUCCESS) {
        // Report failure
        return status;
      }

      f = matrices.f(x);
      c_e = matrices.c_e(x);

      // Resuming from an iterate the filter rejects would break the filter's
      // convergence guarantee, and continuing feasibility restoration from its
      // converged point would return that point again, so report failure.
      //
      // The returned iterate can be rejected because feasibility restoration
      // stops at its own convergence tolerance instead of driving constraint
      // violation to zero, so the filter can still hold entries with lower cost
      // and constraint violation.
      if (!filter.is_acceptable(FilterEntry{f, c_e})) {
        return ExitStatus::FEASIBILITY_RESTORATION_FAILED;
      }
    } else {
      // If full step was accepted, reset full-step rejected counter
      if (α == α_max) {
        full_step_rejected_counter = 0;
      }

      // Update iterates
      x = trial_x;
      y = trial_y;

      f = trial_f;
      c_e = trial_c_e;
    }

    // Update autodiff for Jacobians and Hessian
    A_e = matrices.A_e(x);
    g = matrices.g(x);
    H = matrices.H(x, y);

    // Update the error
    E_0 = unscaled_kkt_error_inf_norm<Scalar>(matrices.scaling, g, A_e, c_e, y,
                                              options.tolerance);

    inner_iter_tracer.stop();

    if (options.diagnostics) {
      print_iterate_diagnostics(iterations, IterationType::NORMAL,
                                inner_iter_tracer.current_duration(), E_0, f,
                                constraint_violation(c_e), Scalar(0), Scalar(0),
                                kkt_system.hessian_regularization(),
                                kkt_system.constraint_jacobian_regularization(),
                                step.primal_inf_norm(), step.dual_inf_norm(), α,
                                α_max, α_reduction_factor, Scalar(1));
    }

    ++iterations;

    // Check for max iterations
    if (iterations >= options.max_iterations) {
      return ExitStatus::MAX_ITERATIONS_EXCEEDED;
    }

    // Check for max wall clock time
    if (std::chrono::steady_clock::now() - solve_start_time > options.timeout) {
      return ExitStatus::TIMEOUT;
    }
  }

  if (!isfinite(E_0)) {
    return ExitStatus::DIVERGING_ITERATES;
  } else {
    return ExitStatus::SUCCESS;
  }
}

extern template SLEIPNIR_DLLEXPORT ExitStatus
sqp(const SQPMatrixCallbacks<double>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options<double>& options, Eigen::VectorX<double>& x);

extern template SLEIPNIR_DLLEXPORT ExitStatus
sqp(const SQPMatrixCallbacks<double>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options<double>& options, Eigen::VectorX<double>& x,
    Eigen::VectorX<double>& y);

}  // namespace slp
