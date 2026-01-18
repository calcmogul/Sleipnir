// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>
#include <cmath>
#include <functional>
#include <optional>
#include <span>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/sqp_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/util/all_finite.hpp"
#include "sleipnir/optimization/solver/util/append_as_triplets.hpp"
#include "sleipnir/optimization/solver/util/constraint_violation.hpp"
#include "sleipnir/optimization/solver/util/feasibility_restoration.hpp"
#include "sleipnir/optimization/solver/util/filter.hpp"
#include "sleipnir/optimization/solver/util/kkt_error.hpp"
#include "sleipnir/optimization/solver/util/kkt_solver.hpp"
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
ExitStatus sqp(const SQPMatrixCallbacks<Scalar>& matrix_callbacks,
               std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
                   iteration_callbacks,
               const Options& options,
               Eigen::Vector<Scalar, Eigen::Dynamic>& x) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  DenseVector y = DenseVector::Zero(matrix_callbacks.num_equality_constraints);
  int iterations = 0;

  return sqp(matrix_callbacks, iteration_callbacks, options, x, y, iterations);
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
/// @param[in,out] iterations The iteration counter.
/// @return The exit status.
template <typename Scalar>
ExitStatus sqp(const SQPMatrixCallbacks<Scalar>& matrix_callbacks,
               std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
                   iteration_callbacks,
               const Options& options, Eigen::Vector<Scalar, Eigen::Dynamic>& x,
               Eigen::Vector<Scalar, Eigen::Dynamic>& y, int& iterations) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using SparseVector = Eigen::SparseVector<Scalar>;

  /// SQP step direction.
  struct Step {
    /// Decision variable primal step.
    DenseVector p_x;
    /// Equality constraint dual step.
    DenseVector p_y;

    /// Returns the infinity norm of the primal step.
    ///
    /// @return The infinity norm of the primal step.
    Scalar primal_inf_norm() const {
      return p_x.template lpNorm<Eigen::Infinity>();
    }

    /// Returns the infinity norm of the dual step.
    ///
    /// @return The infinity norm of the dual step.
    Scalar dual_inf_norm() const {
      return p_y.template lpNorm<Eigen::Infinity>();
    }
  };

  using std::isfinite;

  const auto solve_start_time = std::chrono::steady_clock::now();

  using enum SolveTimeTracerID;

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
        matrix_callbacks.scaling);
  }
  const auto& matrices = traced_matrices ? *traced_matrices : matrix_callbacks;
#else
  const auto& matrices = matrix_callbacks;
#endif

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

  // Kept outside the loop so its storage can be reused
  gch::small_vector<Eigen::Triplet<Scalar>> triplets;

  const int lhs_rows =
      matrices.num_decision_variables + matrices.num_equality_constraints;
  KKTSolver<Scalar> solver{
      // Use sparse solver if lower triangle fills < 25% of system
      H.nonZeros() + A_e.nonZeros() < 0.25 * lhs_rows * lhs_rows,
      matrices.num_decision_variables, matrices.num_equality_constraints};

  // Variables for determining when a step is acceptable
  constexpr Scalar α_reduction_factor(0.5);
  constexpr Scalar α_min(1e-7);

  int full_step_rejected_counter = 0;

  StallDetector<Scalar> stall_detector;

  // Error
  Scalar E_0 = unscaled_kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
      matrices.scaling, g, A_e, c_e, y);

  setup_tracer.stop();

  // Prints initial iterate diagnostics now and closes the iterate diagnostics
  // table when the solver exits
  IterateDiagnosticsTable table{
      options.diagnostics,       E_0,       f,
      constraint_violation(c_e), Scalar(0), Scalar(0)};

  while (E_0 > Scalar(options.tolerance)) {
    ScopedTimeTracer inner_iter_tracer{diagnostics[ITERATION]};

    // Check for diverging iterates
    if (x.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !x.allFinite()) {
      return ExitStatus::DIVERGING_ITERATES;
    }

    PhaseTimeTracer phase_tracer{diagnostics[CALLBACKS]};

    // Call iteration callbacks
    for (const auto& callback : iteration_callbacks) {
      if (callback({iterations, x, {}, y, {}, g, H, A_e, {}})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    phase_tracer.next(diagnostics[KKT_MATRIX_BUILD]);

    // lhs = [H   Aₑᵀ]
    //       [Aₑ   0 ]
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    triplets.clear();
    triplets.reserve(H.nonZeros() + A_e.nonZeros());
    append_as_triplets(triplets, 0, 0, {H, A_e});
    SparseMatrix lhs(
        matrices.num_decision_variables + matrices.num_equality_constraints,
        matrices.num_decision_variables + matrices.num_equality_constraints);
    lhs.setFromSortedTriplets(triplets.begin(), triplets.end());

    // rhs = −[∇f − Aₑᵀy]
    //        [   cₑ    ]
    DenseVector rhs{x.rows() + y.rows()};
    rhs.segment(0, x.rows()) = -g + A_e.transpose() * y;
    rhs.segment(x.rows(), y.rows()) = -c_e;

    phase_tracer.next(diagnostics[KKT_MATRIX_DECOMP]);

    Step step;
    constexpr Scalar α_max(1);
    Scalar α(1);
    bool call_feasibility_restoration = false;

    // Solve the Newton-KKT system
    //
    // [H   Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy]
    // [Aₑ   0 ][−pʸ]    [   cₑ    ]
    if (solver.compute(lhs).info() != Eigen::Success) [[unlikely]] {
      return ExitStatus::FACTORIZATION_FAILED;
    }

    phase_tracer.next(diagnostics[KKT_SYSTEM_SOLVE]);

    auto compute_step = [&](Step& step) {
      // p = [ pˣ]
      //     [−pʸ]
      DenseVector p = solver.solve(rhs);
      step.p_x = p.segment(0, x.rows());
      step.p_y = -p.segment(x.rows(), y.rows());
    };
    compute_step(step);

    phase_tracer.next(diagnostics[LINE_SEARCH]);

    α = α_max;

    const FilterEntry<Scalar> current_entry{f, c_e};
    const Scalar D_ϕ = g.transpose() * step.p_x;

    // Loop until a step is accepted
    while (1) {
      trial_x = x + α * step.p_x;
      trial_y = y + α * step.p_y;

      trial_f = matrices.f(trial_x);
      trial_c_e = matrices.c_e(trial_x);

      // If f(xₖ + αpₖˣ) or cₑ(xₖ + αpₖˣ) aren't finite, reduce step size
      // immediately
      if (!isfinite(trial_f) || !trial_c_e.allFinite()) {
        // Reduce step size
        α *= α_reduction_factor;

        if (α < α_min) {
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
                  kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
                      g, A_e, trial_c_e, trial_y),
                  trial_f, constraint_violation(trial_c_e), Scalar(0),
                  Scalar(0), solver.hessian_regularization(),
                  solver.constraint_jacobian_regularization(),
                  soc_step.primal_inf_norm(), soc_step.dual_inf_norm(), α_soc,
                  Scalar(1), α_reduction_factor, Scalar(1));
            }
          }};

          // Rebuild Newton-KKT rhs with updated constraint values.
          //
          // rhs = −[∇f − Aₑᵀy]
          //        [  cₑˢᵒᶜ  ]
          //
          // where cₑˢᵒᶜ = αc(xₖ) + c(xₖ + αpₖˣ)
          c_e_soc = α_soc * c_e_soc + trial_c_e;
          rhs.bottomRows(y.rows()) = -c_e_soc;

          // Solve the Newton-KKT system
          compute_step(soc_step);

          trial_x = x + α_soc * soc_step.p_x;
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
      // because it may be impeding progress.
      //
      // See section 3.2 case I of [2].
      if (full_step_rejected_counter >= 4 &&
          filter.max_constraint_violation >
              current_entry.constraint_violation / Scalar(10) &&
          filter.last_rejection_due_to_filter()) {
        filter.max_constraint_violation *= Scalar(0.1);
        filter.reset();
        continue;
      }

      // Reduce step size
      α *= α_reduction_factor;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, invoke feasibility restoration.
      if (α < α_min) {
        Scalar current_kkt_error =
            kkt_error<Scalar, KKTErrorType::ONE_NORM>(g, A_e, c_e, y);

        trial_x = x + α_max * step.p_x;
        trial_y = y + α_max * step.p_y;

        trial_f = matrices.f(trial_x);
        trial_c_e = matrices.c_e(trial_x);

        Scalar next_kkt_error = kkt_error<Scalar, KKTErrorType::ONE_NORM>(
            matrices.g(trial_x), matrices.A_e(trial_x), trial_c_e, trial_y);

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (next_kkt_error <= Scalar(0.999) * current_kkt_error &&
            constraint_violation(trial_c_e) <=
                filter.max_constraint_violation) {
          // Accept step
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

      // Feasibility restoration phase
      gch::small_vector<std::function<bool(const IterationInfo<Scalar>& info)>>
          callbacks;
      for (auto& callback : iteration_callbacks) {
        callbacks.emplace_back(callback);
      }
      callbacks.emplace_back([&](const IterationInfo<Scalar>& info) {
        DenseVector trial_x =
            info.x.segment(0, matrices.num_decision_variables);

        DenseVector trial_c_e = matrices.c_e(trial_x);

        FilterEntry trial_entry{matrices.f(trial_x), trial_c_e};
        const Scalar D_ϕ_restoration = g.transpose() * (trial_x - x);

        // If the current iterate sufficiently reduces constraint violation and
        // is accepted by the normal filter, stop feasibility restoration
        return trial_entry.constraint_violation <
                   Scalar(0.9) * initial_entry.constraint_violation &&
               filter.try_add(initial_entry, trial_entry, D_ϕ_restoration,
                              Scalar(1));
      });
      auto status = feasibility_restoration<Scalar>(matrices, callbacks,
                                                    options, x, y, iterations);

      if (status != ExitStatus::SUCCESS) {
        // Report failure
        return status;
      }

      f = matrices.f(x);
      c_e = matrices.c_e(x);
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
    E_0 = unscaled_kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
        matrices.scaling, g, A_e, c_e, y);

    inner_iter_tracer.stop();

    if (options.diagnostics) {
      print_iterate_diagnostics(iterations, IterationType::NORMAL,
                                inner_iter_tracer.current_duration(), E_0, f,
                                constraint_violation(c_e), Scalar(0), Scalar(0),
                                solver.hessian_regularization(),
                                solver.constraint_jacobian_regularization(),
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
    std::span<std::function<bool(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options& options, Eigen::Vector<double, Eigen::Dynamic>& x);

}  // namespace slp
