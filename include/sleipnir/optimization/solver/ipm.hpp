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
#include "sleipnir/optimization/solver/ipm_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/util/all_finite.hpp"
#include "sleipnir/optimization/solver/util/barrier_parameter.hpp"
#include "sleipnir/optimization/solver/util/constraint_violation.hpp"
#include "sleipnir/optimization/solver/util/feasibility_restoration.hpp"
#include "sleipnir/optimization/solver/util/feasibility_restoration_cycle_detector.hpp"
#include "sleipnir/optimization/solver/util/filter.hpp"
#include "sleipnir/optimization/solver/util/fraction_to_the_boundary_rule.hpp"
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
//
// See docs/algorithms.md#Interior-point_method for a derivation of the
// interior-point method formulation being used.

namespace slp {

/// Finds the optimal solution to a nonlinear program using the interior-point
/// method.
///
/// A nonlinear program has the form:
///
/// ```
///      min_x f(x)
/// subject to cₑ(x) = 0
///            cᵢ(x) ≥ 0
/// ```
///
/// where f(x) is the cost function, cₑ(x) are the equality constraints, and
/// cᵢ(x) are the inequality constraints.
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
ExitStatus ipm(
    const IPMMatrixCallbacks<Scalar>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options<Scalar>& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorX<Scalar>& x) {
  using DenseVector = Eigen::VectorX<Scalar>;

  // If cᵢ(x) is sufficiently feasible, start the constraint with zero
  // violation. Otherwise, make Σ = S⁻¹Z ≈ I.
  DenseVector c_i = matrix_callbacks.c_i(x);
  DenseVector s = (c_i.array() >= Scalar(1e-2)).select(c_i, Scalar(1));

  DenseVector y = DenseVector::Zero(matrix_callbacks.num_equality_constraints);
  DenseVector z =
      DenseVector::Ones(matrix_callbacks.num_inequality_constraints);
  Scalar μ = Scalar(0.1) * matrix_callbacks.scaling.f;
  int iterations = 0;

  return ipm(matrix_callbacks, iteration_callbacks, options, false,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
             bound_constraint_mask,
#endif
             x, s, y, z, μ, iterations);
}

/// Finds the optimal solution to a nonlinear program using the interior-point
/// method.
///
/// A nonlinear program has the form:
///
/// ```
///      min_x f(x)
/// subject to cₑ(x) = 0
///            cᵢ(x) ≥ 0
/// ```
///
/// where f(x) is the cost function, cₑ(x) are the equality constraints, and
/// cᵢ(x) are the inequality constraints.
///
/// @tparam Scalar Scalar type.
/// @param[in] matrix_callbacks Matrix callbacks.
/// @param[in] iteration_callbacks The list of callbacks to call at the
///     beginning of each iteration.
/// @param[in] options Solver options.
/// @param[in] in_feasibility_restoration Whether solver is in feasibility
///     restoration mode.
/// @param[in,out] x The initial guess and output location for the decision
///     variables.
/// @param[in,out] s The initial guess and output location for the inequality
///     constraint slack variables.
/// @param[in,out] y The initial guess and output location for the equality
///     constraint dual variables.
/// @param[in,out] z The initial guess and output location for the inequality
///     constraint dual variables.
/// @param[in,out] μ The initial guess and output location for the barrier
///     parameter.
/// @param[in,out] iterations The iteration counter.
/// @return The exit status.
template <typename Scalar>
ExitStatus ipm(
    const IPMMatrixCallbacks<Scalar>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options<Scalar>& options, bool in_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorX<Scalar>& x, Eigen::VectorX<Scalar>& s,
    Eigen::VectorX<Scalar>& y, Eigen::VectorX<Scalar>& z, Scalar& μ,
    int& iterations) {
  using DenseVector = Eigen::VectorX<Scalar>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using SparseVector = Eigen::SparseVector<Scalar>;

  using std::isfinite;

  const auto solve_start_time = std::chrono::steady_clock::now();

  using enum SolveTimeTracerID;

  // Prints final solver diagnostics when the solver exits
  //
  // Feasibility restoration still traces so its iterations can be printed, but
  // the outer solver prints the final solver diagnostics.
  SolveDiagnostics<ipm_time_trace> diagnostics{options.diagnostics,
                                               !in_feasibility_restoration};

  // Set up traced matrix callbacks if diagnostics are enabled. Feasibility
  // restoration's callbacks call the outer solver's, which are already traced.
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  std::optional<IPMMatrixCallbacks<Scalar>> traced_matrices;
  if (options.diagnostics && !in_feasibility_restoration) {
    traced_matrices.emplace(
        matrix_callbacks.num_decision_variables,
        matrix_callbacks.num_equality_constraints,
        matrix_callbacks.num_inequality_constraints,
        traced(diagnostics[COST], matrix_callbacks.f),
        traced(diagnostics[GRADIENT], matrix_callbacks.g),
        traced(diagnostics[HESSIAN], matrix_callbacks.H),
        traced(diagnostics[HESSIAN_CONSTRAINTS], matrix_callbacks.H_c),
        traced(diagnostics[EQUALITY_CONSTRAINTS], matrix_callbacks.c_e),
        traced(diagnostics[EQUALITY_JACOBIAN], matrix_callbacks.A_e),
        traced(diagnostics[INEQUALITY_CONSTRAINTS], matrix_callbacks.c_i),
        traced(diagnostics[INEQUALITY_JACOBIAN], matrix_callbacks.A_i),
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
  std::optional<IPMMatrixCallbacks<Scalar>> tangent_matrices;
  if (!manifold.is_euclidean()) {
    tangent_matrices.emplace(tangent_space.callbacks(ambient_matrices));
  }
  const auto& matrices =
      tangent_matrices ? *tangent_matrices : ambient_matrices;

  ScopedTimeTracer setup_tracer{diagnostics[SETUP]};

  Scalar f = matrices.f(x);
  SparseVector g = matrices.g(x);
  SparseMatrix H = matrices.H(x, y, z);
  DenseVector c_e = matrices.c_e(x);
  SparseMatrix A_e = matrices.A_e(x);
  DenseVector c_i = matrices.c_i(x);
  SparseMatrix A_i = matrices.A_i(x);

  // Ensure matrix callback dimensions are consistent
  slp_assert(g.rows() == matrices.num_decision_variables);
  slp_assert(H.rows() == matrices.num_decision_variables);
  slp_assert(H.cols() == matrices.num_decision_variables);
  slp_assert(c_e.rows() == matrices.num_equality_constraints);
  slp_assert(A_e.rows() == matrices.num_equality_constraints);
  slp_assert(A_e.cols() == matrices.num_decision_variables);
  slp_assert(c_i.rows() == matrices.num_inequality_constraints);
  slp_assert(A_i.rows() == matrices.num_inequality_constraints);
  slp_assert(A_i.cols() == matrices.num_decision_variables);

  DenseVector trial_x;
  DenseVector trial_s;
  DenseVector trial_y;
  DenseVector trial_z;

  Scalar trial_f;
  DenseVector trial_c_e;
  DenseVector trial_c_i;

  // Check for overconstrained problem
  if (matrices.num_equality_constraints > matrices.num_decision_variables) {
    if (options.diagnostics) {
      print_too_few_dofs_error(c_e);
    }

    return ExitStatus::TOO_FEW_DOFS;
  }

  // Check whether initial guess has finite cost, constraints, and derivatives
  if (!isfinite(f) || !all_finite(g) || !all_finite(H) || !c_e.allFinite() ||
      !all_finite(A_e) || !c_i.allFinite() || !all_finite(A_i)) {
    return ExitStatus::NONFINITE_INITIAL_GUESS;
  }

#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
  // We set sʲ = cᵢʲ(x) for each bound inequality constraint index j
  s = bound_constraint_mask.select(c_i, s);
#endif

  // Barrier parameter minimum
  const Scalar μ_min = matrices.scaling.f * options.tolerance / Scalar(10);

  // Fraction-to-the-boundary rule scale factor minimum
  constexpr Scalar τ_min(0.99);

  // Fraction-to-the-boundary rule scale factor τ
  Scalar τ = τ_min;

  Filter<Scalar> filter{constraint_violation(c_e, c_i, s)};

  // The smallest KKT error of a step accepted by the KKT error reduction
  // fallback for the current barrier parameter. Each fallback step must improve
  // on it, so finitely many fallback steps are taken before the barrier
  // subproblem converges.
  Scalar fallback_kkt_error = std::numeric_limits<Scalar>::infinity();

  // This should be run when the error is below a desired threshold for the
  // current barrier parameter
  auto update_barrier_parameter_and_reset_filter = [&] {
    μ = next_barrier_parameter(μ, μ_min);

    // Update the fraction-to-the-boundary rule scaling factor.
    //
    //   τⱼ = max(τₘᵢₙ, 1 − μⱼ)
    //
    // See equation (8) of [2].
    τ = std::max(τ_min, Scalar(1) - μ);

    // Reset the filter when the barrier parameter is updated
    filter.reset();
    fallback_kkt_error = std::numeric_limits<Scalar>::infinity();
  };

  IPMKKTSystem<Scalar> kkt_system{H, A_e, A_i};

  // Step size reduction factor
  constexpr Scalar α_reduction_factor(0.5);

  int full_step_rejected_counter = 0;

  StallDetector<Scalar> stall_detector;

  FeasibilityRestorationCycleDetector<Scalar>
      feasibility_restoration_cycle_detector;

  // Error
  Scalar E_0 = unscaled_kkt_error_inf_norm<Scalar>(
      matrices.scaling, g, A_e, c_e, A_i, c_i, s, y, z, Scalar(0),
      options.tolerance);

  setup_tracer.stop();

  // Prints initial iterate diagnostics now and closes the iterate diagnostics
  // table when the solver exits.
  //
  // Feasibility restoration's iterations are printed in the outer solver's
  // table, so it doesn't open its own.
  IterateDiagnosticsTable table{
      options.diagnostics && !in_feasibility_restoration,
      E_0,
      f,
      constraint_violation(c_e, c_i, s),
      s.dot(z),
      μ};

  while (E_0 > options.tolerance) {
    ScopedTimeTracer inner_iter_tracer{diagnostics[ITERATION]};

    // Check for diverging iterates
    if (x.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !x.allFinite() ||
        s.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !s.allFinite()) {
      return ExitStatus::DIVERGING_ITERATES;
    }

    PhaseTimeTracer phase_tracer{diagnostics[CALLBACKS]};

    // Call iteration callbacks
    for (const auto& callback : iteration_callbacks) {
      if (callback({iterations, x, s, y, z, g, H, A_e, A_i}) ==
          ControlFlow::BREAK) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    phase_tracer.next(diagnostics[KKT_MATRIX_BUILD]);

    kkt_system.build(H, A_e, A_i, s, z);

    phase_tracer.next(diagnostics[KKT_MATRIX_FACTOR]);

    if (!kkt_system.factorize()) [[unlikely]] {
      return ExitStatus::FACTORIZATION_FAILED;
    }

    phase_tracer.next(diagnostics[KKT_SYSTEM_SOLVE]);

    auto step = kkt_system.solve(g, A_e, A_i, y, s, z, μ, c_e, c_i - s);

    phase_tracer.next(diagnostics[LINE_SEARCH]);

    bool call_feasibility_restoration = false;

    // αᵐᵃˣ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
    Scalar α_max = fraction_to_the_boundary_rule<Scalar>(s, step.p_s, τ);
    Scalar α = α_max;

    // αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
    Scalar α_z = fraction_to_the_boundary_rule<Scalar>(z, step.p_z, τ);

    const FilterEntry<Scalar> current_entry{f, s, c_e, c_i, μ};

    // Compute the directional derivative of the log-barrier function along the
    // search direction.
    //
    //   ϕ_μ(x, s) = f(x) − μ∑ᵢ ln(sᵢ)
    //
    //   D_ϕ = ∇ϕ_μ(x, s)ᵀ[pˣ pˢ]
    //       = ∇f(x)ᵀpˣ − μ∑ᵢ pᵢˢ/sᵢ
    const Scalar D_ϕ =
        g.transpose() * step.p_x - μ * s.cwiseInverse().dot(step.p_s);

    const Scalar α_min = filter.min_step_size(current_entry, D_ϕ);

    // If maximum step size is below minimum, invoke feasibility restoration
    if (α < α_min) {
      call_feasibility_restoration = true;
    }

    // α_min can be zero (e.g., when the constraint violation is zero), so also
    // end the line search when the step becomes negligible relative to the
    // iterate. This bounds the number of step size reductions.
    const Scalar α_tiny = step.tiny_step_size(x, s, manifold);

    // Loop until a step is accepted
    while (!call_feasibility_restoration) {
      manifold.retract(x, α, step.p_x, trial_x);
      trial_c_i = matrices.c_i(trial_x);
      if (options.feasible_ipm && c_i.cwiseGreater(Scalar(0)).all()) {
        // If the inequality constraints are all feasible, prevent them from
        // becoming infeasible again.
        //
        // See equation (19.30) of [1].
        trial_s = trial_c_i;
      } else {
        trial_s = s + α * step.p_s;
      }
      trial_y = y + α * step.p_y;
      trial_z = z + α_z * step.p_z;

      trial_f = matrices.f(trial_x);
      trial_c_e = matrices.c_e(trial_x);

      // If f(xₖ + αpₖˣ), cₑ(xₖ + αpₖˣ), or cᵢ(xₖ + αpₖˣ) aren't finite, reduce
      // step size immediately
      if (!isfinite(trial_f) || !trial_c_e.allFinite() ||
          !trial_c_i.allFinite()) {
        // Reduce step size
        α *= α_reduction_factor;

        if (α < α_min || α < α_tiny) {
          call_feasibility_restoration = true;
          break;
        }
        continue;
      }

      // Check whether filter accepts trial iterate
      FilterEntry trial_entry{trial_f, trial_s, trial_c_e, trial_c_i, μ};
      if (filter.try_add(current_entry, trial_entry, D_ϕ, α)) {
        // Accept step
        break;
      }

      Scalar prev_constraint_violation = constraint_violation(c_e, c_i, s);
      Scalar next_constraint_violation =
          constraint_violation(trial_c_e, trial_c_i, trial_s);

      // Second-order corrections
      //
      // If first trial point was rejected and constraint violation stayed the
      // same or went up, apply second-order corrections
      if (α == α_max &&
          next_constraint_violation >= prev_constraint_violation) {
        // Apply second-order corrections. See section 2.4 of [2].
        auto soc_step = step;

        Scalar α_soc = α;
        Scalar α_z_soc = α_z;
        DenseVector c_e_soc = c_e;
        DenseVector c_i_minus_s_soc = c_i - s;

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
                  unscaled_kkt_error_inf_norm<Scalar>(
                      matrices.scaling, g, A_e, trial_c_e, A_i, trial_c_i,
                      trial_s, trial_y, trial_z, Scalar(0), options.tolerance),
                  trial_f, constraint_violation(trial_c_e, trial_c_i, trial_s),
                  trial_s.dot(trial_z), μ, kkt_system.hessian_regularization(),
                  kkt_system.constraint_jacobian_regularization(),
                  soc_step.primal_inf_norm(), soc_step.dual_inf_norm(), α_soc,
                  Scalar(1), α_reduction_factor, α_z_soc);
            }
          }};

          // Solve the Newton-KKT system with corrected constraint values
          //
          //   cₑˢᵒᶜ = αˢᵒᶜcₑ(xₖ) + cₑ(xₖ + αˢᵒᶜpˣˢᵒᶜ)
          //   (cᵢ − s)ˢᵒᶜ =
          //     αˢᵒᶜ(cᵢ(xₖ) − sₖ) + cᵢ(xₖ + αˢᵒᶜpˣˢᵒᶜ) − (sₖ + αˢᵒᶜpˢˢᵒᶜ)
          c_e_soc = α_soc * c_e_soc + trial_c_e;
          c_i_minus_s_soc = α_soc * c_i_minus_s_soc + trial_c_i - trial_s;
          soc_step = kkt_system.solve(g, A_e, A_i, y, s, z, μ, c_e_soc,
                                      c_i_minus_s_soc);

          // αˢᵒᶜ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
          // αₖᶻˢᵒᶜ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
          α_soc = fraction_to_the_boundary_rule<Scalar>(s, soc_step.p_s, τ);
          α_z_soc = fraction_to_the_boundary_rule<Scalar>(z, soc_step.p_z, τ);

          manifold.retract(x, α_soc, soc_step.p_x, trial_x);
          trial_s = s + α_soc * soc_step.p_s;
          trial_y = y + α_soc * soc_step.p_y;
          trial_z = z + α_z_soc * soc_step.p_z;

          trial_f = matrices.f(trial_x);
          trial_c_e = matrices.c_e(trial_x);
          trial_c_i = matrices.c_i(trial_x);

          // Check whether filter accepts trial iterate
          FilterEntry trial_entry{trial_f, trial_s, trial_c_e, trial_c_i, μ};
          if (filter.try_add(current_entry, trial_entry, D_ϕ, α)) {
            step = soc_step;
            α = α_soc;
            α_z = α_z_soc;
            step_acceptable = true;
            break;
          }

          // Constraint violation scale factor for second-order corrections
          constexpr Scalar κ_soc(0.99);

          // If constraint violation hasn't been sufficiently reduced, stop
          // making second-order corrections
          next_constraint_violation =
              constraint_violation(trial_c_e, trial_c_i, trial_s);
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
        Scalar current_kkt_error =
            kkt_error_one_norm<Scalar>(g, A_e, c_e, A_i, c_i, s, y, z, μ);

        manifold.retract(x, α_max, step.p_x, trial_x);
        trial_s = s + α_max * step.p_s;
        trial_y = y + α_max * step.p_y;
        trial_z = z + α_z * step.p_z;

        trial_f = matrices.f(trial_x);
        trial_c_e = matrices.c_e(trial_x);
        trial_c_i = matrices.c_i(trial_x);

        Scalar next_kkt_error = kkt_error_one_norm<Scalar>(
            matrices.g(trial_x), matrices.A_e(trial_x), trial_c_e,
            matrices.A_i(trial_x), trial_c_i, trial_s, trial_y, trial_z, μ);

        // If the step using αᵐᵃˣ reduced the KKT error below the current
        // iterate's and every previous fallback step's, accept it anyway
        if (next_kkt_error <= Scalar(0.999) * std::min(current_kkt_error,
                                                       fallback_kkt_error) &&
            constraint_violation(trial_c_e, trial_c_i, trial_s) <=
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
    if (!call_feasibility_restoration && !in_feasibility_restoration) {
      FilterEntry trial_entry{trial_f, trial_s, trial_c_e, trial_c_i, μ};
      if (stall_detector.update(current_entry, trial_entry, α,
                                filter.min_constraint_violation)) {
        call_feasibility_restoration = true;
      }
    }

    if (call_feasibility_restoration) {
      ScopedTimeTracer feasibility_restoration_tracer{
          diagnostics[FEASIBILITY_RESTORATION]};

      // If already in feasibility restoration mode, running it again won't help
      if (in_feasibility_restoration) {
        return ExitStatus::FEASIBILITY_RESTORATION_FAILED;
      }

      FilterEntry initial_entry{matrices.f(x), s, c_e, c_i, μ};

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
        DenseVector trial_c_i = matrices.c_i(trial_x);

        // Feasibility restoration re-centers the slacks when it returns, so
        // check the iterate the normal phase will resume from
        DenseVector trial_s =
            (trial_c_i.array() >= Scalar(1e-2))
                .select(trial_c_i,
                        info.s.segment(0, matrices.num_inequality_constraints));

        // If the current iterate sufficiently reduces constraint violation and
        // is accepted by the normal filter, stop feasibility restoration
        FilterEntry trial_entry{matrices.f(trial_x), trial_s, trial_c_e,
                                trial_c_i, μ};
        if (trial_entry.constraint_violation <
                Scalar(0.9) * initial_entry.constraint_violation &&
            filter.is_acceptable(trial_entry)) {
          return ControlFlow::BREAK;
        } else {
          return ControlFlow::CONTINUE;
        }
      });
      auto status =
          feasibility_restoration<Scalar>(ambient_matrices, callbacks, options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
                                          bound_constraint_mask,
#endif
                                          x, s, y, z, μ, iterations);

      if (status != ExitStatus::SUCCESS) {
        // Report failure
        return status;
      }

      f = matrices.f(x);
      c_e = matrices.c_e(x);
      c_i = matrices.c_i(x);

      // Resuming from an iterate the filter rejects would break the filter's
      // convergence guarantee, and continuing feasibility restoration from its
      // converged point would return that point again, so report failure.
      //
      // The returned iterate can be rejected because feasibility restoration
      // stops at its own convergence tolerance instead of driving constraint
      // violation to zero, so the filter can still hold entries with lower cost
      // and constraint violation.
      if (!filter.is_acceptable(FilterEntry{f, s, c_e, c_i, μ})) {
        return ExitStatus::FEASIBILITY_RESTORATION_FAILED;
      }
    } else {
      // If full step was accepted, reset full-step rejected counter
      if (α == α_max) {
        full_step_rejected_counter = 0;
      }

      // Update iterates
      x = trial_x;
      s = trial_s;
      y = trial_y;
      z = trial_z;

      f = trial_f;
      c_e = trial_c_e;
      c_i = trial_c_i;
    }

    // A requirement for the convergence proof is that the primal-dual barrier
    // term Hessian Σₖ₊₁ does not deviate arbitrarily much from the primal
    // barrier term Hessian μSₖ₊₁⁻².
    //
    //   Σₖ₊₁ = μSₖ₊₁⁻²
    //   Sₖ₊₁⁻¹Zₖ₊₁ = μSₖ₊₁⁻²
    //   Zₖ₊₁ = μSₖ₊₁⁻¹
    //
    // We ensure this by resetting
    //
    //   zₖ₊₁ = clamp(zₖ₊₁, 1/κ_Σ μ/sₖ₊₁, κ_Σ μ/sₖ₊₁)
    //
    // for some fixed κ_Σ ≥ 1 after each step. See equation (16) of [2].
    for (int row = 0; row < z.rows(); ++row) {
      constexpr Scalar κ_Σ(1e10);
      z[row] =
          std::clamp(z[row], Scalar(1) / κ_Σ * μ / s[row], κ_Σ * μ / s[row]);
    }

    // Update autodiff for Jacobians and Hessian
    A_e = matrices.A_e(x);
    A_i = matrices.A_i(x);
    g = matrices.g(x);
    H = matrices.H(x, y, z);

    // Update the error
    E_0 = unscaled_kkt_error_inf_norm<Scalar>(matrices.scaling, g, A_e, c_e,
                                              A_i, c_i, s, y, z, Scalar(0),
                                              options.tolerance);

    // Update the barrier parameter if necessary
    if (E_0 > options.tolerance) {
      // While the error is below the desired threshold for this barrier
      // parameter value, decrease the barrier parameter further
      Scalar E_μ = kkt_error_inf_norm<Scalar>(g, A_e, c_e, A_i, c_i, s, y, z, μ,
                                              options.tolerance);
      while (μ > μ_min && E_μ <= barrier_subproblem_tolerance(μ)) {
        update_barrier_parameter_and_reset_filter();
        E_μ = kkt_error_inf_norm<Scalar>(g, A_e, c_e, A_i, c_i, s, y, z, μ,
                                         options.tolerance);
      }
    }

    inner_iter_tracer.stop();

    if (options.diagnostics) {
      print_iterate_diagnostics(iterations,
                                in_feasibility_restoration
                                    ? IterationType::FEASIBILITY_RESTORATION
                                    : IterationType::NORMAL,
                                inner_iter_tracer.current_duration(), E_0, f,
                                constraint_violation(c_e, c_i, s), s.dot(z), μ,
                                kkt_system.hessian_regularization(),
                                kkt_system.constraint_jacobian_regularization(),
                                step.primal_inf_norm(), step.dual_inf_norm(), α,
                                α_max, α_reduction_factor, α_z);
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
ipm(const IPMMatrixCallbacks<double>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options<double>& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorX<double>& x);

extern template SLEIPNIR_DLLEXPORT ExitStatus
ipm(const IPMMatrixCallbacks<double>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options<double>& options, bool in_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorX<double>& x, Eigen::VectorX<double>& s,
    Eigen::VectorX<double>& y, Eigen::VectorX<double>& z, double& μ,
    int& iterations);

}  // namespace slp
