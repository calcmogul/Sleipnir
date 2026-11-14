// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>  // IWYU pragma: keep
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <utility>

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
#include "sleipnir/optimization/solver/util/first_derivatives.hpp"
#include "sleipnir/optimization/solver/util/function_values.hpp"
#include "sleipnir/optimization/solver/util/iterates.hpp"
#include "sleipnir/optimization/solver/util/kkt_error.hpp"
#include "sleipnir/optimization/solver/util/kkt_system.hpp"
#include "sleipnir/optimization/solver/util/step.hpp"
#include "sleipnir/optimization/solver/util/time_trace.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/print_diagnostics.hpp"
#include "sleipnir/util/scope_exit.hpp"
#include "sleipnir/util/symbol_exports.hpp"
#include "sleipnir/util/time_tracer.hpp"

// See docs/algorithms.md#Works_cited for citation definitions.
//
// See docs/algorithms.md#Log-domain_interior-point_method for a derivation of
// the interior-point method formulation being used.

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
/// @param[in] is_nlp If true, the solver uses a more conservative barrier
///     parameter reduction strategy that's more reliable on NLPs. Pass false
///     for problems with quadratic or lower-order cost and linear or
///     lower-order constraints.
/// @param[in] iteration_callbacks The list of callbacks to call at the
///     beginning of each iteration.
/// @param[in] options Solver options.
/// @param[in,out] x The initial guess and output location for the decision
///     variables.
/// @return The exit status.
template <typename Scalar>
ExitStatus ipm(
    const IPMMatrixCallbacks<Scalar>& matrix_callbacks, bool is_nlp,
    std::span<std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options<Scalar>& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorX<Scalar>& x) {
  using DenseVector = Eigen::VectorX<Scalar>;

  IPMIterates<Scalar> iterates{
      .x = std::move(x),
      .y = DenseVector::Zero(matrix_callbacks.num_equality_constraints),
      .v = DenseVector::Zero(matrix_callbacks.num_inequality_constraints)};
  Scalar sqrt_μ = Scalar(0.1) * matrix_callbacks.scaling.f;
  int iterations = 0;

  const auto status =
      ipm(matrix_callbacks, is_nlp, iteration_callbacks, options, false,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
          bound_constraint_mask,
#endif
          iterates, sqrt_μ, iterations);

  x = std::move(iterates.x);

  return status;
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
/// @param[in] is_nlp If true, the solver uses a more conservative barrier
///     parameter reduction strategy that's more reliable on NLPs. Pass false
///     for problems with quadratic or lower-order cost and linear or
///     lower-order constraints.
/// @param[in] iteration_callbacks The list of callbacks to call at the
///     beginning of each iteration.
/// @param[in] options Solver options.
/// @param[in] in_feasibility_restoration Whether solver is in feasibility
///     restoration mode.
/// @param[in,out] iterates The initial guess and output location for the
///     iterates.
/// @param[in,out] sqrt_μ The initial guess and output location for the barrier
///     parameter.
/// @param[in,out] iterations The iteration counter.
/// @return The exit status.
template <typename Scalar>
ExitStatus ipm(
    const IPMMatrixCallbacks<Scalar>& matrix_callbacks, bool is_nlp,
    std::span<std::function<ControlFlow(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options<Scalar>& options, bool in_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    IPMIterates<Scalar>& iterates, Scalar& sqrt_μ, int& iterations) {
  using DenseVector = Eigen::VectorX<Scalar>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  using std::isfinite;
  using std::sqrt;

  auto& [x, y, v] = iterates;

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
        matrix_callbacks.scaling);
  }
  const auto& matrices = traced_matrices ? *traced_matrices : matrix_callbacks;
#else
  const auto& matrices = matrix_callbacks;
#endif

  ScopedTimeTracer setup_tracer{diagnostics[SETUP]};

  IPMFunctionValues<Scalar> function_values{
      .f = matrices.f(x), .c_e = matrices.c_e(x), .c_i = matrices.c_i(x)};
  IPMFirstDerivatives<Scalar> first_derivatives{
      .g = matrices.g(x), .A_e = matrices.A_e(x), .A_i = matrices.A_i(x)};
  auto& [f, c_e, c_i] = function_values;
  auto& [g, A_e, A_i] = first_derivatives;
  SparseMatrix H = matrices.H(x, y, v, sqrt_μ);

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

  IPMIterates<Scalar> trial_iterates;
  auto& [trial_x, trial_y, trial_v] = trial_iterates;

  IPMFunctionValues<Scalar> trial_function_values;
  auto& [trial_f, trial_c_e, trial_c_i] = trial_function_values;

  // Check for overconstrained problem
  if (matrices.num_equality_constraints > matrices.num_decision_variables) {
    if (options.diagnostics) {
      print_too_few_dofs_error(c_e);
    }

    return ExitStatus::TOO_FEW_DOFS;
  }

  // Check whether initial guess has finite cost, constraints, and derivatives
  if (!function_values.all_finite() || !first_derivatives.all_finite() ||
      !all_finite(H)) {
    return ExitStatus::NONFINITE_INITIAL_GUESS;
  }

  // Barrier parameter minimum
  const Scalar sqrt_μ_min =
      matrices.scaling.f * sqrt(options.tolerance / Scalar(10));

#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
  // We set sʲ = cᵢʲ(x) for each bound inequality constraint index j
  //
  //   cᵢ − √(μ)e⁻ᵛ = 0
  //   √(μ)e⁻ᵛ = cᵢ
  //   e⁻ᵛ = 1/√(μ) cᵢ
  //   −v = ln(1/√(μ) cᵢ)
  //   v = −ln(1/√(μ) cᵢ)
  v = bound_constraint_mask.select(
      -(c_i * (Scalar(1) / sqrt_μ_min)).array().log().matrix(), v);
#endif

  // eᵛ
  DenseVector exp_v{v.array().exp().matrix()};
  // e⁻ᵛ
  DenseVector exp_neg_v = exp_v.cwiseInverse();
  // s = √(μ)e⁻ᵛ
  DenseVector s = sqrt_μ * exp_neg_v;

  Filter<Scalar> filter{constraint_violation(c_e, c_i, s)};

  // The smallest KKT error of a step accepted by the KKT error reduction
  // fallback for the current barrier parameter. Each fallback step must improve
  // on it, so finitely many fallback steps are taken before the barrier
  // subproblem converges.
  Scalar fallback_kkt_error = std::numeric_limits<Scalar>::infinity();

  IPMKKTSystem<Scalar> kkt_system{H, A_e, A_i};

  setup_tracer.stop();

  auto build_and_factorize_kkt_matrix = [&]() -> bool {
    PhaseTimeTracer phase_tracer{diagnostics[KKT_MATRIX_BUILD]};

    kkt_system.build(H, A_e, A_i, exp_v);

    phase_tracer.next(diagnostics[KKT_MATRIX_FACTOR]);

    return kkt_system.factorize();
  };

  // Solves the Newton-KKT system for the step with the given √(μ)
  auto compute_step = [&](Scalar r) -> IPMStep<Scalar> {
    return kkt_system.solve(g, A_e, A_i, y, r, c_e, c_i);
  };

  // The step in v is taken as a linear step in u = z − s, then mapped back onto
  // the central path Sz = μe with the softplus map
  //
  //   z = (u + √(u² + 4μ))/2
  //   s = (−u + √(u² + 4μ))/2
  //
  // from [4]. Since s = √(μ)e⁻ᵛ and z = √(μ)eᵛ, u = 2√(μ)sinh(v), and the
  // Newton step in u is pᵘ = (z + s)∘pᵛ = 2√(μ)cosh(v)∘pᵛ. Therefore,
  //
  //   u⁺ = u + αpᵘ
  //   v⁺ = asinh(u⁺/(2√(μ)))
  //      = asinh(sinh(v) + α cosh(v)∘pᵛ)
  //
  // The step is approximately linear in whichever of s and z is larger, and the
  // smaller one follows from the central path.
  //
  // The step size is capped by the fraction-to-the-boundary rule for the
  // linearized slack step pˢ = −s∘pᵛ. Slacks only approach the boundary where
  // pᵛ > 0, so
  //
  //   αᵐᵃˣ = max(α ∈ (0, 1] : s + αpˢ ≥ (1−τ)s)
  //        = min(1, τ/max(pᵛ))
  constexpr Scalar τ(0.995);
  auto max_step_size = [&](const DenseVector& p_v) -> Scalar {
    const Scalar p_v_max = p_v.size() > 0 ? p_v.maxCoeff() : Scalar(0);
    return p_v_max > Scalar(0) ? std::min(Scalar(1), τ / p_v_max) : Scalar(1);
  };
  auto step_v = [&](const DenseVector& p_v, Scalar α) -> DenseVector {
    // v⁺ = asinh(sinh(v) + α cosh(v)∘pᵛ)
    return (v.array().sinh() + α * v.array().cosh() * p_v.array())
        .unaryExpr([](Scalar t) {
          using std::asinh;
          return asinh(t);
        })
        .matrix();
  };

  // Initializes the barrier parameter for the current iterate.
  //
  // Returns true on success and false on failure.
  auto init_barrier_parameter = [&] {
    DenseVector p_v_0 = compute_step(Scalar(1e15)).p_v;
    DenseVector p_v_1 = compute_step(Scalar(1)).p_v - p_v_0;

    // See section 3.2.3 of [3]
    if (Scalar dot = p_v_0.transpose() * p_v_1; dot < Scalar(0)) {
      sqrt_μ = std::max(sqrt_μ_min, p_v_1.squaredNorm() / -dot);
    } else {
      // Initialization failed, so use a hardcoded value for μ instead
      sqrt_μ = Scalar(10);
    }
  };

  // Updates the barrier parameter for the current iterate and resets the
  // filter.
  //
  // This should be run when the error is below a desired threshold for the
  // current barrier parameter.
  auto update_barrier_parameter = [&] {
    if (sqrt_μ == sqrt_μ_min) {
      return;
    }

    bool found_μ = false;

    if (is_nlp) {
      // Binary search for smallest μ such that |pᵛ|_∞ ≤ 1 starting from the
      // current value of μ. If one doesn't exist, keep the original.

      constexpr Scalar sqrt_μ_line_search_tol(1e-8);

      Scalar sqrt_μ_lower(0);
      Scalar sqrt_μ_upper = sqrt_μ;

      while (sqrt_μ_upper - sqrt_μ_lower > sqrt_μ_line_search_tol) {
        // Search bias [0, 1] that determines which side of range to check
        constexpr Scalar search_bias(0.75);

        Scalar sqrt_μ_mid = (Scalar(1) - search_bias) * sqrt_μ_lower +
                            search_bias * sqrt_μ_upper;

        // The bounds are adjacent floating-point values, so the search can't
        // make further progress
        if (sqrt_μ_mid == sqrt_μ_lower || sqrt_μ_mid == sqrt_μ_upper) {
          break;
        }

        DenseVector p_v = compute_step(sqrt_μ_mid).p_v;
        Scalar p_v_infnorm = p_v.template lpNorm<Eigen::Infinity>();

        if (p_v_infnorm <= Scalar(1)) {
          // If step down was successful, decrease upper bound and try again
          sqrt_μ = sqrt_μ_mid;
          sqrt_μ_upper = sqrt_μ_mid;
          found_μ = true;

          // If μ hit minimum, stop searching
          if (sqrt_μ <= sqrt_μ_min) {
            sqrt_μ = sqrt_μ_min;
            break;
          }
        } else {
          // Otherwise, increase lower bound and try again
          sqrt_μ_lower = sqrt_μ_mid;
        }
      }
    } else {
      // Line search for smallest μ such that |pᵛ|_∞ ≤ 1. If one doesn't exist,
      // keep the original.
      //
      // For quadratic models, this only requires two system solves instead of a
      // binary search.

      constexpr Scalar dinf_bound(0.99);

      DenseVector p_v_0 = compute_step(Scalar(1e15)).p_v;
      DenseVector p_v_1 = compute_step(Scalar(1)).p_v - p_v_0;

      Scalar α_μ_min(0);
      Scalar α_μ_max(1e15);

      for (int i = 0; i < v.rows(); ++i) {
        Scalar temp_min = (dinf_bound - p_v_0[i]) / p_v_1[i];
        Scalar temp_max = (-dinf_bound - p_v_0[i]) / p_v_1[i];
        if (p_v_1[i] > Scalar(0)) {
          using std::swap;
          swap(temp_min, temp_max);
        }

        α_μ_min = std::max(α_μ_min, temp_min);
        α_μ_max = std::min(α_μ_max, temp_max);
      }

      if (α_μ_min <= α_μ_max) {
        found_μ = true;
        sqrt_μ = std::max(sqrt_μ_min, Scalar(1) / α_μ_max);
      }
    }

    if (found_μ) {
      // Reset the filter when the barrier parameter is updated
      filter.reset();
      fallback_kkt_error = std::numeric_limits<Scalar>::infinity();
    }
  };

  // Step size reduction factor
  constexpr Scalar α_reduction_factor(1.0 / std::numbers::sqrt2);

  int full_step_rejected_counter = 0;

  StallDetector<Scalar> stall_detector;

  FeasibilityRestorationCycleDetector<Scalar>
      feasibility_restoration_cycle_detector;

  // Error
  Scalar E_0 = unscaled_kkt_error_inf_norm<Scalar>(
      matrices.scaling, iterates, function_values, first_derivatives, sqrt_μ,
      Scalar(0), options.tolerance);

  Scalar prev_p_v_infnorm = std::numeric_limits<Scalar>::infinity();
  bool μ_initialized = false;

  // Watchdog (nonmonotone) variables. If a line search fails, accept up to this
  // many steps in a row in case the dual variable steps allow the primal steps
  // to make progress again.
  constexpr int watchdog_max = 5;
  int watchdog_count = 0;

  // Prints initial iterate diagnostics now and closes the iterate diagnostics
  // table when the solver exits.
  //
  // Feasibility restoration's iterations are printed in the outer solver's
  // table, so it doesn't open its own.
  IterateDiagnosticsTable table{
      options.diagnostics && !in_feasibility_restoration, E_0, f,
      constraint_violation(c_e, c_i, s), sqrt_μ * sqrt_μ};

  while (E_0 > options.tolerance) {
    ScopedTimeTracer inner_iter_tracer{diagnostics[ITERATION]};

    // Check for diverging iterates
    if (x.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !x.allFinite() ||
        v.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !v.allFinite()) {
      return ExitStatus::DIVERGING_ITERATES;
    }

    PhaseTimeTracer phase_tracer{diagnostics[CALLBACKS]};

    // Call iteration callbacks
    for (const auto& callback : iteration_callbacks) {
      if (callback({iterations, x, y, v, g, H, A_e, A_i}) ==
          ControlFlow::BREAK) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    phase_tracer.stop();

    if (!build_and_factorize_kkt_matrix()) [[unlikely]] {
      return ExitStatus::FACTORIZATION_FAILED;
    }

    // Update the barrier parameter if necessary
    const Scalar prev_sqrt_μ = sqrt_μ;
    bool v_reset = false;
    if (!μ_initialized) {
      init_barrier_parameter();
      μ_initialized = true;

      if (!in_feasibility_restoration) {
        // Initialize the slack variables to the inequality constraint values,
        // pushed away from zero. Otherwise, every slack starts at √(μ), so the
        // first Newton steps have to close a large residual cᵢ − s, which
        // produces large dual steps and Hessian regularization.
        //
        //   s = max(cᵢ, κ)
        //   v = −ln(s/√(μ))
        constexpr Scalar κ(1e-2);
        v = -(c_i.cwiseMax(κ) / sqrt_μ).array().log().matrix();
        exp_v = v.array().exp().matrix();
        exp_neg_v = exp_v.cwiseInverse();
        v_reset = true;
      }
    } else if (is_nlp) {
      Scalar E_μ = kkt_error_inf_norm<Scalar>(
          iterates, function_values, first_derivatives, sqrt_μ, sqrt_μ * sqrt_μ,
          options.tolerance);
      if (E_μ <= barrier_subproblem_tolerance(sqrt_μ * sqrt_μ)) {
        ScopedTimeTracer μ_update_tracer{diagnostics[BARRIER_PARAMETER_UPDATE]};
        update_barrier_parameter();
      }
    } else if (prev_p_v_infnorm <= Scalar(1)) {
      ScopedTimeTracer μ_update_tracer{diagnostics[BARRIER_PARAMETER_UPDATE]};
      update_barrier_parameter();
    }

    // Update expressions dependent on √(μ)
    if (sqrt_μ != prev_sqrt_μ || v_reset) {
      // s = √(μ)e⁻ᵛ
      s = sqrt_μ * exp_neg_v;

      // When the inequality constraints are nonlinear, the Lagrangian Hessian
      // depends on μ through z = √(μ)eᵛ
      if (is_nlp) {
        H = matrices.H(x, y, v, sqrt_μ);
      }

      // The Newton-KKT matrix depends on the Lagrangian Hessian, which was just
      // updated
      if ((is_nlp || v_reset) && !build_and_factorize_kkt_matrix())
          [[unlikely]] {
        return ExitStatus::FACTORIZATION_FAILED;
      }
    }

    phase_tracer.next(diagnostics[KKT_SYSTEM_SOLVE]);

    // Solve the Newton-KKT system for the step
    auto step = compute_step(sqrt_μ);

    phase_tracer.next(diagnostics[LINE_SEARCH]);

    // The slack variables s = √(μ)e⁻ᵛ are part of the primal step, so x and v
    // share a step size, capped by the fraction-to-the-boundary rule for the
    // slack step.
    Scalar p_v_infnorm = step.p_v.template lpNorm<Eigen::Infinity>();
    const Scalar α_max = max_step_size(step.p_v);
    Scalar α = α_max;

    bool call_feasibility_restoration = false;

    prev_p_v_infnorm = p_v_infnorm;

    const FilterEntry<Scalar> current_entry{v, function_values, sqrt_μ};

    // Compute the directional derivative of the log-barrier function along the
    // search direction.
    //
    //   ϕ_μ(x, v) = f(x) − μ∑ᵢ ln(sᵢ)
    //             = f(x) − μ∑ᵢ ln(√(μ)e⁻ᵛⁱ)
    //             = f(x) + μ∑ᵢ vᵢ − μ∑ᵢ ln(√(μ))
    //
    //   D_ϕ = ∇ϕ_μ(x, v)ᵀ[pˣ pᵛ]
    //       = ∇f(x)ᵀpˣ + μ∑ᵢ pᵢᵛ
    const Scalar D_ϕ =
        g.transpose() * step.p_x + sqrt_μ * sqrt_μ * step.p_v.sum();

    const Scalar α_min = filter.min_step_size(current_entry, D_ϕ);

    // If maximum step size is below minimum, invoke feasibility restoration
    if (α < α_min) {
      call_feasibility_restoration = true;
    }

    // α_min can be zero (e.g., when the constraint violation is zero), so also
    // end the line search when the step becomes negligible relative to the
    // iterate. This bounds the number of step size reductions.
    const Scalar α_tiny = step.tiny_step_size(x, s);

    // Loop until a step is accepted
    while (!call_feasibility_restoration) {
      trial_x = x + α * step.p_x;
      trial_y = y + α * step.p_y;
      trial_v = step_v(step.p_v, α);
      trial_c_i = matrices.c_i(trial_x);
      trial_f = matrices.f(trial_x);
      trial_c_e = matrices.c_e(trial_x);

      // If the inequality constraints are all feasible, prevent them from
      // becoming infeasible again
      const bool keep_feasible =
          options.feasible_ipm && c_i.cwiseGreater(Scalar(0)).all();

      // If f(xₖ + αpₖˣ), cₑ(xₖ + αpₖˣ), or cᵢ(xₖ + αpₖˣ) aren't finite, or the
      // inequality constraints must stay feasible and didn't, reduce step size
      // immediately
      if (!trial_function_values.all_finite() ||
          (keep_feasible && !trial_c_i.cwiseGreater(Scalar(0)).all())) {
        // Reduce step size
        α *= α_reduction_factor;

        if (α < α_min || α < α_tiny) {
          call_feasibility_restoration = true;
          break;
        }
        continue;
      }

      DenseVector trial_s;
      if (keep_feasible) {
        // Set the slack variables to the constraint values, which makes the
        // inequality constraint residual zero.
        //
        //   cᵢ − √(μ)e⁻ᵛ = 0
        //   √(μ)e⁻ᵛ = cᵢ
        //   e⁻ᵛ = 1/√(μ) cᵢ
        //   −v = ln(1/√(μ) cᵢ)
        //   v = −ln(1/√(μ) cᵢ)
        trial_s = trial_c_i;
        trial_v = -(trial_c_i * (Scalar(1) / sqrt_μ)).array().log().matrix();
      } else {
        trial_s = sqrt_μ * (-trial_v).array().exp().matrix();
      }

      // Check whether filter accepts trial iterate
      FilterEntry trial_entry{trial_v, trial_function_values, sqrt_μ};
      if (filter.try_add(current_entry, trial_entry, D_ϕ, α)) {
        // Accept step
        watchdog_count = 0;
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

        Scalar α_soc = α_max;
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
                      matrices.scaling, trial_iterates, trial_function_values,
                      first_derivatives, sqrt_μ, Scalar(0), options.tolerance),
                  trial_f, constraint_violation(trial_c_e, trial_c_i, trial_s),
                  sqrt_μ * sqrt_μ, kkt_system.hessian_regularization(),
                  kkt_system.constraint_jacobian_regularization(),
                  soc_step.primal_inf_norm(), soc_step.dual_inf_norm(), α_soc,
                  α_soc, α_reduction_factor, α_soc);
            }
          }};

          // Solve the Newton-KKT system with corrected constraint values
          //
          //   cₑˢᵒᶜ = αˢᵒᶜcₑ(xₖ) + cₑ(xₖ + αˢᵒᶜpˣ)
          //   (cᵢ − s)ˢᵒᶜ =
          //     αˢᵒᶜ(cᵢ(xₖ) − sₖ) + cᵢ(xₖ + αˢᵒᶜpˣ) − sₖ(αˢᵒᶜpᵛ)
          c_e_soc = α_soc * c_e_soc + trial_c_e;
          c_i_minus_s_soc = α_soc * c_i_minus_s_soc + trial_c_i - trial_s;
          soc_step = kkt_system.solve_soc(g, A_e, A_i, y, sqrt_μ, c_e_soc,
                                          c_i_minus_s_soc);

          α_soc = max_step_size(soc_step.p_v);

          trial_x = x + α_soc * soc_step.p_x;
          trial_y = y + α_soc * soc_step.p_y;
          trial_v = step_v(soc_step.p_v, α_soc);
          trial_s = sqrt_μ * (-trial_v).array().exp().matrix();

          trial_f = matrices.f(trial_x);
          trial_c_e = matrices.c_e(trial_x);
          trial_c_i = matrices.c_i(trial_x);

          // Check whether filter accepts trial iterate
          FilterEntry trial_entry{trial_v, trial_function_values, sqrt_μ};
          if (filter.try_add(current_entry, trial_entry, D_ϕ, α)) {
            step = soc_step;
            α = α_max;
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
          watchdog_count = 0;
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
        Scalar current_kkt_error = kkt_error_one_norm<Scalar>(
            iterates, function_values, first_derivatives, sqrt_μ,
            sqrt_μ * sqrt_μ);

        trial_x = x + α_max * step.p_x;
        trial_y = y + α_max * step.p_y;
        trial_v = step_v(step.p_v, α_max);
        trial_s = sqrt_μ * (-trial_v).array().exp().matrix();

        trial_f = matrices.f(trial_x);
        trial_c_e = matrices.c_e(trial_x);
        trial_c_i = matrices.c_i(trial_x);

        Scalar next_kkt_error = kkt_error_one_norm<Scalar>(
            trial_iterates, trial_function_values,
            IPMFirstDerivatives<Scalar>{matrices.g(trial_x),
                                        matrices.A_e(trial_x),
                                        matrices.A_i(trial_x)},
            sqrt_μ, sqrt_μ * sqrt_μ);

        // If the step using αᵐᵃˣ reduced the KKT error below the current
        // iterate's and every previous fallback step's, accept it anyway
        if (next_kkt_error <= Scalar(0.999) * std::min(current_kkt_error,
                                                       fallback_kkt_error) &&
            constraint_violation(trial_c_e, trial_c_i, trial_s) <=
                filter.max_constraint_violation) {
          // Accept step
          fallback_kkt_error = next_kkt_error;
          α = α_max;
          watchdog_count = 0;
          break;
        }

        // If the dual step is making progress, accept the whole step anyway
        if (p_v_infnorm > Scalar(1e-7) && watchdog_count < watchdog_max) {
          // Accept step
          α = α_max;
          ++watchdog_count;
          break;
        }

        call_feasibility_restoration = true;
        break;
      }
    }

    phase_tracer.stop();

    // If solver stalled, invoke feasibility restoration
    if (!call_feasibility_restoration && !in_feasibility_restoration) {
      FilterEntry trial_entry{trial_v, trial_function_values, sqrt_μ};
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

      FilterEntry initial_entry{v, function_values, sqrt_μ};

      // Square root of the feasibility restoration barrier parameter, which
      // feasibility restoration updates in place
      Scalar fr_sqrt_μ = sqrt_μ;

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

        DenseVector trial_x =
            info.x.segment(0, matrices.num_decision_variables);

        // The feasibility restoration log-domain variables are relative to its
        // barrier parameter, so convert them to the normal solve's barrier
        // parameter such that the slack variables are preserved.
        //
        //   √(μ)e⁻ᵛ = √(fr_μ)exp(−fr_v)
        //   v = fr_v − ln(√(fr_μ)/√(μ))
        using std::log;
        DenseVector trial_v =
            (info.v.segment(0, matrices.num_inequality_constraints).array() -
             log(fr_sqrt_μ / sqrt_μ))
                .matrix();

        const IPMFunctionValues<Scalar> trial_function_values{
            matrices.f(trial_x), matrices.c_e(trial_x), matrices.c_i(trial_x)};

        // Feasibility restoration re-centers the slacks when it returns, so
        // check the iterate the normal phase will resume from.
        //
        //   s = √(μ)e⁻ᵛ
        //   v = ln(√(μ)/s)
        DenseVector trial_s =
            (trial_function_values.c_i.array() >= Scalar(1e-2))
                .select(trial_function_values.c_i,
                        sqrt_μ * (-trial_v).array().exp().matrix());
        trial_v = (sqrt_μ / trial_s.array()).log().matrix();

        // If the current iterate sufficiently reduces constraint violation and
        // is accepted by the normal filter, stop feasibility restoration
        FilterEntry trial_entry{trial_v, trial_function_values, sqrt_μ};
        if (trial_entry.constraint_violation <
                Scalar(0.9) * initial_entry.constraint_violation &&
            filter.is_acceptable(trial_entry)) {
          return ControlFlow::BREAK;
        } else {
          return ControlFlow::CONTINUE;
        }
      });
      auto status = feasibility_restoration<Scalar>(
          matrices, is_nlp, callbacks, options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
          bound_constraint_mask,
#endif
          iterates, sqrt_μ, fr_sqrt_μ, iterations);

      if (status != ExitStatus::SUCCESS) {
        // Report failure
        return status;
      }

      function_values = {matrices.f(x), matrices.c_e(x), matrices.c_i(x)};

      // Resuming from an iterate the filter rejects would break the filter's
      // convergence guarantee, and continuing feasibility restoration from its
      // converged point would return that point again, so report failure.
      //
      // The returned iterate can be rejected because feasibility restoration
      // stops at its own convergence tolerance instead of driving constraint
      // violation to zero, so the filter can still hold entries with lower cost
      // and constraint violation.
      if (!filter.is_acceptable(FilterEntry{v, function_values, sqrt_μ})) {
        return ExitStatus::FEASIBILITY_RESTORATION_FAILED;
      }
    } else {
      // If full step was accepted, reset full-step rejected counter
      if (α == α_max) {
        full_step_rejected_counter = 0;
      }

      // Update iterates and function values
      iterates = trial_iterates;
      function_values = trial_function_values;
    }

    exp_v = v.array().exp().matrix();
    exp_neg_v = exp_v.cwiseInverse();
    s = sqrt_μ * exp_neg_v;

    // Update autodiff for Jacobians and Hessian
    A_e = matrices.A_e(x);
    A_i = matrices.A_i(x);
    g = matrices.g(x);
    H = matrices.H(x, y, v, sqrt_μ);

    // Update the error
    E_0 = unscaled_kkt_error_inf_norm<Scalar>(
        matrices.scaling, iterates, function_values, first_derivatives, sqrt_μ,
        Scalar(0), options.tolerance);

    inner_iter_tracer.stop();

    if (options.diagnostics) {
      print_iterate_diagnostics(
          iterations,
          in_feasibility_restoration ? IterationType::FEASIBILITY_RESTORATION
                                     : IterationType::NORMAL,
          inner_iter_tracer.current_duration(), E_0, f,
          constraint_violation(c_e, c_i, s), sqrt_μ * sqrt_μ,
          kkt_system.hessian_regularization(),
          kkt_system.constraint_jacobian_regularization(),
          step.primal_inf_norm(), step.dual_inf_norm(), α, α_max,
          α_reduction_factor, α);
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
ipm(const IPMMatrixCallbacks<double>& matrix_callbacks, bool is_nlp,
    std::span<std::function<ControlFlow(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options<double>& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorX<double>& x);

extern template SLEIPNIR_DLLEXPORT ExitStatus
ipm(const IPMMatrixCallbacks<double>& matrix_callbacks, bool is_nlp,
    std::span<std::function<ControlFlow(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options<double>& options, bool in_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    IPMIterates<double>& iterates, double& sqrt_μ, int& iterations);

}  // namespace slp
