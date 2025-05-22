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

  auto& [x, v] = iterates;

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
        traced(diagnostics[INEQUALITY_CONSTRAINTS], matrix_callbacks.c_i),
        traced(diagnostics[INEQUALITY_JACOBIAN], matrix_callbacks.A_i),
        matrix_callbacks.scaling);
  }
  const auto& matrices = traced_matrices ? *traced_matrices : matrix_callbacks;
#else
  const auto& matrices = matrix_callbacks;
#endif

  ScopedTimeTracer setup_tracer{diagnostics[SETUP]};

  IPMFunctionValues<Scalar> function_values{.f = matrices.f(x),
                                            .c_i = matrices.c_i(x)};
  IPMFirstDerivatives<Scalar> first_derivatives{.g = matrices.g(x),
                                                .A_i = matrices.A_i(x)};
  auto& [f, c_i] = function_values;
  auto& [g, A_i] = first_derivatives;
  SparseMatrix H = matrices.H(x, v, sqrt_μ);

  // Ensure matrix callback dimensions are consistent
  slp_assert(g.rows() == matrices.num_decision_variables);
  slp_assert(H.rows() == matrices.num_decision_variables);
  slp_assert(H.cols() == matrices.num_decision_variables);
  slp_assert(c_i.rows() == matrices.num_inequality_constraints);
  slp_assert(A_i.rows() == matrices.num_inequality_constraints);
  slp_assert(A_i.cols() == matrices.num_decision_variables);

  IPMIterates<Scalar> trial_iterates;
  auto& [trial_x, trial_v] = trial_iterates;

  IPMFunctionValues<Scalar> trial_function_values;
  auto& [trial_f, trial_c_i] = trial_function_values;

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

  Filter<Scalar> filter{constraint_violation(c_i, s)};

  // The smallest KKT error of a step accepted by the KKT error reduction
  // fallback for the current barrier parameter. Each fallback step must improve
  // on it, so finitely many fallback steps are taken before the barrier
  // subproblem converges.
  Scalar fallback_kkt_error = std::numeric_limits<Scalar>::infinity();

  IPMKKTSystem<Scalar> kkt_system{H, A_i};

  setup_tracer.stop();

  auto build_and_factorize_kkt_matrix = [&]() -> bool {
    PhaseTimeTracer phase_tracer{diagnostics[KKT_MATRIX_BUILD]};

    kkt_system.build(H, A_i, exp_v);

    phase_tracer.next(diagnostics[KKT_MATRIX_FACTOR]);

    return kkt_system.factorize();
  };

  // Solves the Newton-KKT system for the step with the given √(μ). μ_w is the
  // barrier parameter used for the infeasibility and gradient perturbations μw
  // and μβ₁e. It's normally μ, but setting it to zero makes pᵛ affine in
  // 1/√(μ), which the barrier parameter initialization relies on.
  auto compute_step = [&](Scalar r, Scalar μ_w) -> IPMStep<Scalar> {
    return kkt_system.solve(g, A_i, r, c_i, μ_w);
  };

  // The slack variables s = √(μ)e⁻ᵛ take a linear step in s-space,
  //
  //   pˢ = −s∘pᵛ
  //   s⁺ = s + αpˢ = s∘(e − αpᵛ)
  //   v⁺ = v − ln(e − αpᵛ)
  //
  // rather than the exponential step s∘exp(−αpᵛ). The linear step matches the
  // Newton linearization, so cᵢ − s decreases by exactly (1 − α) per step for
  // linear constraints instead of being thrown off by the overshoot of the
  // exponential.
  //
  // Fraction-to-the-boundary rule for the slack step. Slacks only approach the
  // boundary where pᵛ > 0, so
  //
  //   αᵐᵃˣ = max(α ∈ (0, 1] : s + αpˢ ≥ (1−τ)s)
  //        = min(1, τ/max(pᵛ))
  constexpr Scalar τ(0.995);
  auto max_step_size = [&](const DenseVector& p_v) -> Scalar {
    const Scalar p_v_max = p_v.size() > 0 ? p_v.maxCoeff() : Scalar(0);
    return p_v_max > Scalar(0) ? std::min(Scalar(1), τ / p_v_max) : Scalar(1);
  };
  auto step_v = [&](const DenseVector& p_v, Scalar α) -> DenseVector {
    // v⁺ = v − ln(e − αpᵛ)
    return v - (-α * p_v).array().log1p().matrix();
  };

  // Initializes the barrier parameter for the current iterate.
  //
  // Returns true on success and false on failure.
  auto init_barrier_parameter = [&] {
    // The perturbations are omitted here so pᵛ is affine in 1/√(μ)
    DenseVector p_v_0 = compute_step(Scalar(1e15), Scalar(0)).p_v;
    DenseVector p_v_1 = compute_step(Scalar(1), Scalar(0)).p_v - p_v_0;

    // See section 3.2.3 of [3]
    if (Scalar dot = p_v_0.transpose() * p_v_1; dot < Scalar(0)) {
      sqrt_μ = std::max(sqrt_μ_min, p_v_1.squaredNorm() / -dot);
    } else {
      // Initialization failed, so use a hardcoded value for μ instead
      sqrt_μ = Scalar(10);
    }
  };

  // Takes an aggressive step that reduces the barrier parameter along with the
  // relaxed infeasibility, then resets the filter. See [4].
  //
  // The step linearizes the perturbed KKT conditions in x, v, and μ with
  // dμ = −ημ for η ∈ (0, 1], so the barrier parameter after a step of size α
  // is
  //
  //   μ⁺ = (1 − αη)μ
  //
  // The reduced system shares its matrix with the normal step (η = 0).
  //
  //   [H + Aᵢᵀdiag(e²ᵛ)Aᵢ][pˣ] = −[∇f − Aᵢᵀ((2 − η)√(μ)eᵛ −
  //                                 e²ᵛ∘(cᵢ + (1 − η)μw)) − (1 − η)μβ₁e]
  //
  // The slacks take the linear step s⁺ = s∘(e − αqᵛ) where
  //
  //   qᵛ = e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ + (1 − η)μw)
  //
  // so for linear constraints, cᵢ − s + μw shrinks by exactly (1 − αη) along
  // with μ. Since s⁺ = √(μ⁺)exp(−v⁺),
  //
  //   v⁺ = v − ln(e − αqᵛ) + ½ln(1 − αη)
  //
  // Shrinking the relaxed infeasibility and μ at the same rate keeps the duals
  // bounded. Lowering μ on its own would require each active slack to absorb
  // the drop in μw in one step, which fails when sᵢ ≪ μ.
  //
  // η is chosen with Mehrotra's heuristic from the step size of the affine
  // direction (η = 1) so the step isn't immediately blocked by the boundary.
  //
  // This should be run when the error is below a desired threshold for the
  // current barrier parameter. Returns true if a step was taken.
  auto update_barrier_parameter = [&]() -> bool {
    using std::log;
    using std::log1p;
    using std::pow;

    if (sqrt_μ <= sqrt_μ_min) {
      return false;
    }

    const Scalar μ = sqrt_μ * sqrt_μ;
    const Scalar μ_min = sqrt_μ_min * sqrt_μ_min;

    // Returns the direction (pˣ, qᵛ) for the given η
    auto compute_aggressive_step = [&](Scalar η) -> IPMStep<Scalar> {
      return kkt_system.solve(g, A_i, sqrt_μ, c_i, (Scalar(1) - η) * μ, η);
    };

    // Mehrotra's heuristic: σ = (1 − αₐff)³ where αₐff is the
    // fraction-to-the-boundary step size of the affine direction
    const Scalar α_aff = max_step_size(compute_aggressive_step(Scalar(1)).p_v);
    const Scalar η = Scalar(1) - pow(Scalar(1) - α_aff, Scalar(3));

    const auto step = compute_aggressive_step(η);

    // Cap α with the fraction-to-the-boundary rule for the slacks and with
    // μ⁺ ≥ μₘᵢₙ
    Scalar α = std::min(max_step_size(step.p_v), (Scalar(1) - μ_min / μ) / η);

    const bool keep_feasible =
        options.feasible_ipm && c_i.cwiseGreater(Scalar(0)).all();

    // Backtrack until the iterate is close enough to the new barrier
    // subproblem's solution. Steps that reduce μ by less than a factor of
    // (1 − ημ_min) don't make enough progress, since repeating them can
    // converge to a μ > 0 where the boundary blocks the central path.
    constexpr Scalar ημ_min(1e-2);
    while (α * η >= ημ_min) {
      const Scalar trial_μ = (Scalar(1) - α * η) * μ;
      const Scalar trial_sqrt_μ = sqrt(trial_μ);

      trial_x = x + α * step.p_x;
      trial_v = (v.array() - (-α * step.p_v).array().log1p() +
                 Scalar(0.5) * log1p(-α * η))
                    .matrix();
      trial_f = matrices.f(trial_x);
      trial_c_i = matrices.c_i(trial_x);

      if (trial_function_values.all_finite() &&
          (!keep_feasible || trial_c_i.cwiseGreater(Scalar(0)).all())) {
        IPMFirstDerivatives<Scalar> trial_first_derivatives{
            .g = matrices.g(trial_x), .A_i = matrices.A_i(trial_x)};

        Scalar E = kkt_error_inf_norm<Scalar>(
            trial_iterates, trial_function_values, trial_first_derivatives,
            trial_sqrt_μ, trial_μ, options.tolerance);
        if (E <= barrier_subproblem_tolerance(trial_μ)) {
          iterates = trial_iterates;
          function_values = trial_function_values;
          first_derivatives = std::move(trial_first_derivatives);
          sqrt_μ = trial_sqrt_μ;

          exp_v = v.array().exp().matrix();
          exp_neg_v = exp_v.cwiseInverse();

          // Reset the filter when the barrier parameter is updated
          filter.reset();
          fallback_kkt_error = std::numeric_limits<Scalar>::infinity();

          return true;
        }
      }

      α *= Scalar(0.5);
    }

    // The central path is blocked by the boundary (e.g., the barrier
    // subproblem solution is on a branch that becomes infeasible as μ → 0), so
    // decrease μ anyway while preserving the slacks. The iterate then violates
    // the tighter relaxation cᵢ − s + μ⁺w = 0, which the normal step and
    // feasibility restoration can reduce.
    //
    //   √(μ)e⁻ᵛ = √(μ⁺)exp(−v⁺)
    //   v⁺ = v + ln(√(μ⁺)/√(μ))
    constexpr Scalar κ_μ(0.2);
    const Scalar new_sqrt_μ = std::max(sqrt_μ_min, sqrt(κ_μ * μ));
    v.array() += log(new_sqrt_μ / sqrt_μ);
    sqrt_μ = new_sqrt_μ;

    exp_v = v.array().exp().matrix();
    exp_neg_v = exp_v.cwiseInverse();

    // Reset the filter when the barrier parameter is updated
    filter.reset();
    fallback_kkt_error = std::numeric_limits<Scalar>::infinity();

    return true;
  };

  // Step size reduction factor
  constexpr Scalar α_reduction_factor(1.0 / std::numbers::sqrt2);

  int full_step_rejected_counter = 0;

  StallDetector<Scalar> stall_detector;

  FeasibilityRestorationCycleDetector<Scalar>
      feasibility_restoration_cycle_detector;

  // |pᵛ|_∞ of the previous normal step
  Scalar prev_p_v_infnorm = std::numeric_limits<Scalar>::infinity();

  // Error
  Scalar E_0 = unscaled_kkt_error_inf_norm<Scalar>(
      matrices.scaling, iterates, function_values, first_derivatives, sqrt_μ,
      Scalar(0), options.tolerance);

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
      constraint_violation(c_i, s), sqrt_μ * sqrt_μ};

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
      if (callback({iterations, x, {}, v, g, H, {}, A_i}) ==
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
    } else if (
        // The barrier subproblem is approximately solved, and the iterate is
        // close enough to the central path that the previous normal step
        // changed each slack by less than 100%
        prev_p_v_infnorm <= Scalar(1) &&
        kkt_error_inf_norm<Scalar>(iterates, function_values, first_derivatives,
                                   sqrt_μ, sqrt_μ * sqrt_μ,
                                   options.tolerance) <=
            barrier_subproblem_tolerance(sqrt_μ * sqrt_μ)) {
      ScopedTimeTracer μ_update_tracer{diagnostics[BARRIER_PARAMETER_UPDATE]};

      // The aggressive step moves x and v, so the Newton-KKT matrix must be
      // rebuilt even for problems whose Lagrangian Hessian doesn't depend on μ
      v_reset = update_barrier_parameter();
    }

    // Update expressions dependent on √(μ)
    if (sqrt_μ != prev_sqrt_μ || v_reset) {
      // s = √(μ)e⁻ᵛ
      s = sqrt_μ * exp_neg_v;

      // When the inequality constraints are nonlinear, the Lagrangian Hessian
      // depends on μ through z = √(μ)eᵛ
      if (is_nlp) {
        H = matrices.H(x, v, sqrt_μ);
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
    auto step = compute_step(sqrt_μ, sqrt_μ * sqrt_μ);

    phase_tracer.next(diagnostics[LINE_SEARCH]);

    // The slack variables s = √(μ)e⁻ᵛ are part of the primal step, so x and v
    // share a step size, capped by the fraction-to-the-boundary rule for the
    // slack step.
    Scalar p_v_infnorm = step.p_v.template lpNorm<Eigen::Infinity>();
    prev_p_v_infnorm = p_v_infnorm;
    const Scalar α_max = max_step_size(step.p_v);
    Scalar α = α_max;

    bool call_feasibility_restoration = false;

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
      trial_v = step_v(step.p_v, α);
      trial_c_i = matrices.c_i(trial_x);
      trial_f = matrices.f(trial_x);

      // If the inequality constraints are all feasible, prevent them from
      // becoming infeasible again
      const bool keep_feasible =
          options.feasible_ipm && c_i.cwiseGreater(Scalar(0)).all();

      // If f(xₖ + αpₖˣ) or cᵢ(xₖ + αpₖˣ) aren't finite, or the inequality
      // constraints must stay feasible and didn't, reduce step size immediately
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

      Scalar prev_constraint_violation = constraint_violation(c_i, s);
      Scalar next_constraint_violation =
          constraint_violation(trial_c_i, trial_s);

      // Second-order corrections
      //
      // If first trial point was rejected and constraint violation stayed the
      // same or went up, apply second-order corrections
      if (α == α_max &&
          next_constraint_violation >= prev_constraint_violation) {
        // Apply second-order corrections. See section 2.4 of [2].
        auto soc_step = step;

        Scalar α_soc = α_max;
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
                  trial_f, constraint_violation(trial_c_i, trial_s),
                  sqrt_μ * sqrt_μ, kkt_system.hessian_regularization(),
                  kkt_system.constraint_jacobian_regularization(),
                  soc_step.primal_inf_norm(), soc_step.dual_inf_norm(), α_soc,
                  α_soc, α_reduction_factor, α_soc);
            }
          }};

          // Solve the Newton-KKT system with corrected constraint values
          //
          //   (cᵢ − s)ˢᵒᶜ =
          //     αˢᵒᶜ(cᵢ(xₖ) − sₖ) + cᵢ(xₖ + αˢᵒᶜpˣ) − sₖ(αˢᵒᶜpᵛ)
          c_i_minus_s_soc = α_soc * c_i_minus_s_soc + trial_c_i - trial_s;
          soc_step = kkt_system.solve_soc(g, A_i, sqrt_μ, c_i_minus_s_soc);

          α_soc = max_step_size(soc_step.p_v);

          trial_x = x + α_soc * soc_step.p_x;
          trial_v = step_v(soc_step.p_v, α_soc);
          trial_s = sqrt_μ * (-trial_v).array().exp().matrix();

          trial_f = matrices.f(trial_x);
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
          next_constraint_violation = constraint_violation(trial_c_i, trial_s);
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
        trial_v = step_v(step.p_v, α_max);
        trial_s = sqrt_μ * (-trial_v).array().exp().matrix();

        trial_f = matrices.f(trial_x);
        trial_c_i = matrices.c_i(trial_x);

        Scalar next_kkt_error = kkt_error_one_norm<Scalar>(
            trial_iterates, trial_function_values,
            IPMFirstDerivatives<Scalar>{matrices.g(trial_x),
                                        matrices.A_i(trial_x)},
            sqrt_μ, sqrt_μ * sqrt_μ);

        // If the step using αᵐᵃˣ reduced the KKT error below the current
        // iterate's and every previous fallback step's, accept it anyway
        if (next_kkt_error <= Scalar(0.999) * std::min(current_kkt_error,
                                                       fallback_kkt_error) &&
            constraint_violation(trial_c_i, trial_s) <=
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
            matrices.f(trial_x), matrices.c_i(trial_x)};

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

      function_values = {matrices.f(x), matrices.c_i(x)};

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
    A_i = matrices.A_i(x);
    g = matrices.g(x);
    H = matrices.H(x, v, sqrt_μ);

    // Update the error
    E_0 = unscaled_kkt_error_inf_norm<Scalar>(
        matrices.scaling, iterates, function_values, first_derivatives, sqrt_μ,
        Scalar(0), options.tolerance);

    inner_iter_tracer.stop();

    if (options.diagnostics) {
      print_iterate_diagnostics(iterations,
                                in_feasibility_restoration
                                    ? IterationType::FEASIBILITY_RESTORATION
                                    : IterationType::NORMAL,
                                inner_iter_tracer.current_duration(), E_0, f,
                                constraint_violation(c_i, s), sqrt_μ * sqrt_μ,
                                kkt_system.hessian_regularization(),
                                kkt_system.constraint_jacobian_regularization(),
                                step.primal_inf_norm(), step.dual_inf_norm(), α,
                                α_max, α_reduction_factor, α);
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
