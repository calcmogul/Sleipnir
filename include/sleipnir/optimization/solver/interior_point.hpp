// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/interior_point_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/util/error_estimate.hpp"
#include "sleipnir/optimization/solver/util/filter.hpp"
#include "sleipnir/optimization/solver/util/is_locally_infeasible.hpp"
#include "sleipnir/optimization/solver/util/kkt_error.hpp"
#include "sleipnir/optimization/solver/util/regularized_ldlt.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/print_diagnostics.hpp"
#include "sleipnir/util/scope_exit.hpp"
#include "sleipnir/util/scoped_profiler.hpp"
#include "sleipnir/util/solve_profiler.hpp"
#include "sleipnir/util/symbol_exports.hpp"

// See docs/algorithms.md#Works_cited for citation definitions.
//
// See docs/algorithms.md#Interior-point_method for a derivation of the
// interior-point method formulation being used.

namespace slp {

/**
Finds the optimal solution to a nonlinear program using the interior-point
method.

A nonlinear program has the form:

@verbatim
     min_x f(x)
subject to cₑ(x) = 0
           cᵢ(x) ≥ 0
@endverbatim

where f(x) is the cost function, cₑ(x) are the equality constraints, and cᵢ(x)
are the inequality constraints.

@tparam Scalar Scalar type.
@param[in] matrix_callbacks Matrix callbacks.
@param[in] is_nlp If true, the solver uses a more conservative barrier parameter
  reduction strategy that's more reliable on NLPs. Pass false for problems with
  quadratic or lower-order cost and linear or lower-order constraints.
@param[in] iteration_callbacks The list of callbacks to call at the beginning of
  each iteration.
@param[in] options Solver options.
@param[in,out] x The initial guess and output location for the decision
  variables.
@return The exit status.
*/
template <typename Scalar>
ExitStatus interior_point(
    const InteriorPointMatrixCallbacks<Scalar>& matrix_callbacks, bool is_nlp,
    std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::Vector<Scalar, Eigen::Dynamic>& x) {
  /**
   * Interior-point method step direction.
   */
  struct Step {
    /// Primal step.
    Eigen::Vector<Scalar, Eigen::Dynamic> p_x;
    /// Log-domain variable step.
    Eigen::Vector<Scalar, Eigen::Dynamic> p_v;
  };

  using std::isfinite;
  using std::sqrt;

  const auto solve_start_time = std::chrono::steady_clock::now();

  gch::small_vector<SolveProfiler> solve_profilers;
  solve_profilers.emplace_back("solver");
  solve_profilers.emplace_back("  ↳ setup");
  solve_profilers.emplace_back("  ↳ iteration");
  solve_profilers.emplace_back("    ↳ feasibility ✓");
  solve_profilers.emplace_back("    ↳ iter callbacks");
  solve_profilers.emplace_back("    ↳ μ update");
  solve_profilers.emplace_back("    ↳ KKT matrix build");
  solve_profilers.emplace_back("    ↳ KKT matrix decomp");
  solve_profilers.emplace_back("    ↳ KKT system solve");
  solve_profilers.emplace_back("    ↳ line search");
  solve_profilers.emplace_back("      ↳ SOC");
  solve_profilers.emplace_back("    ↳ next iter prep");
  solve_profilers.emplace_back("    ↳ f(x)");
  solve_profilers.emplace_back("    ↳ ∇f(x)");
  solve_profilers.emplace_back("    ↳ ∇²ₓₓL");
  solve_profilers.emplace_back("    ↳ cᵢ(x)");
  solve_profilers.emplace_back("    ↳ ∂cᵢ/∂x");

  auto& solver_prof = solve_profilers[0];
  auto& setup_prof = solve_profilers[1];
  auto& inner_iter_prof = solve_profilers[2];
  auto& feasibility_check_prof = solve_profilers[3];
  auto& iter_callbacks_prof = solve_profilers[4];
  auto& μ_update_prof = solve_profilers[5];
  auto& kkt_matrix_build_prof = solve_profilers[6];
  auto& kkt_matrix_decomp_prof = solve_profilers[7];
  auto& kkt_system_solve_prof = solve_profilers[8];
  auto& line_search_prof = solve_profilers[9];
  // auto& soc_prof = solve_profilers[10];
  auto& next_iter_prep_prof = solve_profilers[11];

  // Set up profiled matrix callbacks
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  auto& f_prof = solve_profilers[12];
  auto& g_prof = solve_profilers[13];
  auto& H_prof = solve_profilers[14];
  auto& c_i_prof = solve_profilers[15];
  auto& A_i_prof = solve_profilers[16];

  InteriorPointMatrixCallbacks<Scalar> matrices{
      [&](const Eigen::Vector<Scalar, Eigen::Dynamic>& x) -> Scalar {
        ScopedProfiler prof{f_prof};
        return matrix_callbacks.f(x);
      },
      [&](const Eigen::Vector<Scalar, Eigen::Dynamic>& x)
          -> Eigen::SparseVector<Scalar> {
        ScopedProfiler prof{g_prof};
        return matrix_callbacks.g(x);
      },
      [&](const Eigen::Vector<Scalar, Eigen::Dynamic>& x,
          const Eigen::Vector<Scalar, Eigen::Dynamic>& v,
          Scalar sqrt_μ) -> Eigen::SparseMatrix<Scalar> {
        ScopedProfiler prof{H_prof};
        return matrix_callbacks.H(x, v, sqrt_μ);
      },
      [&](const Eigen::Vector<Scalar, Eigen::Dynamic>& x)
          -> Eigen::Vector<Scalar, Eigen::Dynamic> {
        ScopedProfiler prof{c_i_prof};
        return matrix_callbacks.c_i(x);
      },
      [&](const Eigen::Vector<Scalar, Eigen::Dynamic>& x)
          -> Eigen::SparseMatrix<Scalar> {
        ScopedProfiler prof{A_i_prof};
        return matrix_callbacks.A_i(x);
      }};
#else
  const auto& matrices = matrix_callbacks;
#endif

  solver_prof.start();
  setup_prof.start();

  Scalar f = matrices.f(x);
  Eigen::Vector<Scalar, Eigen::Dynamic> c_i = matrices.c_i(x);

  int num_decision_variables = x.rows();
  int num_inequality_constraints = c_i.rows();

  Eigen::SparseVector<Scalar> g = matrices.g(x);
  Eigen::SparseMatrix<Scalar> A_i = matrices.A_i(x);

  Eigen::Vector<Scalar, Eigen::Dynamic> v =
      Eigen::Vector<Scalar, Eigen::Dynamic>::Zero(num_inequality_constraints);

  // Barrier parameter minimum
  const Scalar sqrt_μ_min = sqrt(Scalar(options.tolerance) / Scalar(10));

  // Barrier parameter μ
  Scalar sqrt_μ(0);

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
  Eigen::Vector<Scalar, Eigen::Dynamic> exp_v{v.array().exp().matrix()};
  // e⁻ᵛ
  Eigen::Vector<Scalar, Eigen::Dynamic> exp_neg_v = exp_v.cwiseInverse();
  // e²ᵛ
  Eigen::Vector<Scalar, Eigen::Dynamic> exp_2v = exp_v.cwiseProduct(exp_v);
  // s = √(μ)e⁻ᵛ
  Eigen::Vector<Scalar, Eigen::Dynamic> s = sqrt_μ * exp_neg_v;

  Eigen::SparseMatrix<Scalar> H = matrices.H(x, v, sqrt_μ);

  // Ensure matrix callback dimensions are consistent
  slp_assert(g.rows() == num_decision_variables);
  slp_assert(A_i.rows() == num_inequality_constraints);
  slp_assert(A_i.cols() == num_decision_variables);
  slp_assert(H.rows() == num_decision_variables);
  slp_assert(H.cols() == num_decision_variables);

  // Check whether initial guess has finite f(xₖ) and cᵢ(xₖ)
  if (!isfinite(f) || !c_i.allFinite()) {
    return ExitStatus::NONFINITE_INITIAL_COST_OR_CONSTRAINTS;
  }

  int iterations = 0;

  Filter<Scalar> filter;

  // Kept outside the loop so its storage can be reused
  gch::small_vector<Eigen::Triplet<Scalar>> triplets;

  RegularizedLDLT<Scalar> solver{num_decision_variables, 0};
  Eigen::SparseMatrix<Scalar> lhs(num_decision_variables,
                                  num_decision_variables);
  Eigen::Vector<Scalar, Eigen::Dynamic> rhs{x.rows()};

  setup_prof.stop();

  // r is sqrt_μ
  auto build_and_compute_lhs = [&]() -> ExitStatus {
    ScopedProfiler kkt_matrix_build_profiler{kkt_matrix_build_prof};

    // lhs = H + Aᵢᵀdiag(e²ᵛ)Aᵢ
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    lhs = H + (A_i.transpose() * exp_2v.asDiagonal() * A_i)
                  .template triangularView<Eigen::Lower>();

    kkt_matrix_build_profiler.stop();
    ScopedProfiler kkt_matrix_decomp_profiler{kkt_matrix_decomp_prof};

    // Solve the Newton-KKT system
    //
    // [H + Aᵢᵀdiag(e²ᵛ)Aᵢ][pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘(cᵢ + μw)) −
    //                              μβ₁e]
    if (solver.compute(lhs).info() != Eigen::Success) {
      return ExitStatus::FACTORIZATION_FAILED;
    } else {
      return ExitStatus::SUCCESS;
    }
  };

  // r is sqrt_μ
  auto build_rhs = [&](Scalar r) {
    constexpr Scalar β_1(1e-4);
    Eigen::Vector<Scalar, Eigen::Dynamic> μe =
        Eigen::Vector<Scalar, Eigen::Dynamic>::Constant(c_i.rows(), r * r);
    Eigen::Vector<Scalar, Eigen::Dynamic> μβ_1e =
        Eigen::Vector<Scalar, Eigen::Dynamic>::Constant(x.rows(), r * r * β_1);

    // rhs = −[∇f − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘(cᵢ + μw)) − μβ₁e]
    rhs = -g +
          A_i.transpose() *
              (Scalar(2) * r * exp_v - exp_2v.asDiagonal() * (c_i + μe)) +
          μβ_1e;
  };

  // r is sqrt_μ
  auto compute_step = [&](Scalar r) -> Step {
    Step step;

    // p = pˣ
    Eigen::Vector<Scalar, Eigen::Dynamic> p = solver.solve(rhs);
    step.p_x = p.segment(0, x.rows());

    // pᵛ = e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ) − √(μ)eᵛ∘w
    step.p_v = Eigen::Vector<Scalar, Eigen::Dynamic>::Ones(v.rows()) -
               Scalar(1) / r * exp_v.asDiagonal() * (A_i * step.p_x + c_i) -
               r * exp_v;

    return step;
  };

  // Initializes the barrier parameter for the current iterate.
  //
  // Returns true on success and false on failure.
  auto init_barrier_parameter = [&] {
    build_rhs(Scalar(1e15));
    Eigen::Vector<Scalar, Eigen::Dynamic> p_v_0 =
        compute_step(Scalar(1e15)).p_v;
    build_rhs(Scalar(1));
    Eigen::Vector<Scalar, Eigen::Dynamic> p_v_1 =
        compute_step(Scalar(1)).p_v - p_v_0;

    // See section 3.2.3 of [6]
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
  // This should be run when the error estimate is below a desired threshold for
  // the current barrier parameter.
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
        // Search bias that determines which side of range to check. < 0.5 is
        // closer to lower bound and > 0.5 is closer to upper bound.
        constexpr Scalar search_bias(0.75);

        Scalar sqrt_μ_mid = (Scalar(1) - search_bias) * sqrt_μ_lower +
                            search_bias * sqrt_μ_upper;

        build_rhs(sqrt_μ_mid);
        Eigen::Vector<Scalar, Eigen::Dynamic> p_v =
            compute_step(sqrt_μ_mid).p_v;
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

      build_rhs(Scalar(1e15));
      Eigen::Vector<Scalar, Eigen::Dynamic> p_v_0 =
          compute_step(Scalar(1e15)).p_v;
      build_rhs(Scalar(1));
      Eigen::Vector<Scalar, Eigen::Dynamic> p_v_1 =
          compute_step(Scalar(1)).p_v - p_v_0;

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
    }
  };

  // Variables for determining when a step is acceptable
  constexpr Scalar α_reduction_factor(0.75);
  constexpr Scalar α_min(1e-7);

  int full_step_rejected_counter = 0;

  // Error estimate
  Scalar E_0 = std::numeric_limits<Scalar>::infinity();

  // Prints final solver diagnostics when the solver exits
  scope_exit exit{[&] {
    if (options.diagnostics) {
      solver_prof.stop();
      if (iterations > 0) {
        print_bottom_iteration_diagnostics();
      }
      print_solver_diagnostics(solve_profilers);
    }
  }};

  Scalar prev_p_v_infnorm = std::numeric_limits<Scalar>::infinity();
  bool μ_initialized = false;

  // Watchdog (nonmonotone) variables. If a line search fails, accept up to this
  // many steps in a row in case the dual variable steps allow the primal steps
  // to make progress again.
  constexpr int watchdog_max = 1;
  int watchdog_count = 0;

  while (E_0 > Scalar(options.tolerance)) {
    ScopedProfiler inner_iter_profiler{inner_iter_prof};
    ScopedProfiler feasibility_check_profiler{feasibility_check_prof};

    // Check for local inequality constraint infeasibility
    if (is_inequality_locally_infeasible(A_i, c_i)) {
      if (options.diagnostics) {
        print_c_i_local_infeasibility_error(c_i);
      }

      return ExitStatus::LOCALLY_INFEASIBLE;
    }

    // Check for diverging iterates
    if (x.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !x.allFinite() ||
        v.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !v.allFinite()) {
      return ExitStatus::DIVERGING_ITERATES;
    }

    feasibility_check_profiler.stop();
    ScopedProfiler iter_callbacks_profiler{iter_callbacks_prof};

    // Call iteration callbacks
    for (const auto& callback : iteration_callbacks) {
      if (callback({iterations, x, g, H, Eigen::SparseMatrix<Scalar>{}, A_i})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    iter_callbacks_profiler.stop();

    if (auto status = build_and_compute_lhs(); status != ExitStatus::SUCCESS) {
      return status;
    }

    // Update the barrier parameter if necessary
    if (!μ_initialized) {
      init_barrier_parameter();
      μ_initialized = true;
    } else if (is_nlp) {
      Scalar E_sqrt_μ = error_estimate<Scalar>(g, A_i, c_i, v, sqrt_μ);
      if (E_sqrt_μ <= Scalar(10) * sqrt_μ * sqrt_μ) {
        ScopedProfiler μ_update_profiler{μ_update_prof};
        update_barrier_parameter();
      }
    } else if (prev_p_v_infnorm <= Scalar(1)) {
      ScopedProfiler μ_update_profiler{μ_update_prof};
      update_barrier_parameter();
    }

    ScopedProfiler kkt_system_solve_profiler{kkt_system_solve_prof};

    build_rhs(sqrt_μ);

    // Solve the Newton-KKT system for the step
    Step step = compute_step(sqrt_μ);

    kkt_system_solve_profiler.stop();
    ScopedProfiler line_search_profiler{line_search_prof};

    constexpr Scalar α_max(1);
    Scalar α(1);

    // αₖᵛ = min(1, 1/|pᵛ|_∞²)
    Scalar p_v_infnorm = step.p_v.template lpNorm<Eigen::Infinity>();
    Scalar α_v = std::min(Scalar(1), Scalar(1) / (p_v_infnorm * p_v_infnorm));
    prev_p_v_infnorm = p_v_infnorm;

    // Loop until a step is accepted
    while (1) {
      Eigen::Vector<Scalar, Eigen::Dynamic> trial_x = x + α * step.p_x;
      Eigen::Vector<Scalar, Eigen::Dynamic> trial_v = v + α_v * step.p_v;

      Scalar trial_f = matrices.f(trial_x);
      Eigen::Vector<Scalar, Eigen::Dynamic> trial_c_i = matrices.c_i(trial_x);

      // If f(xₖ + αpₖˣ) or cᵢ(xₖ + αpₖˣ) aren't finite, reduce step size
      // immediately
      if (!isfinite(trial_f) || !trial_c_i.allFinite()) {
        // Reduce step size
        α *= α_reduction_factor;

        if (α < α_min) {
          return ExitStatus::LINE_SEARCH_FAILED;
        }
        continue;
      }

      Eigen::Vector<Scalar, Eigen::Dynamic> trial_s;
      if (options.feasible_ipm && c_i.cwiseGreater(Scalar(0)).all()) {
        // If the inequality constraints are all feasible, prevent them from
        // becoming infeasible again.
        //
        //   cᵢ − √(μ)e⁻ᵛ = 0
        //   √(μ)e⁻ᵛ = cᵢ
        //   e⁻ᵛ = 1/√(μ) cᵢ
        //   −v = ln(1/√(μ) cᵢ)
        //   v = −ln(1/√(μ) cᵢ)
        trial_s = c_i;
        trial_v = -(c_i * (Scalar(1) / sqrt_μ)).array().log().matrix();
      } else {
        trial_s = sqrt_μ * (-trial_v).array().exp().matrix();
      }

      // Check whether filter accepts trial iterate
      if (filter.try_add(FilterEntry{trial_f, trial_v, trial_c_i, sqrt_μ}, α)) {
        // Accept step
        watchdog_count = 0;
        break;
      }

#if 0
      Scalar prev_constraint_violation =
          c_e.template lpNorm<1>() + (c_i - s).template lpNorm<1>();
      Scalar next_constraint_violation =
          trial_c_e.template lpNorm<1>() +
          (trial_c_i - trial_s).template lpNorm<1>();

      // Second-order corrections
      //
      // If first trial point was rejected and constraint violation stayed the
      // same or went up, apply second-order corrections
      if (α == α_max &&
          next_constraint_violation >= prev_constraint_violation) {
        // Apply second-order corrections. See section 2.4 of [2].
        auto soc_step = step;

        Scalar α_v_soc = α_v;
        Eigen::Vector<Scalar, Eigen::Dynamic> c_e_soc = c_e;

        bool step_acceptable = false;
        for (int soc_iteration = 0; soc_iteration < 5 && !step_acceptable;
             ++soc_iteration) {
          ScopedProfiler soc_profiler{soc_prof};

          scope_exit soc_exit{[&] {
            soc_profiler.stop();

            if (options.diagnostics) {
              print_iteration_diagnostics(
                  iterations,
                  step_acceptable ? IterationType::ACCEPTED_SOC
                                  : IterationType::REJECTED_SOC,
                  soc_profiler.current_duration(),
                  error_estimate<Scalar>(g, A_e, trial_c_e, trial_y), trial_f,
                  trial_c_e.template lpNorm<1>() +
                      (trial_c_i - trial_s).template lpNorm<1>(),
                  sqrt_μ * sqrt_μ, solver.hessian_regularization(), Scalar(1),
                  Scalar(1), α_reduction_factor, α_v_soc);
            }
          }};

          // Rebuild Newton-KKT rhs with updated constraint values.
          //
          // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘cᵢ)]
          //        [              cₑˢᵒᶜ              ]
          //
          // where cₑˢᵒᶜ = c(xₖ) + c(xₖ + αpₖˣ)
          c_e_soc += trial_c_e;
          rhs.bottomRows(y.rows()) = -c_e_soc;

          // Solve the Newton-KKT system
          soc_step = compute_step(sqrt_μ);

          trial_x = x + soc_step.p_x;
          trial_y = y + soc_step.p_y;

          // αₖᵛ = 1/max(1, |pᵛ|_∞²)
          Scalar p_v_infnorm = step.p_v.template lpNorm<Eigen::Infinity>();
          α_v_soc = Scalar(1) / std::max(Scalar(1), p_v_infnorm * p_v_infnorm);

          trial_v = v + α_v_soc * soc_step.p_v;
          trial_s = sqrt_μ * (-trial_v).array().exp().matrix();

          trial_f = matrices.f(trial_x);
          trial_c_e = matrices.c_e(trial_x);
          trial_c_i = matrices.c_i(trial_x);

          // Constraint violation scale factor for second-order corrections
          constexpr Scalar κ_soc(0.99);

          // If constraint violation hasn't been sufficiently reduced, stop
          // making second-order corrections
          next_constraint_violation =
              trial_c_e.template lpNorm<1>() +
              (trial_c_i - trial_s).template lpNorm<1>();
          if (next_constraint_violation > κ_soc * prev_constraint_violation) {
            break;
          }

          // Check whether filter accepts trial iterate
          if (filter.try_add(
                  FilterEntry{trial_f, trial_v, trial_c_e, trial_c_i, sqrt_μ},
                  α)) {
            step = soc_step;
            α = Scalar(1);
            α_v = α_v_soc;
            step_acceptable = true;
          }
        }

        if (step_acceptable) {
          // Accept step
          watchdog_count = 0;
          break;
        }
      }
#endif

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
              filter.back().constraint_violation / Scalar(10)) {
        filter.max_constraint_violation *= Scalar(0.1);
        filter.reset();
        continue;
      }

      // Reduce step size
      α *= α_reduction_factor;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, report line search failure.
      if (α < α_min) {
        Scalar current_kkt_error = kkt_error<Scalar>(g, A_i, c_i, v, sqrt_μ);

        trial_x = x + α_max * step.p_x;
        trial_v = v + α_v * step.p_v;

        trial_c_i = matrices.c_i(trial_x);

        Scalar next_kkt_error =
            kkt_error<Scalar>(matrices.g(trial_x), matrices.A_i(trial_x),
                              trial_c_i, trial_v, sqrt_μ);

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (next_kkt_error <= Scalar(0.999) * current_kkt_error) {
          α = α_max;

          // Accept step
          watchdog_count = 0;
          break;
        }

        // If the dual step is making progress, accept the whole step anyway
        if (p_v_infnorm > α_min && watchdog_count < watchdog_max) {
          // Accept step
          ++watchdog_count;
          break;
        }

        return ExitStatus::LINE_SEARCH_FAILED;
      }
    }

    line_search_profiler.stop();

    // If full step was accepted, reset full-step rejected counter
    if (α == α_max) {
      full_step_rejected_counter = 0;
    }

    // xₖ₊₁ = xₖ + αₖpₖˣ
    // vₖ₊₁ = vₖ + αₖᵛpₖᵛ
    x += α * step.p_x;
    v += α_v * step.p_v;

    exp_v = v.array().exp().matrix();
    exp_neg_v = exp_v.cwiseInverse();
    exp_2v = exp_v.cwiseProduct(exp_v);
    s = sqrt_μ * exp_neg_v;

    // Update autodiff for Jacobians and Hessian
    f = matrices.f(x);
    A_i = matrices.A_i(x);
    g = matrices.g(x);
    H = matrices.H(x, v, sqrt_μ);

    ScopedProfiler next_iter_prep_profiler{next_iter_prep_prof};

    c_i = matrices.c_i(x);

    // Update the error estimate
    E_0 = error_estimate<Scalar>(g, A_i, c_i, v, sqrt_μ_min);

    next_iter_prep_profiler.stop();
    inner_iter_profiler.stop();

    if (options.diagnostics) {
      print_iteration_diagnostics(
          iterations, IterationType::NORMAL,
          inner_iter_profiler.current_duration(), E_0, f,
          (c_i - s).template lpNorm<1>(), sqrt_μ * sqrt_μ,
          solver.hessian_regularization(), α, α_max, α_reduction_factor, α_v);
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

  return ExitStatus::SUCCESS;
}

extern template SLEIPNIR_DLLEXPORT ExitStatus interior_point(
    const InteriorPointMatrixCallbacks<double>& matrix_callbacks, bool is_nlp,
    std::span<std::function<bool(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::Vector<double, Eigen::Dynamic>& x);

}  // namespace slp
