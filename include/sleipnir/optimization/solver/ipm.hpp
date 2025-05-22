// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numbers>
#include <span>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/ipm_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/util/all_finite.hpp"
#include "sleipnir/optimization/solver/util/feasibility_restoration.hpp"
#include "sleipnir/optimization/solver/util/filter.hpp"
#include "sleipnir/optimization/solver/util/kkt_error.hpp"
#include "sleipnir/optimization/solver/util/kkt_solver.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/print_diagnostics.hpp"
#include "sleipnir/util/profiler.hpp"
#include "sleipnir/util/scope_exit.hpp"
#include "sleipnir/util/symbol_exports.hpp"

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
ExitStatus ipm(const IPMMatrixCallbacks<Scalar>& matrix_callbacks, bool is_nlp,
               std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
                   iteration_callbacks,
               const Options& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
               const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
               Eigen::Vector<Scalar, Eigen::Dynamic>& x) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  DenseVector y = DenseVector::Zero(matrix_callbacks.num_equality_constraints);
  DenseVector v =
      DenseVector::Zero(matrix_callbacks.num_inequality_constraints);
  Scalar sqrt_μ = Scalar(0.1) * matrix_callbacks.scaling.f;
  int iterations = 0;

  return ipm(matrix_callbacks, is_nlp, iteration_callbacks, options, false,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
             bound_constraint_mask,
#endif
             x, v, sqrt_μ, iterations);
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
/// @param[in,out] x The initial guess and output location for the decision
///     variables.
/// @param[in,out] v The initial guess and output location for the log-domain
///     variables.
/// @param[in,out] sqrt_μ The initial guess and output location for the barrier
///     parameter.
/// @param[in,out] iterations The iteration counter.
/// @return The exit status.
template <typename Scalar>
ExitStatus ipm(const IPMMatrixCallbacks<Scalar>& matrix_callbacks, bool is_nlp,
               std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
                   iteration_callbacks,
               const Options& options, bool in_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
               const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
               Eigen::Vector<Scalar, Eigen::Dynamic>& x,
               Eigen::Vector<Scalar, Eigen::Dynamic>& v, Scalar& sqrt_μ,
               int& iterations) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using SparseVector = Eigen::SparseVector<Scalar>;

  /// Interior-point method step direction.
  struct Step {
    /// Decision variable primal step.
    DenseVector p_x;
    /// Log-domain variable step.
    DenseVector p_v;
  };

  using std::isfinite;
  using std::sqrt;

  const auto solve_start_time = std::chrono::steady_clock::now();

  gch::small_vector<SolveProfiler> solve_profilers;
  solve_profilers.emplace_back("solver");
  solve_profilers.emplace_back("↳ setup");
  solve_profilers.emplace_back("↳ iteration");
  solve_profilers.emplace_back("  ↳ callbacks");
  solve_profilers.emplace_back("  ↳ μ update");
  solve_profilers.emplace_back("  ↳ KKT matrix build");
  solve_profilers.emplace_back("  ↳ KKT matrix decomp");
  solve_profilers.emplace_back("  ↳ KKT system solve");
  solve_profilers.emplace_back("  ↳ line search");
  solve_profilers.emplace_back("    ↳ SOC");
  solve_profilers.emplace_back("  ↳ feas. restoration");
  solve_profilers.emplace_back("  ↳ f(x)");
  solve_profilers.emplace_back("  ↳ ∇f(x)");
  solve_profilers.emplace_back("  ↳ ∇²ₓₓL");
  solve_profilers.emplace_back("  ↳ ∇²ₓₓL_c");
  solve_profilers.emplace_back("  ↳ cᵢ(x)");
  solve_profilers.emplace_back("  ↳ ∂cᵢ/∂x");

  auto& solver_prof = solve_profilers[0];
  auto& setup_prof = solve_profilers[1];
  auto& inner_iter_prof = solve_profilers[2];
  auto& iter_callbacks_prof = solve_profilers[3];
  auto& μ_update_prof = solve_profilers[4];
  auto& kkt_matrix_build_prof = solve_profilers[5];
  auto& kkt_matrix_decomp_prof = solve_profilers[6];
  auto& kkt_system_solve_prof = solve_profilers[7];
  auto& line_search_prof = solve_profilers[8];
  auto& soc_prof = solve_profilers[9];
  auto& feasibility_restoration_prof = solve_profilers[10];

  // Set up profiled matrix callbacks
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  auto& f_prof = solve_profilers[11];
  auto& g_prof = solve_profilers[12];
  auto& H_prof = solve_profilers[13];
  auto& H_c_prof = solve_profilers[14];
  auto& c_i_prof = solve_profilers[15];
  auto& A_i_prof = solve_profilers[16];

  IPMMatrixCallbacks<Scalar> matrices{
      matrix_callbacks.num_decision_variables,
      matrix_callbacks.num_equality_constraints,
      matrix_callbacks.num_inequality_constraints,
      [&](const DenseVector& x) -> Scalar {
        ScopedProfiler prof{f_prof};
        return matrix_callbacks.f(x);
      },
      [&](const DenseVector& x) -> SparseVector {
        ScopedProfiler prof{g_prof};
        return matrix_callbacks.g(x);
      },
      [&](const DenseVector& x, const DenseVector& v,
          Scalar sqrt_μ) -> SparseMatrix {
        ScopedProfiler prof{H_prof};
        return matrix_callbacks.H(x, v, sqrt_μ);
      },
      [&](const DenseVector& x, const DenseVector& v,
          Scalar sqrt_μ) -> SparseMatrix {
        ScopedProfiler prof{H_c_prof};
        return matrix_callbacks.H_c(x, v, sqrt_μ);
      },
      [&](const DenseVector& x) -> DenseVector {
        ScopedProfiler prof{c_i_prof};
        return matrix_callbacks.c_i(x);
      },
      [&](const DenseVector& x) -> SparseMatrix {
        ScopedProfiler prof{A_i_prof};
        return matrix_callbacks.A_i(x);
      },
      matrix_callbacks.scaling};
#else
  const auto& matrices = matrix_callbacks;
#endif

  solver_prof.start();
  setup_prof.start();

  Scalar f = matrices.f(x);
  SparseVector g = matrices.g(x);
  SparseMatrix H = matrices.H(x, v, sqrt_μ);
  DenseVector c_i = matrices.c_i(x);
  SparseMatrix A_i = matrices.A_i(x);

  // Ensure matrix callback dimensions are consistent
  slp_assert(g.rows() == matrices.num_decision_variables);
  slp_assert(H.rows() == matrices.num_decision_variables);
  slp_assert(H.cols() == matrices.num_decision_variables);
  slp_assert(c_i.rows() == matrices.num_inequality_constraints);
  slp_assert(A_i.rows() == matrices.num_inequality_constraints);
  slp_assert(A_i.cols() == matrices.num_decision_variables);

  DenseVector trial_x;
  DenseVector trial_v;

  Scalar trial_f;
  DenseVector trial_c_i;

  // Check whether initial guess has finite cost, constraints, and derivatives
  if (!isfinite(f) || !all_finite(g) || !all_finite(H) || !c_i.allFinite() ||
      !all_finite(A_i)) {
    return ExitStatus::NONFINITE_INITIAL_GUESS;
  }

  // Barrier parameter minimum
  const Scalar sqrt_μ_min =
      matrices.scaling.f * sqrt(Scalar(options.tolerance) / Scalar(10));

  // The barrier subproblem is considered solved when its KKT error is at most
  // κ_ε·μ
  constexpr Scalar κ_ε(10);

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
  // e²ᵛ
  DenseVector exp_2v = exp_v.cwiseProduct(exp_v);
  // s = √(μ)e⁻ᵛ
  DenseVector s = sqrt_μ * exp_neg_v;

  Filter<Scalar> filter{(c_i - s).template lpNorm<1>()};

  // Kept outside the loop so its storage can be reused
  gch::small_vector<Eigen::Triplet<Scalar>> triplets;

  const int lhs_rows = matrices.num_decision_variables;
  KKTSolver<Scalar> solver{
      // Use sparse solver if lower triangle fills < 25% of system
      H.nonZeros() + (A_i.transpose() * A_i)
                         .template triangularView<Eigen::Lower>()
                         .eval()
                         .nonZeros() <
          0.25 * lhs_rows * lhs_rows,
      matrices.num_decision_variables, matrices.num_equality_constraints,
      // Constraint regularization is forced to zero in feasibility restoration
      // because the equality constraint Jacobian cannot be rank-deficient
      in_feasibility_restoration ? Scalar(0) : Scalar(1e-10)};
  SparseMatrix lhs(matrices.num_decision_variables,
                   matrices.num_decision_variables);
  DenseVector rhs{x.rows()};

  setup_prof.stop();

  // r is √(μ)
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

  // r is √(μ). μ_w is the barrier parameter used for the infeasibility and
  // gradient perturbations μw and μβ₁e. It's normally r², but setting it to
  // zero makes pᵛ affine in 1/r, which the barrier parameter initialization
  // relies on.
  constexpr Scalar β_1(1e-4);
  auto build_rhs = [&](Scalar r, Scalar μ_w) {
    // rhs = −[∇f − Aᵢᵀ(2√(μ)eᵛ − e²ᵛ∘(cᵢ + μw)) − μβ₁e]
    rhs = -g + A_i.transpose() *
                   (Scalar(2) * r * exp_v -
                    exp_2v.asDiagonal() * (c_i.array() + μ_w).matrix());
    rhs.array() += μ_w * β_1;
  };

  // r is √(μ). μ_w is the barrier parameter used for the infeasibility
  // perturbation μw.
  auto compute_step = [&](Scalar r, Scalar μ_w) -> Step {
    Step step;

    // p = pˣ
    DenseVector p = solver.solve(rhs);
    step.p_x = p.segment(0, x.rows());

    // pᵛ = e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ + μw)
    step.p_v = DenseVector::Ones(v.rows()) -
               Scalar(1) / r * exp_v.asDiagonal() *
                   ((A_i * step.p_x + c_i).array() + μ_w).matrix();

    return step;
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
    build_rhs(Scalar(1e15), Scalar(0));
    DenseVector p_v_0 = compute_step(Scalar(1e15), Scalar(0)).p_v;
    build_rhs(Scalar(1), Scalar(0));
    DenseVector p_v_1 = compute_step(Scalar(1), Scalar(0)).p_v - p_v_0;

    // See section 3.2.3 of [5]
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
    using std::log1p;
    using std::pow;

    if (sqrt_μ <= sqrt_μ_min) {
      return false;
    }

    const Scalar μ = sqrt_μ * sqrt_μ;
    const Scalar μ_min = sqrt_μ_min * sqrt_μ_min;

    // Returns the direction (pˣ, qᵛ) for the given η
    auto compute_aggressive_step = [&](Scalar η) -> Step {
      const Scalar μ_w = (Scalar(1) - η) * μ;

      // rhs = −[∇f − Aᵢᵀ((2 − η)√(μ)eᵛ − e²ᵛ∘(cᵢ + (1 − η)μw)) −
      //         (1 − η)μβ₁e]
      rhs = -g + A_i.transpose() *
                     ((Scalar(2) - η) * sqrt_μ * exp_v -
                      exp_2v.asDiagonal() * (c_i.array() + μ_w).matrix());
      rhs.array() += μ_w * ipm_β_1<Scalar>;

      Step step;
      step.p_x = solver.solve(rhs);

      // qᵛ = e − 1/√(μ) eᵛ∘(Aᵢpˣ + cᵢ + (1 − η)μw)
      step.p_v = DenseVector::Ones(v.rows()) -
                 Scalar(1) / sqrt_μ * exp_v.asDiagonal() *
                     ((A_i * step.p_x + c_i).array() + μ_w).matrix();

      return step;
    };

    // Mehrotra's heuristic: σ = (1 − αₐff)³ where αₐff is the
    // fraction-to-the-boundary step size of the affine direction
    const Scalar α_aff = max_step_size(compute_aggressive_step(Scalar(1)).p_v);
    const Scalar η = Scalar(1) - pow(Scalar(1) - α_aff, Scalar(3));

    const Step step = compute_aggressive_step(η);

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
      trial_c_i = matrices.c_i(trial_x);
      trial_f = matrices.f(trial_x);

      if (isfinite(trial_f) && trial_c_i.allFinite() &&
          (!keep_feasible || trial_c_i.cwiseGreater(Scalar(0)).all())) {
        SparseVector trial_g = matrices.g(trial_x);
        SparseMatrix trial_A_i = matrices.A_i(trial_x);

        Scalar E = kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
            trial_g, trial_A_i, trial_c_i, trial_v, trial_sqrt_μ, trial_μ);
        if (E <= κ_ε * trial_μ) {
          x = trial_x;
          v = trial_v;
          sqrt_μ = trial_sqrt_μ;

          f = trial_f;
          c_i = trial_c_i;
          g = std::move(trial_g);
          A_i = std::move(trial_A_i);

          exp_v = v.array().exp().matrix();
          exp_neg_v = exp_v.cwiseInverse();
          exp_2v = exp_v.cwiseProduct(exp_v);

          // Reset the filter when the barrier parameter is updated
          filter.reset();

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
    exp_2v = exp_v.cwiseProduct(exp_v);

    // Reset the filter when the barrier parameter is updated
    filter.reset();

    return true;
  };

  // Variables for determining when a step is acceptable
  constexpr Scalar α_reduction_factor(1.0 / std::numbers::sqrt2);
  constexpr Scalar α_min(1e-7);

  int full_step_rejected_counter = 0;

  // |pᵛ|_∞ of the previous normal step
  Scalar prev_p_v_infnorm = std::numeric_limits<Scalar>::infinity();

  // Error
  Scalar E_0 = unscaled_kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
      matrices.scaling, g, A_i, c_i, v, sqrt_μ, Scalar(0));

  // Prints final solver diagnostics when the solver exits
  scope_exit exit{[&] {
    if (options.diagnostics) {
      solver_prof.stop();

      if (in_feasibility_restoration) {
        return;
      }

      if (iterations > 0) {
        print_bottom_iteration_diagnostics();
      }
      print_solver_diagnostics(solve_profilers);
    }
  }};

  bool μ_initialized = false;

  // Watchdog (nonmonotone) variables. If a line search fails, accept up to this
  // many steps in a row in case the dual variable steps allow the primal steps
  // to make progress again.
  constexpr int watchdog_max = 5;
  int watchdog_count = 0;

  // Print initial iterate diagnostics
  if (options.diagnostics) {
    print_initial_iterate_diagnostics(E_0, f, (c_i - s).template lpNorm<1>(),
                                      sqrt_μ * sqrt_μ);
  }

  while (E_0 > Scalar(options.tolerance)) {
    ScopedProfiler inner_iter_profiler{inner_iter_prof};

    // Check for diverging iterates
    if (x.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !x.allFinite() ||
        v.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !v.allFinite()) {
      return ExitStatus::DIVERGING_ITERATES;
    }

    ScopedProfiler iter_callbacks_profiler{iter_callbacks_prof};

    // Call iteration callbacks
    for (const auto& callback : iteration_callbacks) {
      if (callback({iterations, x, {}, v, g, H, {}, A_i})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    iter_callbacks_profiler.stop();

    if (auto status = build_and_compute_lhs(); status != ExitStatus::SUCCESS) {
      return status;
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
        exp_2v = exp_v.cwiseProduct(exp_v);
        v_reset = true;
      }
    } else if (
        // The barrier subproblem is approximately solved, and the iterate is
        // close enough to the central path that the previous normal step
        // changed each slack by less than 100%
        prev_p_v_infnorm <= Scalar(1) &&
        kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
            g, A_i, c_i, v, sqrt_μ, sqrt_μ * sqrt_μ) <= κ_ε * sqrt_μ * sqrt_μ) {
      ScopedProfiler μ_update_profiler{μ_update_prof};

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
      if (is_nlp || v_reset) {
        if (auto status = build_and_compute_lhs();
            status != ExitStatus::SUCCESS) {
          return status;
        }
      }
    }

    ScopedProfiler kkt_system_solve_profiler{kkt_system_solve_prof};

    build_rhs(sqrt_μ, sqrt_μ * sqrt_μ);

    // Solve the Newton-KKT system for the step
    Step step = compute_step(sqrt_μ, sqrt_μ * sqrt_μ);

    kkt_system_solve_profiler.stop();
    ScopedProfiler line_search_profiler{line_search_prof};

    // The slack variables s = √(μ)e⁻ᵛ are part of the primal step, so x and v
    // share a step size, capped by the fraction-to-the-boundary rule for the
    // slack step.
    Scalar p_v_infnorm = step.p_v.template lpNorm<Eigen::Infinity>();
    prev_p_v_infnorm = p_v_infnorm;
    const Scalar α_max = max_step_size(step.p_v);
    Scalar α = α_max;

    // If the cap on the step size is already below the minimum, the Newton
    // step drives some slack variables toward zero by orders of magnitude
    // (e.g., violated inequality constraints). Backtracking can't help, so
    // invoke feasibility restoration.
    bool call_feasibility_restoration = α_max < α_min;

    const FilterEntry<Scalar> current_entry{f, v, c_i, sqrt_μ};

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
      if (!isfinite(trial_f) || !trial_c_i.allFinite() ||
          (keep_feasible && !trial_c_i.cwiseGreater(Scalar(0)).all())) {
        // Reduce step size
        α *= α_reduction_factor;

        if (α < α_min) {
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
      FilterEntry trial_entry{trial_f, trial_v, trial_c_i, sqrt_μ};
      if (filter.try_add(current_entry, trial_entry, D_ϕ, α)) {
        // Accept step
        watchdog_count = 0;
        break;
      }

      Scalar prev_constraint_violation = (c_i - s).template lpNorm<1>();
      Scalar next_constraint_violation =
          (trial_c_i - trial_s).template lpNorm<1>();

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
          ScopedProfiler soc_profiler{soc_prof};

          scope_exit soc_exit{[&] {
            soc_profiler.stop();

            if (options.diagnostics && step_acceptable) {
              print_iteration_diagnostics(
                  iterations, IterationType::SECOND_ORDER_CORRECTION,
                  soc_profiler.current_duration(),
                  unscaled_kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
                      matrices.scaling, g, A_i, trial_c_i, trial_v, sqrt_μ,
                      Scalar(0)),
                  trial_f, (trial_c_i - trial_s).template lpNorm<1>(),
                  sqrt_μ * sqrt_μ, solver.hessian_regularization(),
                  solver.constraint_jacobian_regularization(),
                  soc_step.p_x.template lpNorm<Eigen::Infinity>(),
                  soc_step.p_v.template lpNorm<Eigen::Infinity>(), α_soc, α_soc,
                  α_reduction_factor, α_soc);
            }
          }};

          // Rebuild Newton-KKT rhs with updated constraint values.
          //
          // Since e²ᵛ∘s = √(μ)eᵛ, the rhs can be written in terms of the
          // inequality constraint residual cᵢ − s.
          //
          // rhs = −[∇f − Aᵢᵀ(√(μ)eᵛ − e²ᵛ∘(cᵢ − s)ˢᵒᶜ)]
          //
          // where
          //
          //   (cᵢ − s)ˢᵒᶜ =
          //     αˢᵒᶜ(cᵢ(xₖ) − sₖ) + cᵢ(xₖ + αˢᵒᶜpˣ) − sₖ(αˢᵒᶜpᵛ)
          c_i_minus_s_soc = α_soc * c_i_minus_s_soc + trial_c_i - trial_s;
          rhs = -g + A_i.transpose() * (sqrt_μ * exp_v -
                                        exp_2v.asDiagonal() * c_i_minus_s_soc);

          // Solve the Newton-KKT system
          //
          //   pᵛ = −1/√(μ) eᵛ∘(Aᵢpˣ + (cᵢ − s)ˢᵒᶜ)
          DenseVector p = solver.solve(rhs);
          soc_step.p_x = p.segment(0, x.rows());
          soc_step.p_v = -(Scalar(1) / sqrt_μ) * exp_v.asDiagonal() *
                         (A_i * soc_step.p_x + c_i_minus_s_soc);

          α_soc = max_step_size(soc_step.p_v);

          trial_x = x + α_soc * soc_step.p_x;
          trial_v = step_v(soc_step.p_v, α_soc);
          trial_s = sqrt_μ * (-trial_v).array().exp().matrix();

          trial_f = matrices.f(trial_x);
          trial_c_i = matrices.c_i(trial_x);

          // Check whether filter accepts trial iterate
          FilterEntry trial_entry{trial_f, trial_v, trial_c_i, sqrt_μ};
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
              (trial_c_i - trial_s).template lpNorm<1>();
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
        Scalar current_kkt_error = kkt_error<Scalar, KKTErrorType::ONE_NORM>(
            g, A_i, c_i, v, sqrt_μ, sqrt_μ * sqrt_μ);

        trial_x = x + α_max * step.p_x;
        trial_v = step_v(step.p_v, α_max);

        trial_f = matrices.f(trial_x);
        trial_c_i = matrices.c_i(trial_x);

        Scalar next_kkt_error = kkt_error<Scalar, KKTErrorType::ONE_NORM>(
            matrices.g(trial_x), matrices.A_i(trial_x), trial_c_i, trial_v,
            sqrt_μ, sqrt_μ * sqrt_μ);

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (next_kkt_error <= Scalar(0.999) * current_kkt_error) {
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

        call_feasibility_restoration = true;
        break;
      }
    }

    line_search_profiler.stop();

    if (call_feasibility_restoration) {
      ScopedProfiler feasibility_restoration_profiler{
          feasibility_restoration_prof};

      // If already in feasibility restoration mode, running it again won't help
      if (in_feasibility_restoration) {
        return ExitStatus::FEASIBILITY_RESTORATION_FAILED;
      }

      FilterEntry initial_entry{matrices.f(x), v, c_i, sqrt_μ};

      // Square root of the feasibility restoration barrier parameter, which
      // feasibility restoration updates in place
      Scalar fr_sqrt_μ = sqrt_μ;

      // Feasibility restoration phase
      gch::small_vector<std::function<bool(const IterationInfo<Scalar>& info)>>
          callbacks;
      for (auto& callback : iteration_callbacks) {
        callbacks.emplace_back(callback);
      }
      callbacks.emplace_back([&](const IterationInfo<Scalar>& info) {
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

        DenseVector trial_c_i = matrices.c_i(trial_x);

        // If the current iterate sufficiently reduces constraint violation and
        // is accepted by the normal filter, stop feasibility restoration.
        //
        // The step is the displacement from the iterate where feasibility
        // restoration started, so the directional derivative of the
        // log-barrier function is taken along that displacement with a step
        // size of 1.
        //
        //   D_ϕ = ∇f(x)ᵀ(xₜᵣᵢₐₗ − x) + μ∑ᵢ (vₜᵣᵢₐₗ − v)ᵢ
        FilterEntry trial_entry{matrices.f(trial_x), trial_v, trial_c_i,
                                sqrt_μ};
        const Scalar D_ϕ_restoration = g.transpose() * (trial_x - x) +
                                       sqrt_μ * sqrt_μ * (trial_v - v).sum();
        return trial_entry.constraint_violation <
                   Scalar(0.9) * initial_entry.constraint_violation &&
               filter.try_add(initial_entry, trial_entry, D_ϕ_restoration,
                              Scalar(1));
      });
      auto status =
          feasibility_restoration<Scalar>(matrices, is_nlp, callbacks, options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
                                          bound_constraint_mask,
#endif
                                          x, v, sqrt_μ, fr_sqrt_μ, iterations);

      if (status != ExitStatus::SUCCESS) {
        // Report failure
        return status;
      }

      f = matrices.f(x);
      c_i = matrices.c_i(x);
    } else {
      // If full step was accepted, reset full-step rejected counter
      if (α == α_max) {
        full_step_rejected_counter = 0;
      }

      // Update iterates
      x = trial_x;
      v = trial_v;

      f = trial_f;
      c_i = trial_c_i;
    }

    exp_v = v.array().exp().matrix();
    exp_neg_v = exp_v.cwiseInverse();
    exp_2v = exp_v.cwiseProduct(exp_v);
    s = sqrt_μ * exp_neg_v;

    // Update autodiff for Jacobians and Hessian
    A_i = matrices.A_i(x);
    g = matrices.g(x);
    H = matrices.H(x, v, sqrt_μ);

    // Update the error
    E_0 = unscaled_kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
        matrices.scaling, g, A_i, c_i, v, sqrt_μ, Scalar(0));

    inner_iter_profiler.stop();

    if (options.diagnostics) {
      print_iteration_diagnostics(
          iterations,
          in_feasibility_restoration ? IterationType::FEASIBILITY_RESTORATION
                                     : IterationType::NORMAL,
          inner_iter_profiler.current_duration(), E_0, f,
          (c_i - s).template lpNorm<1>(), sqrt_μ * sqrt_μ,
          solver.hessian_regularization(),
          solver.constraint_jacobian_regularization(),
          step.p_x.template lpNorm<Eigen::Infinity>(),
          step.p_v.template lpNorm<Eigen::Infinity>(), α, α_max,
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
    std::span<std::function<bool(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::Vector<double, Eigen::Dynamic>& x);

}  // namespace slp
