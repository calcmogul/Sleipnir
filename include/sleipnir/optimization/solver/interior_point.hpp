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
#include "sleipnir/optimization/solver/interior_point_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/util/all_finite.hpp"
#include "sleipnir/optimization/solver/util/append_as_triplets.hpp"
#include "sleipnir/optimization/solver/util/feasibility_restoration.hpp"
#include "sleipnir/optimization/solver/util/filter.hpp"
#include "sleipnir/optimization/solver/util/is_locally_infeasible.hpp"
#include "sleipnir/optimization/solver/util/kkt_error.hpp"
#include "sleipnir/optimization/solver/util/regularized_ldlt.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/print_diagnostics.hpp"
#include "sleipnir/util/profiler.hpp"
#include "sleipnir/util/scope_exit.hpp"
#include "sleipnir/util/symbol_exports.hpp"

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
ExitStatus interior_point(
    const InteriorPointMatrixCallbacks<Scalar>& matrix_callbacks, bool is_nlp,
    std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::Vector<Scalar, Eigen::Dynamic>& x) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  DenseVector y = DenseVector::Zero(matrix_callbacks.num_equality_constraints);
  DenseVector u =
      DenseVector::Zero(matrix_callbacks.num_inequality_constraints);
  DenseVector v =
      DenseVector::Zero(matrix_callbacks.num_inequality_constraints);
  Scalar sqrt_μ = Scalar(0.1) * matrix_callbacks.scaling.f;
  int iterations = 0;

  return interior_point(matrix_callbacks, is_nlp, iteration_callbacks, options,
                        false,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
                        bound_constraint_mask,
#endif
                        x, y, u, v, sqrt_μ, iterations);
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
/// @param[in,out] y The initial guess and output location for the equality
///     constraint dual variables.
/// @param[in,out] u The initial guess and output location for the log-domain
///     slack variables.
/// @param[in,out] v The initial guess and output location for the log-domain
///     dual variables.
/// @param[in,out] sqrt_μ The initial guess and output location for the barrier
///     parameter.
/// @param[in,out] iterations The iteration counter.
/// @return The exit status.
template <typename Scalar>
ExitStatus interior_point(
    const InteriorPointMatrixCallbacks<Scalar>& matrix_callbacks, bool is_nlp,
    std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options& options, bool in_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::Vector<Scalar, Eigen::Dynamic>& x,
    Eigen::Vector<Scalar, Eigen::Dynamic>& y,
    Eigen::Vector<Scalar, Eigen::Dynamic>& u,
    Eigen::Vector<Scalar, Eigen::Dynamic>& v, Scalar& sqrt_μ, int& iterations) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using SparseVector = Eigen::SparseVector<Scalar>;

  /// Interior-point method step direction.
  struct Step {
    /// Decision variable primal step.
    DenseVector p_x;
    /// Equality constraint dual step.
    DenseVector p_y;
    /// Log-domain slack variable step.
    DenseVector p_u;
    /// Log-domain dual variable step.
    DenseVector p_v;
  };

  using std::isfinite;
  using std::sqrt;

  const auto solve_start_time = std::chrono::steady_clock::now();

  gch::small_vector<SolveProfiler> solve_profilers;
  solve_profilers.emplace_back("solver");
  solve_profilers.emplace_back("↳ setup");
  solve_profilers.emplace_back("↳ iteration");
  solve_profilers.emplace_back("  ↳ feasibility check");
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
  solve_profilers.emplace_back("  ↳ cₑ(x)");
  solve_profilers.emplace_back("  ↳ ∂cₑ/∂x");
  solve_profilers.emplace_back("  ↳ cᵢ(x)");
  solve_profilers.emplace_back("  ↳ ∂cᵢ/∂x");

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
  auto& soc_prof = solve_profilers[10];
  auto& feasibility_restoration_prof = solve_profilers[11];

  // Set up profiled matrix callbacks
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  auto& f_prof = solve_profilers[12];
  auto& g_prof = solve_profilers[13];
  auto& H_prof = solve_profilers[14];
  auto& H_c_prof = solve_profilers[15];
  auto& c_e_prof = solve_profilers[16];
  auto& A_e_prof = solve_profilers[17];
  auto& c_i_prof = solve_profilers[18];
  auto& A_i_prof = solve_profilers[19];

  InteriorPointMatrixCallbacks<Scalar> matrices{
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
      [&](const DenseVector& x, const DenseVector& y, const DenseVector& v,
          Scalar sqrt_μ) -> SparseMatrix {
        ScopedProfiler prof{H_prof};
        return matrix_callbacks.H(x, y, v, sqrt_μ);
      },
      [&](const DenseVector& x, const DenseVector& y, const DenseVector& v,
          Scalar sqrt_μ) -> SparseMatrix {
        ScopedProfiler prof{H_c_prof};
        return matrix_callbacks.H_c(x, y, v, sqrt_μ);
      },
      [&](const DenseVector& x) -> DenseVector {
        ScopedProfiler prof{c_e_prof};
        return matrix_callbacks.c_e(x);
      },
      [&](const DenseVector& x) -> SparseMatrix {
        ScopedProfiler prof{A_e_prof};
        return matrix_callbacks.A_e(x);
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
  SparseMatrix H = matrices.H(x, y, v, sqrt_μ);
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
  DenseVector trial_y;
  DenseVector trial_u;
  DenseVector trial_v;

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
  //
  //   cᵢ − √(μ)e⁻ᵘ = 0
  //   √(μ)e⁻ᵘ = cᵢ
  //   e⁻ᵘ = 1/√(μ) cᵢ
  //   −u = ln(1/√(μ) cᵢ)
  //   u = −ln(1/√(μ) cᵢ)
  u = bound_constraint_mask.select(
      -(c_i * (Scalar(1) / sqrt_μ_min)).array().log().matrix(), u);
#endif

  // eᵘ
  DenseVector exp_u{u.array().exp().matrix()};
  // eᵛ
  DenseVector exp_v{v.array().exp().matrix()};
  // eᵘ⁺ᵛ
  DenseVector exp_u_plus_v = exp_u.cwiseProduct(exp_v);
  // eᵘ⁻ᵛ
  DenseVector exp_u_minus_v = exp_v.cwiseProduct(exp_v.cwiseInverse());
  // s = √(μ)e⁻ᵘ
  DenseVector s = sqrt_μ * exp_u.cwiseInverse();

  // Barrier parameter minimum
  const Scalar sqrt_μ_min =
      matrices.scaling.f * sqrt(Scalar(options.tolerance) / Scalar(10));

  Filter<Scalar> filter{c_e.template lpNorm<1>() +
                        (c_i - s).template lpNorm<1>()};

  // Kept outside the loop so its storage can be reused
  gch::small_vector<Eigen::Triplet<Scalar>> triplets;

  const int lhs_rows =
      matrices.num_decision_variables + matrices.num_equality_constraints;
  RegularizedLDLT<Scalar> solver{
      // Use sparse solver if lower triangle fills < 25% of system
      H.nonZeros() +
              (A_i.transpose() * A_i)
                  .template triangularView<Eigen::Lower>()
                  .eval()
                  .nonZeros() +
              A_e.nonZeros() <
          0.25 * lhs_rows * lhs_rows,
      matrices.num_decision_variables, matrices.num_equality_constraints,
      // Constraint regularization is forced to zero in feasibility restoration
      // because the equality constraint Jacobian cannot be rank-deficient
      in_feasibility_restoration ? Scalar(0) : Scalar(1e-10)};
  SparseMatrix lhs(
      matrices.num_decision_variables + matrices.num_equality_constraints,
      matrices.num_decision_variables + matrices.num_equality_constraints);
  DenseVector rhs{x.rows() + y.rows()};

  setup_prof.stop();

  // r is sqrt_μ
  auto build_and_compute_lhs = [&]() -> ExitStatus {
    ScopedProfiler kkt_matrix_build_profiler{kkt_matrix_build_prof};

    // lhs = [H + Aᵢᵀdiag(e²ᵛ)Aᵢ  Aₑᵀ]
    //       [        Aₑ           0 ]
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    const SparseMatrix top_left =
        H + (A_i.transpose() * exp_u_plus_v.asDiagonal() * A_i)
                .template triangularView<Eigen::Lower>();
    triplets.clear();
    triplets.reserve(top_left.nonZeros() + A_e.nonZeros());
    append_as_triplets(triplets, 0, 0, {top_left, A_e});
    lhs.setFromSortedTriplets(triplets.begin(), triplets.end());

    kkt_matrix_build_profiler.stop();
    ScopedProfiler kkt_matrix_decomp_profiler{kkt_matrix_decomp_prof};

    // Solve the Newton-KKT system
    //
    // [H + Aᵢᵀdiag(eᵘ⁺ᵛ)Aᵢ  Aₑᵀ][ pˣ] =
    // [        Aₑ            0 ][−pʸ]
    //     −[∇f − Aₑᵀy − Aᵢᵀ(−√(μ)eᵘ + 2√(μ)eᵛ − eᵘ⁺ᵛ∘cᵢ)]
    //      [                     cₑ                     ]
    if (solver.compute(lhs).info() != Eigen::Success) {
      return ExitStatus::FACTORIZATION_FAILED;
    } else {
      return ExitStatus::SUCCESS;
    }
  };

  // r is sqrt_μ
  auto build_rhs = [&](Scalar r) {
    // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(−√(μ)eᵘ + 2√(μ)eᵛ − eᵘ⁺ᵛ∘cᵢ)]
    //        [                     cₑ                     ]
    rhs.segment(0, x.rows()) =
        -g + A_e.transpose() * y +
        A_i.transpose() * (-r * exp_u + Scalar(2) * r * exp_v -
                           exp_u_plus_v.asDiagonal() * c_i);
    rhs.segment(x.rows(), y.rows()) = -c_e;
  };

  // r is sqrt_μ
  auto compute_step = [&](Scalar r) -> Step {
    Step step;

    // p = [ pˣ]
    //     [−pʸ]
    DenseVector p = solver.solve(rhs);
    step.p_x = p.segment(0, x.rows());
    step.p_y = -p.segment(x.rows(), y.rows());

    // pᵘ = e − 1/√(μ) eᵘ∘(Aᵢpˣ + cᵢ)
    step.p_u = DenseVector::Ones(v.rows()) -
               Scalar(1) / r * exp_u.asDiagonal() * (A_i * step.p_x + c_i);

    // pᵛ = e − 1/√(μ) eᵘ∘(Aᵢpˣ + cᵢ) + eᵘ⁻ᵛ
    step.p_v = DenseVector::Ones(v.rows()) -
               Scalar(1) / r * exp_u.asDiagonal() * (A_i * step.p_x + c_i) +
               exp_u_minus_v;

    return step;
  };

  // Initializes the barrier parameter for the current iterate.
  //
  // Returns true on success and false on failure.
  auto init_barrier_parameter = [&] {
    build_rhs(Scalar(1e15));
    DenseVector p_v_0 = compute_step(Scalar(1e15)).p_v;
    build_rhs(Scalar(1));
    DenseVector p_v_1 = compute_step(Scalar(1)).p_v - p_v_0;

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

        build_rhs(sqrt_μ_mid);
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

      build_rhs(Scalar(1e15));
      DenseVector p_v_0 = compute_step(Scalar(1e15)).p_v;
      build_rhs(Scalar(1));
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
    }
  };

  // Variables for determining when a step is acceptable
  constexpr Scalar α_reduction_factor(1.0 / std::numbers::sqrt2);
  constexpr Scalar α_min(1e-7);

  int full_step_rejected_counter = 0;

  // Error
  Scalar E_0 = unscaled_kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
      matrices.scaling, g, A_e, c_e, A_i, c_i, y, u, v, sqrt_μ_min);

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

  Scalar prev_p_u_infnorm = std::numeric_limits<Scalar>::infinity();
  Scalar prev_p_v_infnorm = std::numeric_limits<Scalar>::infinity();
  bool μ_initialized = false;

  // Watchdog (nonmonotone) variables. If a line search fails, accept up to this
  // many steps in a row in case the dual variable steps allow the primal steps
  // to make progress again.
  constexpr int watchdog_max = 5;
  int watchdog_count = 0;

  while (E_0 > Scalar(options.tolerance)) {
    ScopedProfiler inner_iter_profiler{inner_iter_prof};
    ScopedProfiler feasibility_check_profiler{feasibility_check_prof};

    // Check for local equality constraint infeasibility
    if (is_equality_locally_infeasible(A_e, c_e)) {
      if (options.diagnostics) {
        print_c_e_local_infeasibility_error(c_e);
      }

      return ExitStatus::LOCALLY_INFEASIBLE;
    }

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
      if (callback({iterations, x, y, u, v, g, H, A_e, A_i})) {
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
      Scalar E_sqrt_μ = kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
          g, A_e, c_e, A_i, c_i, y, u, v, sqrt_μ);
      if (E_sqrt_μ <= Scalar(10) * sqrt_μ * sqrt_μ) {
        ScopedProfiler μ_update_profiler{μ_update_prof};
        update_barrier_parameter();
      }
    } else if (prev_p_u_infnorm <= Scalar(1) && prev_p_v_infnorm <= Scalar(1)) {
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
    bool call_feasibility_restoration = false;

    // αₖᵘ = min(1, 1/|pᵘ|_∞²)
    Scalar p_u_infnorm = step.p_u.template lpNorm<Eigen::Infinity>();
    Scalar α_u = std::min(Scalar(1), Scalar(1) / (p_u_infnorm * p_u_infnorm));
    prev_p_u_infnorm = p_u_infnorm;

    // αₖᵛ = min(1, 1/|pᵛ|_∞²)
    Scalar p_v_infnorm = step.p_v.template lpNorm<Eigen::Infinity>();
    Scalar α_v = std::min(Scalar(1), Scalar(1) / (p_v_infnorm * p_v_infnorm));
    prev_p_v_infnorm = p_v_infnorm;

    const FilterEntry<Scalar> current_entry{f, v, c_e, c_i, sqrt_μ};

    // Loop until a step is accepted
    while (1) {
      trial_x = x + α * step.p_x;
      DenseVector trial_s;
      if (options.feasible_ipm && c_i.cwiseGreater(Scalar(0)).all()) {
        // If the inequality constraints are all feasible, prevent them from
        // becoming infeasible again.
        //
        //   cᵢ − √(μ)e⁻ᵘ = 0
        //   √(μ)e⁻ᵘ = cᵢ
        //   e⁻ᵘ = 1/√(μ) cᵢ
        //   −u = ln(1/√(μ) cᵢ)
        //   u = −ln(1/√(μ) cᵢ)
        trial_s = c_i;
        trial_u = -(c_i * (Scalar(1) / sqrt_μ)).array().log().matrix();
      } else {
        trial_s = sqrt_μ * (-trial_u).array().exp().matrix();
      }
      trial_y = y + α * step.p_y;
      trial_u = v + α_u * step.p_u;
      trial_v = v + α_v * step.p_v;

      trial_f = matrices.f(trial_x);
      trial_c_e = matrices.c_e(trial_x);
      trial_c_i = matrices.c_i(trial_x);

      // If f(xₖ + αpₖˣ), cₑ(xₖ + αpₖˣ), or cᵢ(xₖ + αpₖˣ) aren't finite, reduce
      // step size immediately
      if (!isfinite(trial_f) || !trial_c_e.allFinite() ||
          !trial_c_i.allFinite()) {
        // Reduce step size
        α *= α_reduction_factor;

        if (α < α_min) {
          call_feasibility_restoration = true;
          break;
        }
        continue;
      }

      // Check whether filter accepts trial iterate
      FilterEntry trial_entry{trial_f, trial_u, trial_c_e, trial_c_i, sqrt_μ};
      if (filter.try_add(current_entry, trial_entry, step.p_x, g, α)) {
        // Accept step
        watchdog_count = 0;
        break;
      }

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

        Scalar α_u_soc = α_u;
        Scalar α_v_soc = α_v;
        DenseVector c_e_soc = c_e;

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
                      matrices.scaling, g, A_e, trial_c_e, A_i, trial_c_i,
                      trial_y, trial_u, trial_v, Scalar(0)),
                  trial_f,
                  trial_c_e.template lpNorm<1>() +
                      (trial_c_i - trial_s).template lpNorm<1>(),
                  sqrt_μ * sqrt_μ, solver.hessian_regularization(),
                  solver.constraint_jacobian_regularization(),
                  soc_step.p_x.template lpNorm<Eigen::Infinity>(),
                  std::max(soc_step.p_y.template lpNorm<Eigen::Infinity>(),
                           soc_step.p_v.template lpNorm<Eigen::Infinity>()),
                  Scalar(1), Scalar(1), α_reduction_factor, α_v_soc);
            }
          }};

          // Rebuild Newton-KKT rhs with updated constraint values.
          //
          // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(−√(μ)eᵘ + 2√(μ)eᵛ − eᵘ⁺ᵛ∘cᵢ)]
          //        [                     cₑ                     ]
          //
          // where cₑˢᵒᶜ = c(xₖ) + c(xₖ + αpₖˣ)
          c_e_soc += trial_c_e;
          rhs.bottomRows(y.rows()) = -c_e_soc;

          // Solve the Newton-KKT system
          soc_step = compute_step(sqrt_μ);

          // αₖᵘ = 1/max(1, |pᵘ|_∞²)
          Scalar p_u_infnorm = step.p_u.template lpNorm<Eigen::Infinity>();
          α_u_soc = Scalar(1) / std::max(Scalar(1), p_u_infnorm * p_u_infnorm);

          // αₖᵛ = 1/max(1, |pᵛ|_∞²)
          Scalar p_v_infnorm = step.p_v.template lpNorm<Eigen::Infinity>();
          α_v_soc = Scalar(1) / std::max(Scalar(1), p_v_infnorm * p_v_infnorm);

          trial_x = x + soc_step.p_x;
          trial_y = y + soc_step.p_y;
          trial_u = u + α_u_soc * soc_step.p_u;
          trial_v = v + α_v_soc * soc_step.p_v;
          trial_s = sqrt_μ * (-trial_u).array().exp().matrix();

          trial_f = matrices.f(trial_x);
          trial_c_e = matrices.c_e(trial_x);
          trial_c_i = matrices.c_i(trial_x);

          // Check whether filter accepts trial iterate
          FilterEntry trial_entry{trial_f, trial_u, trial_c_e, trial_c_i,
                                  sqrt_μ};
          if (filter.try_add(current_entry, trial_entry, step.p_x, g, α)) {
            step = soc_step;
            α = Scalar(1);
            α_v = α_v_soc;
            step_acceptable = true;
            break;
          }

          // Constraint violation scale factor for second-order corrections
          constexpr Scalar κ_soc(0.99);

          // If constraint violation hasn't been sufficiently reduced, stop
          // making second-order corrections
          next_constraint_violation =
              trial_c_e.template lpNorm<1>() +
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
            g, A_e, c_e, A_i, c_i, y, u, v, sqrt_μ);

        trial_x = x + α_max * step.p_x;
        trial_y = y + α_max * step.p_y;
        trial_u = u + α_u * step.p_u;
        trial_v = v + α_v * step.p_v;

        trial_f = matrices.f(trial_x);
        trial_c_e = matrices.c_e(trial_x);
        trial_c_i = matrices.c_i(trial_x);

        Scalar next_kkt_error = kkt_error<Scalar, KKTErrorType::ONE_NORM>(
            matrices.g(trial_x), matrices.A_e(trial_x), trial_c_e,
            matrices.A_i(trial_x), trial_c_i, trial_y, trial_u, trial_v,
            sqrt_μ);

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

      FilterEntry initial_entry{matrices.f(x), u, c_e, c_i, sqrt_μ};

      // Feasibility restoration phase
      gch::small_vector<std::function<bool(const IterationInfo<Scalar>& info)>>
          callbacks;
      for (auto& callback : iteration_callbacks) {
        callbacks.emplace_back(callback);
      }
      callbacks.emplace_back([&](const IterationInfo<Scalar>& info) {
        DenseVector trial_x =
            info.x.segment(0, matrices.num_decision_variables);
        DenseVector trial_u =
            info.u.segment(0, matrices.num_inequality_constraints);

        DenseVector trial_c_e = matrices.c_e(trial_x);
        DenseVector trial_c_i = matrices.c_i(trial_x);

        // If the current iterate sufficiently reduces constraint violation and
        // is accepted by the normal filter, stop feasibility restoration
        FilterEntry trial_entry{matrices.f(trial_x), trial_u, trial_c_e,
                                trial_c_i, sqrt_μ};
        return trial_entry.constraint_violation <
                   Scalar(0.9) * initial_entry.constraint_violation &&
               filter.try_add(initial_entry, trial_entry, trial_x - x, g, α);
      });
      auto status =
          feasibility_restoration<Scalar>(matrices, is_nlp, callbacks, options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
                                          bound_constraint_mask,
#endif
                                          x, y, u, v, sqrt_μ, iterations);

      if (status != ExitStatus::SUCCESS) {
        // Report failure
        return status;
      }

      f = matrices.f(x);
      c_e = matrices.c_e(x);
      c_i = matrices.c_i(x);
    } else {
      // If full step was accepted, reset full-step rejected counter
      if (α == α_max) {
        full_step_rejected_counter = 0;
      }

      // Update iterates
      x = trial_x;
      y = trial_y;
      u = trial_u;
      v = trial_v;

      f = trial_f;
      c_e = trial_c_e;
      c_i = trial_c_i;
    }

    exp_u = u.array().exp().matrix();
    exp_v = v.array().exp().matrix();
    exp_u_plus_v = exp_u.cwiseProduct(exp_v);
    exp_u_minus_v = exp_v.cwiseProduct(exp_v.cwiseInverse());
    s = sqrt_μ * exp_u.cwiseInverse();

    // Update autodiff for Jacobians and Hessian
    A_e = matrices.A_e(x);
    A_i = matrices.A_i(x);
    g = matrices.g(x);
    H = matrices.H(x, y, v, sqrt_μ);

    // Update the error
    E_0 = unscaled_kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
        matrices.scaling, g, A_e, c_e, A_i, c_i, y, u, v, sqrt_μ_min);

    inner_iter_profiler.stop();

    if (options.diagnostics) {
      print_iteration_diagnostics(
          iterations,
          in_feasibility_restoration ? IterationType::FEASIBILITY_RESTORATION
                                     : IterationType::NORMAL,
          inner_iter_profiler.current_duration(), E_0, f,
          c_e.template lpNorm<1>() + (c_i - s).template lpNorm<1>(),
          sqrt_μ * sqrt_μ, solver.hessian_regularization(),
          solver.constraint_jacobian_regularization(),
          step.p_x.template lpNorm<Eigen::Infinity>(),
          std::max(step.p_y.template lpNorm<Eigen::Infinity>(),
                   step.p_v.template lpNorm<Eigen::Infinity>()),
          α, α_max, α_reduction_factor, α_v);
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
