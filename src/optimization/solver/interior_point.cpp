// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/interior_point.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <ranges>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "optimization/regularized_ldlt.hpp"
#include "optimization/solver/util/error_estimate.hpp"
#include "optimization/solver/util/filter.hpp"
#include "optimization/solver/util/is_locally_infeasible.hpp"
#include "optimization/solver/util/kkt_error.hpp"
#include "sleipnir/autodiff/gradient.hpp"
#include "sleipnir/autodiff/hessian.hpp"
#include "sleipnir/autodiff/jacobian.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/util/scoped_profiler.hpp"
#include "sleipnir/util/setup_profiler.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "sleipnir/util/solve_profiler.hpp"
#include "util/scope_exit.hpp"

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
#include "sleipnir/util/spy.hpp"
#include "util/print_diagnostics.hpp"
#endif

// See docs/algorithms.md#Works_cited for citation definitions.
//
// See docs/algorithms.md#Interior-point_method for a derivation of the
// interior-point method formulation being used.

namespace {

/**
 * Interior-point method step direction.
 */
struct Step {
  /// Primal step.
  Eigen::VectorXd p_x;
  /// Equality constraint dual step.
  Eigen::VectorXd p_y;
  /// Log-domain variable step.
  Eigen::VectorXd p_v;
};

}  // namespace

namespace slp {

ExitStatus interior_point(
    std::span<Variable> decision_variables,
    std::span<Variable> equality_constraints,
    std::span<Variable> inequality_constraints, Variable& f,
    std::span<std::function<bool(const IterationInfo& info)>> callbacks,
    const Options& options, Eigen::VectorXd& x) {
  const auto solve_start_time = std::chrono::steady_clock::now();

  small_vector<SetupProfiler> setup_profilers;
  setup_profilers.emplace_back("setup").start();

  VariableMatrix x_ad{decision_variables};

  VariableMatrix c_e_ad{equality_constraints};
  Eigen::VectorXd c_e = c_e_ad.value();

  VariableMatrix c_i_ad{inequality_constraints};
  Eigen::VectorXd c_i = c_i_ad.value();

  setup_profilers.emplace_back("  ↳ ∇f(x) setup").start();

  // Gradient of f ∇f
  Gradient gradient_f{f, x_ad};

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∇f(x) init solve").start();

  Eigen::SparseVector<double> g = gradient_f.value();

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∂cₑ/∂x setup").start();

  // Equality constraint Jacobian Aₑ
  //
  //         [∇ᵀcₑ₁(xₖ)]
  // Aₑ(x) = [∇ᵀcₑ₂(xₖ)]
  //         [    ⋮    ]
  //         [∇ᵀcₑₘ(xₖ)]
  Jacobian jacobian_c_e{c_e_ad, x_ad};

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∂cₑ/∂x init solve").start();

  Eigen::SparseMatrix<double> A_e = jacobian_c_e.value();

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∂cᵢ/∂x setup").start();

  // Inequality constraint Jacobian Aᵢ
  //
  //         [∇ᵀcᵢ₁(xₖ)]
  // Aᵢ(x) = [∇ᵀcᵢ₂(xₖ)]
  //         [    ⋮    ]
  //         [∇ᵀcᵢₘ(xₖ)]
  Jacobian jacobian_c_i{c_i_ad, x_ad};

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∂cᵢ/∂x init solve").start();

  Eigen::SparseMatrix<double> A_i = jacobian_c_i.value();

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ y,v setup").start();

  // Create autodiff variables for y for Lagrangian
  Eigen::VectorXd y = Eigen::VectorXd::Zero(equality_constraints.size());
  VariableMatrix y_ad(equality_constraints.size());
  y_ad.set_value(y);

  // Create autodiff variables for v for Lagrangian
  Eigen::VectorXd v = Eigen::VectorXd::Zero(inequality_constraints.size());
  VariableMatrix v_ad(inequality_constraints.size());
  v_ad.set_value(v);

  // Barrier parameter μ
  double sqrt_μ = 0.1;
  Variable sqrt_μ_ad;
  sqrt_μ_ad.set_value(sqrt_μ);

  // eᵛ
  Eigen::VectorXd exp_v{v.array().exp().matrix()};
  // e⁻ᵛ
  Eigen::VectorXd exp_neg_v{(-v).array().exp().matrix()};
  // e⁻²ᵛ
  Eigen::VectorXd exp_neg_2v{(-2 * v).array().exp().matrix()};
  // s = √(μ)eᵛ
  Eigen::VectorXd s = sqrt_μ * exp_v;

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ L setup").start();

  // Lagrangian L
  //
  //   L(x, s, y, z) = f(x) − yᵀcₑ(x) − zᵀ(cᵢ(x) − s)
  //
  // We omit the barrier term because it won't be in ∇ₓₓL.
  //
  //   L(x, y, v) = f(x) − yᵀcₑ(x) − √(μ)e⁻ᵛᵀ(cᵢ(x) − √(μ)eᵛ)
  //   L(x, y, v) = f(x) − yᵀcₑ(x) − (√(μ)e⁻ᵛᵀcᵢ(x) − μ)
  //   L(x, y, v) = f(x) − yᵀcₑ(x) − √(μ)e⁻ᵛᵀcᵢ(x) + μ
  //
  // Omit constants since they won't be in ∇ₓₓL.
  //
  //   L(x, y, v) = f(x) − yᵀcₑ(x) − √(μ)e⁻ᵛᵀcᵢ(x)
  auto L = f - (y_ad.T() * c_e_ad)[0] -
           sqrt_μ_ad * ((-v_ad).cwise_transform(&slp::exp).T() * c_i_ad)[0];

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∇²ₓₓL setup").start();

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(xₖ, yₖ, vₖ)
  Hessian<Eigen::Lower> hessian_L{L, x_ad};

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∇²ₓₓL init solve").start();

  Eigen::SparseMatrix<double> H = hessian_L.value();

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ precondition ✓").start();

  // Check for overconstrained problem
  if (equality_constraints.size() > decision_variables.size()) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      print_too_many_dofs_error(c_e);
    }
#endif

    return ExitStatus::TOO_FEW_DOFS;
  }

  // Check whether initial guess has finite f(xₖ), cₑ(xₖ), and cᵢ(xₖ)
  if (!std::isfinite(f.value()) || !c_e.allFinite() || !c_i.allFinite()) {
    return ExitStatus::NONFINITE_INITIAL_COST_OR_CONSTRAINTS;
  }

  setup_profilers.back().stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  // Sparsity pattern files written when spy flag is set in Config
  std::unique_ptr<Spy> H_spy;
  std::unique_ptr<Spy> A_e_spy;
  std::unique_ptr<Spy> A_i_spy;
  std::unique_ptr<Spy> lhs_spy;
  if (options.spy) {
    H_spy = std::make_unique<Spy>("H.spy", "Hessian", "Decision variables",
                                  "Decision variables", H.rows(), H.cols());
    A_e_spy = std::make_unique<Spy>("A_e.spy", "Equality constraint Jacobian",
                                    "Constraints", "Decision variables",
                                    A_e.rows(), A_e.cols());
    A_i_spy = std::make_unique<Spy>("A_i.spy", "Inequality constraint Jacobian",
                                    "Constraints", "Decision variables",
                                    A_i.rows(), A_i.cols());
    lhs_spy = std::make_unique<Spy>(
        "lhs.spy", "Newton-KKT system left-hand side", "Rows", "Columns",
        H.rows() + A_e.rows(), H.cols() + A_e.rows());
  }
#endif

  int iterations = 0;

  // Barrier parameter minimum
  const double sqrt_μ_min = std::sqrt(options.tolerance / 10.0);

  Filter filter;

  // Kept outside the loop so its storage can be reused
  small_vector<Eigen::Triplet<double>> triplets;

  RegularizedLDLT solver{decision_variables.size(),
                         equality_constraints.size()};
  Eigen::SparseMatrix<double> lhs(
      decision_variables.size() + equality_constraints.size(),
      decision_variables.size() + equality_constraints.size());
  Eigen::VectorXd rhs{x.rows() + y.rows()};

  small_vector<SolveProfiler> solve_profilers;
  solve_profilers.emplace_back("solve");
  solve_profilers.emplace_back("  ↳ feasibility ✓");
  solve_profilers.emplace_back("  ↳ user callbacks");
  solve_profilers.emplace_back("  ↳ iter matrix build");
  solve_profilers.emplace_back("  ↳ iter matrix compute");
  solve_profilers.emplace_back("  ↳ iter matrix solve");
  solve_profilers.emplace_back("  ↳ line search");
  solve_profilers.emplace_back("    ↳ SOC");
  solve_profilers.emplace_back("  ↳ spy writes");
  solve_profilers.emplace_back("  ↳ next iter prep");

  auto& inner_iter_prof = solve_profilers[0];
  auto& feasibility_check_prof = solve_profilers[1];
  auto& user_callbacks_prof = solve_profilers[2];
  auto& linear_system_build_prof = solve_profilers[3];
  auto& linear_system_compute_prof = solve_profilers[4];
  auto& linear_system_solve_prof = solve_profilers[5];
  auto& line_search_prof = solve_profilers[6];
  [[maybe_unused]]
  auto& soc_prof = solve_profilers[7];
  [[maybe_unused]]
  auto& spy_writes_prof = solve_profilers[8];
  auto& next_iter_prep_prof = solve_profilers[9];

  const bool is_nlp = [&] -> bool {
    if (f.type() > ExpressionType::QUADRATIC) {
      return true;
    } else if (!equality_constraints.empty() &&
               std::ranges::max(equality_constraints, {}, &Variable::type)
                       .type() > ExpressionType::LINEAR) {
      return true;
    } else if (!inequality_constraints.empty() &&
               std::ranges::max(inequality_constraints, {}, &Variable::type)
                       .type() > ExpressionType::LINEAR) {
      return true;
    }

    return false;
  }();

  // r is sqrt_μ
  auto build_and_compute_lhs = [&] -> ExitStatus {
    ScopedProfiler linear_system_build_profiler{linear_system_build_prof};

    // lhs = [H + Aᵢᵀdiag(e⁻²ᵛ)Aᵢ  Aₑᵀ]
    //       [         Aₑ           0 ]
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    const Eigen::SparseMatrix<double> top_left =
        H + (A_i.transpose() * exp_neg_2v.asDiagonal() * A_i)
                .triangularView<Eigen::Lower>();
    triplets.clear();
    triplets.reserve(top_left.nonZeros() + A_e.nonZeros());
    for (int col = 0; col < H.cols(); ++col) {
      // Append column of H + Aᵢᵀdiag(e⁻²ᵛ)Aᵢ lower triangle in top-left
      // quadrant
      for (Eigen::SparseMatrix<double>::InnerIterator it{top_left, col}; it;
           ++it) {
        triplets.emplace_back(it.row(), it.col(), it.value());
      }
      // Append column of Aₑ in bottom-left quadrant
      for (Eigen::SparseMatrix<double>::InnerIterator it{A_e, col}; it; ++it) {
        triplets.emplace_back(H.rows() + it.row(), it.col(), it.value());
      }
    }
    lhs.setFromSortedTriplets(triplets.begin(), triplets.end(),
                              [](const auto&, const auto& b) { return b; });

    linear_system_build_profiler.stop();
    ScopedProfiler linear_system_compute_profiler{linear_system_compute_prof};

    // Solve the Newton-KKT system
    //
    // [H + Aᵢᵀdiag(e⁻²ᵛ)Aᵢ  Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(−e⁻²ᵛ∘cᵢ)]
    // [         Aₑ           0 ][−pʸ]    [           cₑ            ]
    if (solver.compute(lhs).info() != Eigen::Success) [[unlikely]] {
      return ExitStatus::FACTORIZATION_FAILED;
    }

    return ExitStatus::SUCCESS;
  };

  auto build_rhs = [&] {
    // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(−e⁻²ᵛ∘cᵢ)]
    //        [           cₑ            ]
    rhs.segment(0, x.rows()) =
        -g + A_e.transpose() * y -
        A_i.transpose() * (exp_neg_2v.asDiagonal() * c_i);
    rhs.segment(x.rows(), y.rows()) = -c_e;
  };

  // r is sqrt_μ
  auto compute_step = [&](double r) -> Step {
    Step step;

    // p = [ pˣ]
    //     [−pʸ]
    Eigen::VectorXd p = solver.solve(rhs);
    step.p_x = p.segment(0, x.rows());
    step.p_y = -p.segment(x.rows(), y.rows());

    // pᵛ = 1/√(μ) e⁻ᵛ∘(Aᵢpˣ + cᵢ) − e
    step.p_v = 1.0 / r * exp_neg_v.asDiagonal() * (A_i * step.p_x + c_i) -
               Eigen::VectorXd::Ones(v.rows());

    return step;
  };

  // Updates the barrier parameter for the current iterate and resets the
  // filter.
  //
  // This should be run when the error estimate is below a desired threshold for
  // the current barrier parameter.
  //
  // Returns true on success and false on failure.
  auto update_barrier_parameter = [&] {
    if (sqrt_μ == sqrt_μ_min) {
      return;
    }

    // Barrier parameter linear decrease power in "κ_μ μ". Range of (0, 1).
    constexpr double κ_μ = 0.2;

    constexpr double dinf_bound = 0.99;

    if (is_nlp) {
      // Barrier parameter linear decrease power in "κ_μ μ". Range of (0, 1).
      constexpr double κ_μ = 0.2;

      // Barrier parameter superlinear decrease power in "μ^(θ_μ)".
      // Range of (1, 2).
      constexpr double θ_μ = 1.5;

      // Update the barrier parameter.
      //
      //   μⱼ₊₁ = max(εₜₒₗ/10, min(κ_μ μⱼ, μⱼ^θ_μ))
      //
      // See equation (7) of [2].
      sqrt_μ = std::max(
          sqrt_μ_min, std::min(std::sqrt(κ_μ) * sqrt_μ, std::pow(sqrt_μ, θ_μ)));
    } else {
      build_rhs();
      Eigen::VectorXd p_v_0 = compute_step(1e15).p_v;
      Eigen::VectorXd p_v_1 = compute_step(1.0).p_v - p_v_0;

      double α_min = 0.0;
      double α_max = 1e15;

      for (int i = 0; i < v.rows(); ++i) {
        double temp_min = (dinf_bound - p_v_0[i]) / p_v_1[i];
        double temp_max = (-dinf_bound - p_v_0[i]) / p_v_1[i];
        if (p_v_1[i] > 0.0) {
          std::swap(temp_min, temp_max);
        }

        α_min = std::max(α_min, temp_min);
        α_max = std::min(α_max, temp_max);
      }

      if (α_min <= α_max) {
        sqrt_μ = std::max(sqrt_μ_min, 1.0 / α_max);
      } else {
        // Line search failed, so use shortstep method for reducing μ instead
        sqrt_μ = std::max(sqrt_μ_min, std::sqrt(κ_μ) * sqrt_μ);
      }
    }

    // Reset the filter when the barrier parameter is updated
    filter.reset();
  };

  // Variables for determining when a step is acceptable
  constexpr double α_red_factor = 0.5;
  constexpr double α_min = 1e-7;
  int acceptable_iter_counter = 0;

  int full_step_rejected_counter = 0;

  // Error estimate
  double E_0 = std::numeric_limits<double>::infinity();

  setup_profilers[0].stop();

  // Prints final diagnostics when the solver exits
  scope_exit exit{[&] {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      // Append gradient profilers
      solve_profilers.push_back(gradient_f.get_profilers()[0]);
      solve_profilers.back().name = "  ↳ ∇f(x)";
      for (const auto& profiler :
           gradient_f.get_profilers() | std::views::drop(1)) {
        solve_profilers.push_back(profiler);
      }

      // Append Hessian profilers
      solve_profilers.push_back(hessian_L.get_profilers()[0]);
      solve_profilers.back().name = "  ↳ ∇²ₓₓL";
      for (const auto& profiler :
           hessian_L.get_profilers() | std::views::drop(1)) {
        solve_profilers.push_back(profiler);
      }

      // Append equality constraint Jacobian profilers
      solve_profilers.push_back(jacobian_c_e.get_profilers()[0]);
      solve_profilers.back().name = "  ↳ ∂cₑ/∂x";
      for (const auto& profiler :
           jacobian_c_e.get_profilers() | std::views::drop(1)) {
        solve_profilers.push_back(profiler);
      }

      // Append inequality constraint Jacobian profilers
      solve_profilers.push_back(jacobian_c_i.get_profilers()[0]);
      solve_profilers.back().name = "  ↳ ∂cᵢ/∂x";
      for (const auto& profiler :
           jacobian_c_i.get_profilers() | std::views::drop(1)) {
        solve_profilers.push_back(profiler);
      }

      print_final_diagnostics(iterations, setup_profilers, solve_profilers);
    }
#endif
  }};

  double prev_p_v_infnorm = std::numeric_limits<double>::infinity();
  bool μ_initialized = false;

  while (E_0 > options.tolerance &&
         acceptable_iter_counter < options.max_acceptable_iterations) {
    ScopedProfiler inner_iter_profiler{inner_iter_prof};
    ScopedProfiler feasibility_check_profiler{feasibility_check_prof};

    // Check for local equality constraint infeasibility
    if (is_equality_locally_infeasible(A_e, c_e)) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (options.diagnostics) {
        print_c_e_local_infeasibility_error(c_e);
      }
#endif

      return ExitStatus::LOCALLY_INFEASIBLE;
    }

    // Check for local inequality constraint infeasibility
    if (is_inequality_locally_infeasible(A_i, c_i)) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (options.diagnostics) {
        print_c_i_local_infeasibility_error(c_i);
      }
#endif

      return ExitStatus::LOCALLY_INFEASIBLE;
    }

    // Check for diverging iterates
    if (x.lpNorm<Eigen::Infinity>() > 1e20 || !x.allFinite() ||
        v.lpNorm<Eigen::Infinity>() > 1e20 || !v.allFinite()) {
      return ExitStatus::DIVERGING_ITERATES;
    }

    feasibility_check_profiler.stop();
    ScopedProfiler user_callbacks_profiler{user_callbacks_prof};

    // Call user callbacks
    for (const auto& callback : callbacks) {
      if (callback({iterations, x, g, H, A_e, A_i})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    user_callbacks_profiler.stop();

    if (auto status = build_and_compute_lhs(); status != ExitStatus::SUCCESS) {
      return status;
    }

    // Update the barrier parameter if necessary
    if (!μ_initialized) {
      update_barrier_parameter();
      μ_initialized = true;
    } else if (is_nlp) {
      double E_sqrt_μ = error_estimate(g, A_e, c_e, A_i, c_i, y, v, sqrt_μ);
      if (E_sqrt_μ <= 10.0 * sqrt_μ * sqrt_μ) {
        update_barrier_parameter();
      }
    } else if (prev_p_v_infnorm <= 1.0) {
      update_barrier_parameter();
    }

    build_rhs();

    ScopedProfiler linear_system_solve_profiler{linear_system_solve_prof};

    // Solve the Newton-KKT system for the step
    Step step = compute_step(sqrt_μ);

    linear_system_solve_profiler.stop();
    ScopedProfiler line_search_profiler{line_search_prof};

    double α_max = 1.0;
    double α = 1.0;
    double α_v = 1.0;

    // αₖᵛ = 1/max(1, |pᵛ|_∞²)
    double p_v_infnorm = step.p_v.lpNorm<Eigen::Infinity>();
    α_v = 1.0 / std::max(1.0, p_v_infnorm * p_v_infnorm);
    prev_p_v_infnorm = p_v_infnorm;

    // Loop until a step is accepted
    while (1) {
      Eigen::VectorXd trial_x = x + α * step.p_x;
      Eigen::VectorXd trial_y = y + α * step.p_y;
      Eigen::VectorXd trial_v = v + α_v * step.p_v;

      x_ad.set_value(trial_x);
      y_ad.set_value(trial_y);
      v_ad.set_value(trial_v);

      Eigen::VectorXd trial_c_e = c_e_ad.value();
      Eigen::VectorXd trial_c_i = c_i_ad.value();

      // If f(xₖ + αpₖˣ), cₑ(xₖ + αpₖˣ), or cᵢ(xₖ + αpₖˣ) aren't finite, reduce
      // step size immediately
      if (!std::isfinite(f.value()) || !trial_c_e.allFinite() ||
          !trial_c_i.allFinite()) {
        // Reduce step size
        α *= α_red_factor;

        if (α < α_min) {
          return ExitStatus::LINE_SEARCH_FAILED;
        }
        continue;
      }

      Eigen::VectorXd trial_s;
      if (options.feasible_ipm && c_i.cwiseGreater(0.0).all()) {
        // If the inequality constraints are all feasible, prevent them from
        // becoming infeasible again.
        //
        //   cᵢ − √(μ)eᵛ = 0
        //   √(μ)eᵛ = cᵢ
        //   eᵛ = 1/√(μ) cᵢ
        //   v = ln(1/√(μ) cᵢ)
        trial_s = c_i;
        trial_v = (c_i * (1.0 / sqrt_μ)).array().log().matrix();
      } else {
        trial_s = sqrt_μ * trial_v.array().exp().matrix();
      }

      // Check whether filter accepts trial iterate
      if (filter.try_add(FilterEntry{f, trial_v, trial_c_e, trial_c_i, sqrt_μ},
                         α)) {
        // Accept step
        break;
      }

#if 0
      double prev_constraint_violation =
          c_e.lpNorm<1>() + (c_i - s).lpNorm<1>();
      double next_constraint_violation =
          trial_c_e.lpNorm<1>() + (trial_c_i - trial_s).lpNorm<1>();

      // Second-order corrections
      //
      // If first trial point was rejected and constraint violation stayed the
      // same or went up, apply second-order corrections
      if (α == α_max &&
          next_constraint_violation >= prev_constraint_violation) {
        // Apply second-order corrections. See section 2.4 of [2].
        auto soc_step = step;

        double α_soc = α;
        double α_v_soc = α_v;
        Eigen::VectorXd c_e_soc = c_e;

        bool step_acceptable = false;
        for (int soc_iteration = 0; soc_iteration < 5 && !step_acceptable;
             ++soc_iteration) {
          ScopedProfiler soc_profiler{soc_prof};

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
          scope_exit soc_exit{[&] {
            soc_profiler.stop();

            if (options.diagnostics) {
              double E = error_estimate(g, A_e, trial_c_e, trial_y);
              print_iteration_diagnostics(
                  iterations,
                  step_acceptable ? IterationType::ACCEPTED_SOC
                                  : IterationType::REJECTED_SOC,
                  soc_profiler.current_duration(), E, f.value(),
                  trial_c_e.lpNorm<1>() +
                      (trial_c_i - sqrt_μ * (-trial_v).array().exp().matrix())
                          .lpNorm<1>(),
                  sqrt_μ * sqrt_μ, solver.hessian_regularization(), α_soc, 1.0,
                  α_v_soc);
            }
          }};
#endif

          // Rebuild Newton-KKT rhs with updated constraint values.
          //
          // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(−e⁻²ᵛ∘cᵢ)]
          //        [          cₑˢᵒᶜ          ]
          //
          // where cₑˢᵒᶜ = αc(xₖ) + c(xₖ + αpₖˣ)
          c_e_soc = α_soc * c_e_soc + trial_c_e;
          rhs.bottomRows(y.rows()) = -c_e_soc;

          // Solve the Newton-KKT system
          soc_step = compute_step(sqrt_μ);

          α_soc = 1.0;
          trial_x = x + α_soc * soc_step.p_x;
          trial_y = y + α_soc * soc_step.p_y;

          // αₖᵛ = 1/max(1, |pᵛ|_∞²)
          double p_v_infnorm = step.p_v.lpNorm<Eigen::Infinity>();
          α_v_soc = 1.0 / std::max(1.0, p_v_infnorm * p_v_infnorm);

          trial_v = v + α_v_soc * soc_step.p_v;

          x_ad.set_value(trial_x);
          y_ad.set_value(trial_y);
          v_ad.set_value(trial_v);

          trial_c_e = c_e_ad.value();
          trial_c_i = c_i_ad.value();

          // Constraint violation scale factor for second-order corrections
          constexpr double κ_soc = 0.99;

          // If constraint violation hasn't been sufficiently reduced, stop
          // making second-order corrections
          next_constraint_violation =
              trial_c_e.lpNorm<1>() +
              (trial_c_i - sqrt_μ * exp_neg_v).lpNorm<1>();
          if (next_constraint_violation > κ_soc * prev_constraint_violation) {
            break;
          }

          // Check whether filter accepts trial iterate
          if (filter.try_add(
                  FilterEntry{f, trial_v, trial_c_e, trial_c_i, sqrt_μ}, α)) {
            step = soc_step;
            α = α_soc;
            α_v = α_v_soc;
            step_acceptable = true;
          }
        }

        if (step_acceptable) {
          // Accept step
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
              filter.back().constraint_violation / 10.0) {
        filter.max_constraint_violation *= 0.1;
        filter.reset();
        continue;
      }

      // Reduce step size
      α *= α_red_factor;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, report line search failure.
      if (α < α_min) {
        double current_kkt_error =
            kkt_error(g, A_e, c_e, A_i, c_i, y, v, sqrt_μ);

        trial_x = x + α_max * step.p_x;
        trial_y = y + α_max * step.p_y;
        trial_v = v + α_v * step.p_v;

        // Upate autodiff
        x_ad.set_value(trial_x);
        y_ad.set_value(trial_y);
        v_ad.set_value(trial_v);

        trial_c_e = c_e_ad.value();
        trial_c_i = c_i_ad.value();

        double next_kkt_error = kkt_error(
            gradient_f.value(), jacobian_c_e.value(), trial_c_e,
            jacobian_c_i.value(), trial_c_i, trial_y, trial_v, sqrt_μ);

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (next_kkt_error <= 0.999 * current_kkt_error) {
          α = α_max;

          // Accept step
          break;
        }

        return ExitStatus::LINE_SEARCH_FAILED;
      }
    }

    line_search_profiler.stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    // Write out spy file contents if that's enabled
    if (options.spy) {
      ScopedProfiler spy_writes_profiler{spy_writes_prof};
      H_spy->add(H);
      A_e_spy->add(A_e);
      A_i_spy->add(A_i);
      lhs_spy->add(lhs);
    }
#endif

    // If full step was accepted, reset full-step rejected counter
    if (α == α_max) {
      full_step_rejected_counter = 0;
    }

    // Handle very small search directions by letting αₖ = αₖᵐᵃˣ when
    // max(|pₖˣ(i)|/(1 + |xₖ(i)|)) < 10ε_mach.
    //
    // See section 3.9 of [2].
    double max_step_scaled = 0.0;
    for (int row = 0; row < x.rows(); ++row) {
      max_step_scaled = std::max(
          max_step_scaled, std::abs(step.p_x[row]) / (1.0 + std::abs(x[row])));
    }
    if (max_step_scaled < 10.0 * std::numeric_limits<double>::epsilon()) {
      α = α_max;
    }

    // xₖ₊₁ = xₖ + αₖpₖˣ
    // yₖ₊₁ = yₖ + αₖpₖʸ
    // vₖ₊₁ = vₖ + αₖᵛpₖᵛ
    x += α * step.p_x;
    y += α * step.p_y;
    v += α_v * step.p_v;

    exp_v = v.array().exp().matrix();
    exp_neg_v = (-v).array().exp().matrix();
    exp_neg_2v = (-2 * v).array().exp().matrix();
    s = sqrt_μ * exp_neg_v;

    // Update autodiff for Jacobians and Hessian
    x_ad.set_value(x);
    y_ad.set_value(y);
    v_ad.set_value(v);
    sqrt_μ_ad.set_value(sqrt_μ);
    A_e = jacobian_c_e.value();
    A_i = jacobian_c_i.value();
    g = gradient_f.value();
    H = hessian_L.value();

    ScopedProfiler next_iter_prep_profiler{next_iter_prep_prof};

    c_e = c_e_ad.value();
    c_i = c_i_ad.value();

    // Update the error estimate
    E_0 = error_estimate(g, A_e, c_e, A_i, c_i, y, v, sqrt_μ_min);
    if (E_0 < options.acceptable_tolerance) {
      ++acceptable_iter_counter;
    } else {
      acceptable_iter_counter = 0;
    }

    next_iter_prep_profiler.stop();
    inner_iter_profiler.stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      print_iteration_diagnostics(
          iterations, IterationType::NORMAL,
          inner_iter_profiler.current_duration(), E_0, f.value(),
          c_e.lpNorm<1>() + (c_i - sqrt_μ * exp_neg_v).lpNorm<1>(),
          sqrt_μ * sqrt_μ, solver.hessian_regularization(), α, α_max, α_v);
    }
#endif

    ++iterations;

    // Check for max iterations
    if (iterations >= options.max_iterations) {
      return ExitStatus::MAX_ITERATIONS_EXCEEDED;
    }

    // Check for max wall clock time
    if (std::chrono::steady_clock::now() - solve_start_time > options.timeout) {
      return ExitStatus::TIMEOUT;
    }

    // Check for solve to acceptable tolerance
    if (E_0 > options.tolerance &&
        acceptable_iter_counter == options.max_acceptable_iterations) {
      return ExitStatus::SOLVED_TO_ACCEPTABLE_TOLERANCE;
    }
  }

  return ExitStatus::SUCCESS;
}

}  // namespace slp
