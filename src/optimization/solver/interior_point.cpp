// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/interior_point.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <ranges>

#include <Eigen/SparseCholesky>
#include <wpi/SmallVector.h>

#include "optimization/regularized_ldlt.hpp"
#include "optimization/solver/util/error_estimate.hpp"
#include "optimization/solver/util/filter.hpp"
#include "optimization/solver/util/fraction_to_the_boundary_rule.hpp"
#include "optimization/solver/util/is_locally_infeasible.hpp"
#include "optimization/solver/util/kkt_error.hpp"
#include "sleipnir/autodiff/gradient.hpp"
#include "sleipnir/autodiff/hessian.hpp"
#include "sleipnir/autodiff/jacobian.hpp"
#include "sleipnir/optimization/solver_exit_condition.hpp"
#include "sleipnir/util/scoped_profiler.hpp"
#include "sleipnir/util/setup_profiler.hpp"
#include "sleipnir/util/solve_profiler.hpp"
#include "util/scope_exit.hpp"

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
#include "sleipnir/util/print.hpp"
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
  /// Inequality constraint slack variable step.
  Eigen::VectorXd p_s;
  /// Inequality constraint dual step.
  Eigen::VectorXd p_z;
};

}  // namespace

namespace sleipnir {

void interior_point(
    std::span<Variable> decision_variables,
    std::span<Variable> equality_constraints,
    std::span<Variable> inequality_constraints, Variable& f,
    std::span<std::function<bool(const SolverIterationInfo& info)>> callbacks,
    const SolverConfig& config, Eigen::VectorXd& x, SolverStatus* status) {
  const auto solve_start_time = std::chrono::steady_clock::now();

  wpi::SmallVector<SetupProfiler> setup_profilers;
  setup_profilers.emplace_back("setup").start();

  setup_profilers.emplace_back("  ↳ s,y,z setup").start();

  // Map decision variables and constraints to VariableMatrices for Lagrangian
  VariableMatrix x_ad{decision_variables};
  x_ad.set_value(x);
  VariableMatrix c_e_ad{equality_constraints};
  VariableMatrix c_i_ad{inequality_constraints};

  // Create autodiff variables for s, y, and z for Lagrangian
  VariableMatrix s_ad(inequality_constraints.size());
  for (auto& s : s_ad) {
    s.set_value(1.0);
  }
  VariableMatrix y_ad(equality_constraints.size());
  for (auto& y : y_ad) {
    y.set_value(0.0);
  }
  VariableMatrix z_ad(inequality_constraints.size());
  for (auto& z : z_ad) {
    z.set_value(1.0);
  }

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ L setup").start();

  // Lagrangian L
  //
  // L(xₖ, sₖ, yₖ, zₖ) = f(xₖ) − yₖᵀcₑ(xₖ) − zₖᵀ(cᵢ(xₖ) − sₖ)
  auto L = f - (y_ad.T() * c_e_ad)[0] - (z_ad.T() * (c_i_ad - s_ad))[0];

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
  setup_profilers.emplace_back("  ↳ ∇f(x) setup").start();

  // Gradient of f ∇f
  Gradient gradient_f{f, x_ad};

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∇f(x) init solve").start();

  Eigen::SparseVector<double> g = gradient_f.value();

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∇²ₓₓL setup").start();

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(xₖ, sₖ, yₖ, zₖ)
  Hessian<Eigen::Lower> hessian_L{L, x_ad};

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∇²ₓₓL init solve").start();

  Eigen::SparseMatrix<double> H = hessian_L.value();

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ precondition ✓").start();

  Eigen::VectorXd s = s_ad.value();
  Eigen::VectorXd y = y_ad.value();
  Eigen::VectorXd z = z_ad.value();
  Eigen::VectorXd c_e = c_e_ad.value();
  Eigen::VectorXd c_i = c_i_ad.value();

  // Check for overconstrained problem
  if (equality_constraints.size() > decision_variables.size()) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (config.diagnostics) {
      sleipnir::println("The problem has too few degrees of freedom.");
      sleipnir::println(
          "Violated constraints (cₑ(x) = 0) in order of declaration:");
      for (int row = 0; row < c_e.rows(); ++row) {
        if (c_e[row] < 0.0) {
          sleipnir::println("  {}/{}: {} = 0", row + 1, c_e.rows(), c_e[row]);
        }
      }
    }
#endif

    status->exit_condition = SolverExitCondition::TOO_FEW_DOFS;
    return;
  }

  // Check whether initial guess has finite f(xₖ), cₑ(xₖ), and cᵢ(xₖ)
  if (!std::isfinite(f.value()) || !c_e.allFinite() || !c_i.allFinite()) {
    status->exit_condition =
        SolverExitCondition::NONFINITE_INITIAL_COST_OR_CONSTRAINTS;
    return;
  }

  setup_profilers.back().stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  // Sparsity pattern files written when spy flag is set in SolverConfig
  std::unique_ptr<Spy> H_spy;
  std::unique_ptr<Spy> A_e_spy;
  std::unique_ptr<Spy> A_i_spy;
  std::unique_ptr<Spy> lhs_spy;
  if (config.spy) {
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
  const double μ_min = config.tolerance / 10.0;

  // Barrier parameter μ
  double μ = 0.1;

  // Fraction-to-the-boundary rule scale factor minimum
  constexpr double τ_min = 0.99;

  // Fraction-to-the-boundary rule scale factor τ
  double τ = τ_min;

  Filter filter{f};

  // This should be run when the error estimate is below a desired threshold for
  // the current barrier parameter
  auto update_barrier_parameter_and_reset_filter = [&] {
    // Barrier parameter linear decrease power in "κ_μ μ". Range of (0, 1).
    constexpr double κ_μ = 0.2;

    // Barrier parameter superlinear decrease power in "μ^(θ_μ)". Range of (1,
    // 2).
    constexpr double θ_μ = 1.5;

    // Update the barrier parameter.
    //
    //   μⱼ₊₁ = max(εₜₒₗ/10, min(κ_μ μⱼ, μⱼ^θ_μ))
    //
    // See equation (7) of [2].
    μ = std::max(μ_min, std::min(κ_μ * μ, std::pow(μ, θ_μ)));

    // Update the fraction-to-the-boundary rule scaling factor.
    //
    //   τⱼ = max(τₘᵢₙ, 1 − μⱼ)
    //
    // See equation (8) of [2].
    τ = std::max(τ_min, 1.0 - μ);

    // Reset the filter when the barrier parameter is updated
    filter.reset();
  };

  // Kept outside the loop so its storage can be reused
  wpi::SmallVector<Eigen::Triplet<double>> triplets;

  RegularizedLDLT solver;

  // Variables for determining when a step is acceptable
  constexpr double α_red_factor = 0.5;
  constexpr double α_min = 1e-20;
  int acceptable_iter_counter = 0;

  int full_step_rejected_counter = 0;
  int step_too_small_counter = 0;

  // Error estimate
  double E_0 = std::numeric_limits<double>::infinity();

  setup_profilers[0].stop();

  wpi::SmallVector<SolveProfiler> solve_profilers;
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
  auto& soc_prof = solve_profilers[7];
  [[maybe_unused]]
  auto& spy_writes_prof = solve_profilers[8];
  auto& next_iter_prep_prof = solve_profilers[9];

  // Prints final diagnostics when the solver exits
  scope_exit exit{[&] {
    status->cost = f.value();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (config.diagnostics) {
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

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  if (config.diagnostics) {
    sleipnir::println("Error tolerance: {}\n", config.tolerance);
  }
#endif

  while (E_0 > config.tolerance &&
         acceptable_iter_counter < config.max_acceptable_iterations) {
    ScopedProfiler inner_iter_profiler{inner_iter_prof};
    ScopedProfiler feasibility_check_profiler{feasibility_check_prof};

    // Check for local equality constraint infeasibility
    if (is_equality_locally_infeasible(A_e, c_e)) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (config.diagnostics) {
        sleipnir::println(
            "The problem is locally infeasible due to violated equality "
            "constraints.");
        sleipnir::println(
            "Violated constraints (cₑ(x) = 0) in order of declaration:");
        for (int row = 0; row < c_e.rows(); ++row) {
          if (c_e[row] < 0.0) {
            sleipnir::println("  {}/{}: {} = 0", row + 1, c_e.rows(), c_e[row]);
          }
        }
      }
#endif

      status->exit_condition = SolverExitCondition::LOCALLY_INFEASIBLE;
      return;
    }

    // Check for local inequality constraint infeasibility
    if (is_inequality_locally_infeasible(A_i, c_i)) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (config.diagnostics) {
        sleipnir::println(
            "The problem is infeasible due to violated inequality "
            "constraints.");
        sleipnir::println(
            "Violated constraints (cᵢ(x) ≥ 0) in order of declaration:");
        for (int row = 0; row < c_i.rows(); ++row) {
          if (c_i[row] < 0.0) {
            sleipnir::println("  {}/{}: {} ≥ 0", row + 1, c_i.rows(), c_i[row]);
          }
        }
      }
#endif

      status->exit_condition = SolverExitCondition::LOCALLY_INFEASIBLE;
      return;
    }

    // Check for diverging iterates
    if (x.lpNorm<Eigen::Infinity>() > 1e20 || !x.allFinite() ||
        s.lpNorm<Eigen::Infinity>() > 1e20 || !s.allFinite()) {
      status->exit_condition = SolverExitCondition::DIVERGING_ITERATES;
      return;
    }

    feasibility_check_profiler.stop();
    ScopedProfiler user_callbacks_profiler{user_callbacks_prof};

    // Call user callbacks
    for (const auto& callback : callbacks) {
      if (callback({iterations, x, s, g, H, A_e, A_i})) {
        status->exit_condition = SolverExitCondition::CALLBACK_REQUESTED_STOP;
        return;
      }
    }

    user_callbacks_profiler.stop();
    ScopedProfiler linear_system_build_profiler{linear_system_build_prof};

    //     [s₁ 0 ⋯ 0 ]
    // S = [0  ⋱   ⋮ ]
    //     [⋮    ⋱ 0 ]
    //     [0  ⋯ 0 sₘ]
    //
    //     [z₁ 0 ⋯ 0 ]
    // Z = [0  ⋱   ⋮ ]
    //     [⋮    ⋱ 0 ]
    //     [0  ⋯ 0 zₘ]
    //
    // Σ = S⁻¹Z
    const Eigen::SparseMatrix<double> Σ{s.cwiseInverse().asDiagonal() *
                                        z.asDiagonal()};

    // lhs = [H + AᵢᵀΣAᵢ  Aₑᵀ]
    //       [    Aₑ       0 ]
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    const Eigen::SparseMatrix<double> top_left =
        H + (A_i.transpose() * Σ * A_i).triangularView<Eigen::Lower>();
    triplets.clear();
    triplets.reserve(top_left.nonZeros() + A_e.nonZeros());
    for (int col = 0; col < H.cols(); ++col) {
      // Append column of H + AᵢᵀΣAᵢ lower triangle in top-left quadrant
      for (Eigen::SparseMatrix<double>::InnerIterator it{top_left, col}; it;
           ++it) {
        triplets.emplace_back(it.row(), it.col(), it.value());
      }
      // Append column of Aₑ in bottom-left quadrant
      for (Eigen::SparseMatrix<double>::InnerIterator it{A_e, col}; it; ++it) {
        triplets.emplace_back(H.rows() + it.row(), it.col(), it.value());
      }
    }
    Eigen::SparseMatrix<double> lhs(
        decision_variables.size() + equality_constraints.size(),
        decision_variables.size() + equality_constraints.size());
    lhs.setFromSortedTriplets(triplets.begin(), triplets.end(),
                              [](const auto&, const auto& b) { return b; });

    // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(−Σcᵢ + μS⁻¹e + z)]
    //        [               cₑ                ]
    Eigen::VectorXd rhs{x.rows() + y.rows()};
    rhs.segment(0, x.rows()) =
        -g + A_e.transpose() * y +
        A_i.transpose() * (-Σ * c_i + μ * s.cwiseInverse() + z);
    rhs.segment(x.rows(), y.rows()) = -c_e;

    linear_system_build_profiler.stop();
    ScopedProfiler linear_system_compute_profiler{linear_system_compute_prof};

    Step step;
    double α_max = 1.0;
    double α = 1.0;
    double α_z = 1.0;

    // Solve the Newton-KKT system
    //
    // [H + AᵢᵀΣAᵢ  Aₑᵀ][ pₖˣ] = −[∇f − Aₑᵀy − Aᵢᵀ(−Σcᵢ + μS⁻¹e + z)]
    // [    Aₑ       0 ][−pₖʸ]    [               cₑ                ]
    if (solver.compute(lhs, equality_constraints.size(), μ).info() !=
        Eigen::Success) [[unlikely]] {
      status->exit_condition = SolverExitCondition::FACTORIZATION_FAILED;
      return;
    }

    linear_system_compute_profiler.stop();
    ScopedProfiler linear_system_solve_profiler{linear_system_solve_prof};

    auto compute_step = [&](Step& step) {
      // p = [ pₖˣ]
      //     [−pₖʸ]
      Eigen::VectorXd p = solver.solve(rhs);
      step.p_x = p.segment(0, x.rows());
      step.p_y = -p.segment(x.rows(), y.rows());

      // pₖˢ = cᵢ − s + Aᵢpₖˣ
      // pₖᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ
      step.p_s = c_i - s + A_i * step.p_x;
      step.p_z = -Σ * c_i + μ * s.cwiseInverse() - Σ * A_i * step.p_x;
    };
    compute_step(step);

    linear_system_solve_profiler.stop();
    ScopedProfiler line_search_profiler{line_search_prof};

    // αᵐᵃˣ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
    α_max = fraction_to_the_boundary_rule(s, step.p_s, τ);
    α = α_max;

    // αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
    α_z = fraction_to_the_boundary_rule(z, step.p_z, τ);

    // Loop until a step is accepted
    while (1) {
      Eigen::VectorXd trial_x = x + α * step.p_x;
      Eigen::VectorXd trial_y = y + α_z * step.p_y;
      Eigen::VectorXd trial_z = z + α_z * step.p_z;

      x_ad.set_value(trial_x);

      Eigen::VectorXd trial_c_e = c_e_ad.value();
      Eigen::VectorXd trial_c_i = c_i_ad.value();

      // If f(xₖ + αpₖˣ), cₑ(xₖ + αpₖˣ), or cᵢ(xₖ + αpₖˣ) aren't finite, reduce
      // step size immediately
      if (!std::isfinite(f.value()) || !trial_c_e.allFinite() ||
          !trial_c_i.allFinite()) {
        // Reduce step size
        α *= α_red_factor;
        continue;
      }

      Eigen::VectorXd trial_s;
      if (config.feasible_ipm && c_i.cwiseGreater(0.0).all()) {
        // If the inequality constraints are all feasible, prevent them from
        // becoming infeasible again.
        //
        // See equation (19.30) in [1].
        trial_s = trial_c_i;
      } else {
        trial_s = s + α * step.p_s;
      }

      // Check whether filter accepts trial iterate
      auto entry = filter.make_entry(trial_s, trial_c_e, trial_c_i, μ);
      if (filter.try_add(entry, α)) {
        // Accept step
        break;
      }

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
        double α_z_soc = α_z;
        Eigen::VectorXd c_e_soc = c_e;

        bool step_acceptable = false;
        for (int soc_iteration = 0; soc_iteration < 5 && !step_acceptable;
             ++soc_iteration) {
          ScopedProfiler soc_profiler{soc_prof};

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
          scope_exit soc_exit{[&] {
            soc_profiler.stop();

            if (config.diagnostics) {
              double E = error_estimate(g, A_e, trial_c_e, trial_y);
              print_iteration_diagnostics(
                  iterations,
                  step_acceptable ? IterationType::ACCEPTED_SOC
                                  : IterationType::REJECTED_SOC,
                  soc_profiler.current_duration(), E, f.value(),
                  trial_c_e.lpNorm<1>() + (trial_c_i - trial_s).lpNorm<1>(),
                  trial_s.dot(trial_z), μ, solver.hessian_regularization(),
                  α_soc, 1.0, α_z_soc);
            }
          }};
#endif

          // Rebuild Newton-KKT rhs with updated constraint values.
          //
          // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(−Σcᵢ + μS⁻¹e + z)]
          //        [              cₑˢᵒᶜ              ]
          //
          // where cₑˢᵒᶜ = αc(xₖ) + c(xₖ + αpₖˣ)
          c_e_soc = α_soc * c_e_soc + trial_c_e;
          rhs.bottomRows(y.rows()) = -c_e_soc;

          // Solve the Newton-KKT system
          compute_step(soc_step);

          // αˢᵒᶜ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
          α_soc = fraction_to_the_boundary_rule(s, soc_step.p_s, τ);
          trial_x = x + α_soc * soc_step.p_x;
          trial_s = s + α_soc * soc_step.p_s;

          // αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
          α_z_soc = fraction_to_the_boundary_rule(z, soc_step.p_z, τ);
          trial_y = y + α_z_soc * soc_step.p_y;
          trial_z = z + α_z_soc * soc_step.p_z;

          x_ad.set_value(trial_x);

          trial_c_e = c_e_ad.value();
          trial_c_i = c_i_ad.value();

          // Constraint violation scale factor for second-order corrections
          constexpr double κ_soc = 0.99;

          // If constraint violation hasn't been sufficiently reduced, stop
          // making second-order corrections
          next_constraint_violation =
              trial_c_e.lpNorm<1>() + (trial_c_i - trial_s).lpNorm<1>();
          if (next_constraint_violation > κ_soc * prev_constraint_violation) {
            break;
          }

          // Check whether filter accepts trial iterate
          entry = filter.make_entry(trial_s, trial_c_e, trial_c_i, μ);
          if (filter.try_add(entry, α)) {
            step = soc_step;
            α = α_soc;
            α_z = α_z_soc;
            step_acceptable = true;
          }
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
          filter.max_constraint_violation > entry.constraint_violation / 10.0) {
        filter.max_constraint_violation *= 0.1;
        filter.reset();
        continue;
      }

      // Reduce step size
      α *= α_red_factor;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, report bad line search.
      if (α < α_min) {
        double current_kkt_error = kkt_error(g, A_e, c_e, A_i, c_i, s, y, z, μ);

        trial_x = x + α_max * step.p_x;
        trial_s = s + α_max * step.p_s;

        trial_y = y + α_z * step.p_y;
        trial_z = z + α_z * step.p_z;

        // Upate autodiff
        x_ad.set_value(trial_x);
        s_ad.set_value(trial_s);
        y_ad.set_value(trial_y);
        z_ad.set_value(trial_z);

        trial_c_e = c_e_ad.value();
        trial_c_i = c_i_ad.value();

        double next_kkt_error = kkt_error(
            gradient_f.value(), jacobian_c_e.value(), trial_c_e,
            jacobian_c_i.value(), trial_c_i, trial_s, trial_y, trial_z, μ);

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (next_kkt_error <= 0.999 * current_kkt_error) {
          α = α_max;

          // Accept step
          break;
        }

        status->exit_condition = SolverExitCondition::LINE_SEARCH_FAILED;
        return;
      }
    }

    line_search_profiler.stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    // Write out spy file contents if that's enabled
    if (config.spy) {
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
      ++step_too_small_counter;
    } else {
      step_too_small_counter = 0;
    }

    // xₖ₊₁ = xₖ + αₖpₖˣ
    // sₖ₊₁ = sₖ + αₖpₖˢ
    // yₖ₊₁ = yₖ + αₖᶻpₖʸ
    // zₖ₊₁ = zₖ + αₖᶻpₖᶻ
    x += α * step.p_x;
    s += α * step.p_s;
    y += α_z * step.p_y;
    z += α_z * step.p_z;

    // A requirement for the convergence proof is that the "primal-dual barrier
    // term Hessian" Σₖ does not deviate arbitrarily much from the "primal
    // Hessian" μⱼSₖ⁻². We ensure this by resetting
    //
    //   zₖ₊₁⁽ⁱ⁾ = max(min(zₖ₊₁⁽ⁱ⁾, κ_Σ μⱼ/sₖ₊₁⁽ⁱ⁾), μⱼ/(κ_Σ sₖ₊₁⁽ⁱ⁾))
    //
    // for some fixed κ_Σ ≥ 1 after each step. See equation (16) of [2].
    for (int row = 0; row < z.rows(); ++row) {
      // Barrier parameter scale factor for inequality constraint Lagrange
      // multiplier safeguard
      constexpr double κ_Σ = 1e10;

      z[row] = std::max(std::min(z[row], κ_Σ * μ / s[row]), μ / (κ_Σ * s[row]));
    }

    // Update autodiff for Jacobians and Hessian
    x_ad.set_value(x);
    s_ad.set_value(s);
    y_ad.set_value(y);
    z_ad.set_value(z);
    A_e = jacobian_c_e.value();
    A_i = jacobian_c_i.value();
    g = gradient_f.value();
    H = hessian_L.value();

    ScopedProfiler next_iter_prep_profiler{next_iter_prep_prof};

    c_e = c_e_ad.value();
    c_i = c_i_ad.value();

    // Update the error estimate
    E_0 = error_estimate(g, A_e, c_e, A_i, c_i, s, y, z, 0.0);
    if (E_0 < config.acceptable_tolerance) {
      ++acceptable_iter_counter;
    } else {
      acceptable_iter_counter = 0;
    }

    // Update the barrier parameter if necessary
    if (E_0 > config.tolerance) {
      // Barrier parameter scale factor for tolerance checks
      constexpr double κ_ε = 10.0;

      // While the error estimate is below the desired threshold for this
      // barrier parameter value, decrease the barrier parameter further
      double E_μ = error_estimate(g, A_e, c_e, A_i, c_i, s, y, z, μ);
      while (μ > μ_min && E_μ <= κ_ε * μ) {
        update_barrier_parameter_and_reset_filter();
        E_μ = error_estimate(g, A_e, c_e, A_i, c_i, s, y, z, μ);
      }
    }

    next_iter_prep_profiler.stop();
    inner_iter_profiler.stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (config.diagnostics) {
      print_iteration_diagnostics(
          iterations, IterationType::NORMAL,
          inner_iter_profiler.current_duration(), E_0, f.value(),
          c_e.lpNorm<1>() + (c_i - s).lpNorm<1>(), s.dot(z), μ,
          solver.hessian_regularization(), α, α_max, α_z);
    }
#endif

    ++iterations;

    // Check for max iterations
    if (iterations >= config.max_iterations) {
      status->exit_condition = SolverExitCondition::MAX_ITERATIONS_EXCEEDED;
      return;
    }

    // Check for max wall clock time
    if (std::chrono::steady_clock::now() - solve_start_time > config.timeout) {
      status->exit_condition = SolverExitCondition::TIMEOUT;
      return;
    }

    // Check for solve to acceptable tolerance
    if (E_0 > config.tolerance &&
        acceptable_iter_counter == config.max_acceptable_iterations) {
      status->exit_condition =
          SolverExitCondition::SOLVED_TO_ACCEPTABLE_TOLERANCE;
      return;
    }

    // The search direction has been very small twice, so assume the problem has
    // been solved as well as possible given finite precision and reduce the
    // barrier parameter.
    //
    // See section 3.9 of [2].
    if (step_too_small_counter >= 2 && μ > μ_min) {
      update_barrier_parameter_and_reset_filter();
      continue;
    }
  }
}

}  // namespace sleipnir
