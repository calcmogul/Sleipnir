// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/sqp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <ranges>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "optimization/regularized_ldlt.hpp"
#include "optimization/solver/util/error_estimate.hpp"
#include "optimization/solver/util/filter.hpp"
#include "optimization/solver/util/is_locally_infeasible.hpp"
#include "optimization/solver/util/kkt_error.hpp"
#include "optimization/solver/util/lagrange_multiplier_estimate.hpp"
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

namespace {

/**
 * SQP step direction.
 */
struct Step {
  /// Primal step.
  Eigen::VectorXd p_x;
  /// Dual step.
  Eigen::VectorXd p_y;
};

}  // namespace

namespace slp {

ExitStatus sqp(
    std::span<Variable> decision_variables,
    std::span<Variable> equality_constraints, Variable& f,
    std::span<std::function<bool(const IterationInfo& info)>> callbacks,
    const Options& options, Eigen::VectorXd& x) {
  const auto solve_start_time = std::chrono::steady_clock::now();

  small_vector<SetupProfiler> setup_profilers;
  setup_profilers.emplace_back("setup").start();

  VariableMatrix x_ad{decision_variables};

  VariableMatrix c_e_ad{equality_constraints};
  Eigen::VectorXd c_e = c_e_ad.value();

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
  setup_profilers.emplace_back("  ↳ y setup").start();

  // Create autodiff variables for y for Lagrangian
  Eigen::VectorXd y = lagrange_multiplier_estimate(g, A_e);
  VariableMatrix y_ad(equality_constraints.size());
  y_ad.set_value(y);

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ L setup").start();

  // Lagrangian L
  //
  // L(xₖ, yₖ) = f(xₖ) − yₖᵀcₑ(xₖ)
  auto L = f - (y_ad.T() * c_e_ad)[0];

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∇²ₓₓL setup").start();

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(xₖ, yₖ)
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

  // Check whether initial guess has finite f(xₖ) and cₑ(xₖ)
  if (!std::isfinite(f.value()) || !c_e.allFinite()) {
    return ExitStatus::NONFINITE_INITIAL_COST_OR_CONSTRAINTS;
  }

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ spy setup").start();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  // Sparsity pattern files written when spy flag is set in Config
  std::unique_ptr<Spy> H_spy;
  std::unique_ptr<Spy> A_e_spy;
  std::unique_ptr<Spy> lhs_spy;
  if (options.spy) {
    H_spy = std::make_unique<Spy>("H.spy", "Hessian", "Decision variables",
                                  "Decision variables", H.rows(), H.cols());
    A_e_spy = std::make_unique<Spy>("A_e.spy", "Equality constraint Jacobian",
                                    "Constraints", "Decision variables",
                                    A_e.rows(), A_e.cols());
    lhs_spy = std::make_unique<Spy>(
        "lhs.spy", "Newton-KKT system left-hand side", "Rows", "Columns",
        H.rows() + A_e.rows(), H.cols() + A_e.rows());
  }
#endif

  setup_profilers.back().stop();

  int iterations = 0;

  Filter filter;

  // Kept outside the loop so its storage can be reused
  small_vector<Eigen::Triplet<double>> triplets;

  RegularizedLDLT solver{decision_variables.size(),
                         equality_constraints.size()};

  // Variables for determining when a step is acceptable
  constexpr double α_reduction_factor = 0.5;
  constexpr double α_min = 1e-7;
  int acceptable_iter_counter = 0;

  int full_step_rejected_counter = 0;

  // Error estimate
  double E_0 = std::numeric_limits<double>::infinity();

  setup_profilers[0].stop();

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
  auto& soc_prof = solve_profilers[7];
  [[maybe_unused]]
  auto& spy_writes_prof = solve_profilers[8];
  auto& next_iter_prep_prof = solve_profilers[9];

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

      print_final_diagnostics(iterations, setup_profilers, solve_profilers);
    }
#endif
  }};

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

    // Check for diverging iterates
    if (x.lpNorm<Eigen::Infinity>() > 1e20 || !x.allFinite()) {
      return ExitStatus::DIVERGING_ITERATES;
    }

    feasibility_check_profiler.stop();
    ScopedProfiler user_callbacks_profiler{user_callbacks_prof};

    // Call user callbacks
    for (const auto& callback : callbacks) {
      if (callback({iterations, x, g, H, A_e, Eigen::SparseMatrix<double>{}})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    user_callbacks_profiler.stop();
    ScopedProfiler linear_system_build_profiler{linear_system_build_prof};

    // lhs = [H   Aₑᵀ]
    //       [Aₑ   0 ]
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    triplets.clear();
    triplets.reserve(H.nonZeros() + A_e.nonZeros());
    for (int col = 0; col < H.cols(); ++col) {
      // Append column of H lower triangle in top-left quadrant
      for (Eigen::SparseMatrix<double>::InnerIterator it{H, col}; it; ++it) {
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

    // rhs = −[∇f − Aₑᵀy]
    //        [   cₑ    ]
    Eigen::VectorXd rhs{x.rows() + y.rows()};
    rhs.segment(0, x.rows()) = -g + A_e.transpose() * y;
    rhs.segment(x.rows(), y.rows()) = -c_e;

    linear_system_build_profiler.stop();
    ScopedProfiler linear_system_compute_profiler{linear_system_compute_prof};

    Step step;
    constexpr double α_max = 1.0;
    double α = 1.0;

    // Solve the Newton-KKT system
    //
    // [H   Aₑᵀ][ pˣ] = −[∇f − Aₑᵀy]
    // [Aₑ   0 ][−pʸ]    [   cₑ    ]
    if (solver.compute(lhs).info() != Eigen::Success) [[unlikely]] {
      return ExitStatus::FACTORIZATION_FAILED;
    }

    linear_system_compute_profiler.stop();
    ScopedProfiler linear_system_solve_profiler{linear_system_solve_prof};

    auto compute_step = [&](Step& step) {
      // p = [ pˣ]
      //     [−pʸ]
      Eigen::VectorXd p = solver.solve(rhs);
      step.p_x = p.segment(0, x.rows());
      step.p_y = -p.segment(x.rows(), y.rows());
    };
    compute_step(step);

    linear_system_solve_profiler.stop();
    ScopedProfiler line_search_profiler{line_search_prof};

    α = α_max;

    // Loop until a step is accepted
    while (1) {
      Eigen::VectorXd trial_x = x + α * step.p_x;
      Eigen::VectorXd trial_y = y + α * step.p_y;

      x_ad.set_value(trial_x);

      Eigen::VectorXd trial_c_e = c_e_ad.value();

      // If f(xₖ + αpₖˣ) or cₑ(xₖ + αpₖˣ) aren't finite, reduce step size
      // immediately
      if (!std::isfinite(f.value()) || !trial_c_e.allFinite()) {
        // Reduce step size
        α *= α_reduction_factor;
        continue;
      }

      // Check whether filter accepts trial iterate
      if (filter.try_add(FilterEntry{f, trial_c_e}, α)) {
        // Accept step
        break;
      }

      double prev_constraint_violation = c_e.lpNorm<1>();
      double next_constraint_violation = trial_c_e.lpNorm<1>();

      // Second-order corrections
      //
      // If first trial point was rejected and constraint violation stayed the
      // same or went up, apply second-order corrections
      if (α == α_max &&
          next_constraint_violation >= prev_constraint_violation) {
        // Apply second-order corrections. See section 2.4 of [2].
        auto soc_step = step;

        double α_soc = α;
        Eigen::VectorXd c_e_soc = c_e;

        bool step_acceptable = false;
        for (int soc_iteration = 0; soc_iteration < 5 && !step_acceptable;
             ++soc_iteration) {
          ScopedProfiler soc_profiler{soc_prof};

          scope_exit soc_exit{[&] {
            soc_profiler.stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
            if (options.diagnostics) {
              double E = error_estimate(g, A_e, trial_c_e, trial_y);
              print_iteration_diagnostics(iterations,
                                          step_acceptable
                                              ? IterationType::ACCEPTED_SOC
                                              : IterationType::REJECTED_SOC,
                                          soc_profiler.current_duration(), E,
                                          f.value(), trial_c_e.lpNorm<1>(), 0.0,
                                          0.0, solver.hessian_regularization(),
                                          α_soc, 1.0, α_reduction_factor, 1.0);
            }
#endif
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

          x_ad.set_value(trial_x);

          trial_c_e = c_e_ad.value();

          // Constraint violation scale factor for second-order corrections
          constexpr double κ_soc = 0.99;

          // If constraint violation hasn't been sufficiently reduced, stop
          // making second-order corrections
          next_constraint_violation = trial_c_e.lpNorm<1>();
          if (next_constraint_violation > κ_soc * prev_constraint_violation) {
            break;
          }

          // Check whether filter accepts trial iterate
          if (filter.try_add(FilterEntry{f, trial_c_e}, α)) {
            step = soc_step;
            α = α_soc;
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
          filter.max_constraint_violation >
              filter.back().constraint_violation / 10.0) {
        filter.max_constraint_violation *= 0.1;
        filter.reset();
        continue;
      }

      // Reduce step size
      α *= α_reduction_factor;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, report line search failure.
      if (α < α_min) {
        double current_kkt_error = kkt_error(g, A_e, c_e, y);

        trial_x = x + α_max * step.p_x;
        trial_y = y + α_max * step.p_y;

        // Upate autodiff
        x_ad.set_value(trial_x);
        y_ad.set_value(trial_y);

        trial_c_e = c_e_ad.value();

        double next_kkt_error = kkt_error(
            gradient_f.value(), jacobian_c_e.value(), trial_c_e, trial_y);

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
          max_step_scaled, std::abs(step.p_x(row)) / (1.0 + std::abs(x(row))));
    }
    if (max_step_scaled < 10.0 * std::numeric_limits<double>::epsilon()) {
      α = α_max;
    }

    // xₖ₊₁ = xₖ + αₖpₖˣ
    // yₖ₊₁ = yₖ + αₖpₖʸ
    x += α * step.p_x;
    y += α * step.p_y;

    // Update autodiff for Jacobians and Hessian
    x_ad.set_value(x);
    y_ad.set_value(y);
    A_e = jacobian_c_e.value();
    g = gradient_f.value();
    H = hessian_L.value();

    ScopedProfiler next_iter_prep_profiler{next_iter_prep_prof};

    c_e = c_e_ad.value();

    // Update the error estimate
    E_0 = error_estimate(g, A_e, c_e, y);
    if (E_0 < options.acceptable_tolerance) {
      ++acceptable_iter_counter;
    } else {
      acceptable_iter_counter = 0;
    }

    next_iter_prep_profiler.stop();
    inner_iter_profiler.stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      print_iteration_diagnostics(iterations, IterationType::NORMAL,
                                  inner_iter_profiler.current_duration(), E_0,
                                  f.value(), c_e.lpNorm<1>(), 0.0, 0.0,
                                  solver.hessian_regularization(), α, α_max,
                                  α_reduction_factor, α);
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
