// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/newton.hpp"

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
#include "optimization/solver/util/kkt_error.hpp"
#include "sleipnir/autodiff/gradient.hpp"
#include "sleipnir/autodiff/hessian.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/util/scoped_profiler.hpp"
#include "sleipnir/util/setup_profiler.hpp"
#include "sleipnir/util/solve_profiler.hpp"
#include "util/scope_exit.hpp"

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
#include "sleipnir/util/spy.hpp"
#include "util/print_diagnostics.hpp"
#endif

// See docs/algorithms.md#Works_cited for citation definitions.

namespace slp {

ExitStatus newton(
    std::span<Variable> decision_variables, Variable& f,
    std::span<std::function<bool(const IterationInfo& info)>> callbacks,
    const Options& options, Eigen::VectorXd& x) {
  const auto solve_start_time = std::chrono::steady_clock::now();

  small_vector<SetupProfiler> setup_profilers;
  setup_profilers.emplace_back("setup").start();

  VariableMatrix x_ad{decision_variables};

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∇f(x) setup").start();

  // Gradient of f ∇f
  Gradient gradient_f{f, x_ad};

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∇f(x) init solve").start();

  Eigen::SparseVector<double> g = gradient_f.value();

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ L setup").start();

  // Lagrangian L
  //
  // L(xₖ, yₖ) = f(xₖ)
  auto L = f;

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

  // Check whether initial guess has finite f(xₖ)
  if (!std::isfinite(f.value())) {
    return ExitStatus::NONFINITE_INITIAL_COST_OR_CONSTRAINTS;
  }

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ spy setup").start();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  // Sparsity pattern files written when spy flag is set in Config
  std::unique_ptr<Spy> H_spy;
  std::unique_ptr<Spy> lhs_spy;
  if (options.spy) {
    H_spy = std::make_unique<Spy>("H.spy", "Hessian", "Decision variables",
                                  "Decision variables", H.rows(), H.cols());
    lhs_spy =
        std::make_unique<Spy>("lhs.spy", "Newton-KKT system left-hand side",
                              "Rows", "Columns", H.rows(), H.cols());
  }
#endif

  setup_profilers.back().stop();

  int iterations = 0;

  Filter filter;

  RegularizedLDLT solver{decision_variables.size(), 0};

  // Variables for determining when a step is acceptable
  constexpr double α_red_factor = 0.5;
  constexpr double α_min = 1e-20;
  int acceptable_iter_counter = 0;

  // Error estimate
  double E_0 = std::numeric_limits<double>::infinity();

  setup_profilers[0].stop();

  small_vector<SolveProfiler> solve_profilers;
  solve_profilers.emplace_back("solve");
  solve_profilers.emplace_back("  ↳ feasibility ✓");
  solve_profilers.emplace_back("  ↳ user callbacks");
  solve_profilers.emplace_back("  ↳ iter matrix compute");
  solve_profilers.emplace_back("  ↳ iter matrix solve");
  solve_profilers.emplace_back("  ↳ line search");
  solve_profilers.emplace_back("  ↳ spy writes");
  solve_profilers.emplace_back("  ↳ next iter prep");

  auto& inner_iter_prof = solve_profilers[0];
  auto& feasibility_check_prof = solve_profilers[1];
  auto& user_callbacks_prof = solve_profilers[2];
  auto& linear_system_compute_prof = solve_profilers[3];
  auto& linear_system_solve_prof = solve_profilers[4];
  auto& line_search_prof = solve_profilers[5];
  [[maybe_unused]]
  auto& spy_writes_prof = solve_profilers[6];
  auto& next_iter_prep_prof = solve_profilers[7];

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

      print_final_diagnostics(iterations, setup_profilers, solve_profilers);
    }
#endif
  }};

  while (E_0 > options.tolerance &&
         acceptable_iter_counter < options.max_acceptable_iterations) {
    ScopedProfiler inner_iter_profiler{inner_iter_prof};
    ScopedProfiler feasibility_check_profiler{feasibility_check_prof};

    // Check for diverging iterates
    if (x.lpNorm<Eigen::Infinity>() > 1e20 || !x.allFinite()) {
      return ExitStatus::DIVERGING_ITERATES;
    }

    feasibility_check_profiler.stop();
    ScopedProfiler user_callbacks_profiler{user_callbacks_prof};

    // Call user callbacks
    for (const auto& callback : callbacks) {
      if (callback({iterations, x, Eigen::VectorXd::Zero(0), g, H,
                    Eigen::SparseMatrix<double>{},
                    Eigen::SparseMatrix<double>{}})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    user_callbacks_profiler.stop();
    ScopedProfiler linear_system_compute_profiler{linear_system_compute_prof};

    // Solve the Newton-KKT system
    //
    // Hpₖˣ = −∇f
    solver.compute(H);

    linear_system_compute_profiler.stop();
    ScopedProfiler linear_system_solve_profiler{linear_system_solve_prof};

    Eigen::VectorXd p_x = solver.solve(-g);

    linear_system_solve_profiler.stop();
    ScopedProfiler line_search_profiler{line_search_prof};

    constexpr double α_max = 1.0;
    double α = α_max;

    // Loop until a step is accepted. If a step becomes acceptable, the loop
    // will exit early.
    while (1) {
      Eigen::VectorXd trial_x = x + α * p_x;

      x_ad.set_value(trial_x);

      // If f(xₖ + αpₖˣ) isn't finite, reduce step size immediately
      if (!std::isfinite(f.value())) {
        // Reduce step size
        α *= α_red_factor;
        continue;
      }

      // Check whether filter accepts trial iterate
      if (filter.try_add(FilterEntry{f}, α)) {
        // Accept step
        break;
      }

      // Reduce step size
      α *= α_red_factor;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, report bad line search.
      if (α < α_min) {
        double current_kkt_error = kkt_error(g);

        Eigen::VectorXd trial_x = x + α_max * p_x;

        // Upate autodiff
        x_ad.set_value(trial_x);

        double next_kkt_error = kkt_error(gradient_f.value());

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (next_kkt_error <= 0.999 * current_kkt_error) {
          α = α_max;

          // Accept step
          break;
        }

        return ExitStatus::LINE_SEARCH_FAILED;
      }
    }

    // Handle very small search directions by letting αₖ = αₖᵐᵃˣ when
    // max(|pₖˣ(i)|/(1 + |xₖ(i)|)) < 10ε_mach.
    //
    // See section 3.9 of [2].
    double max_step_scaled = 0.0;
    for (int row = 0; row < x.rows(); ++row) {
      max_step_scaled = std::max(max_step_scaled,
                                 std::abs(p_x(row)) / (1.0 + std::abs(x(row))));
    }
    if (max_step_scaled < 10.0 * std::numeric_limits<double>::epsilon()) {
      α = α_max;
    }

    line_search_profiler.stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    // Write out spy file contents if that's enabled
    if (options.spy) {
      ScopedProfiler spy_writes_profiler{spy_writes_prof};
      H_spy->add(H);
      lhs_spy->add(H);
    }
#endif

    // xₖ₊₁ = xₖ + αₖpₖˣ
    x += α * p_x;

    // Update autodiff for Hessian
    x_ad.set_value(x);
    g = gradient_f.value();
    H = hessian_L.value();

    ScopedProfiler next_iter_prep_profiler{next_iter_prep_prof};

    // Update the error estimate
    E_0 = error_estimate(g);
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
          inner_iter_profiler.current_duration(), E_0, f.value(), 0.0, 0.0, 0.0,
          solver.hessian_regularization(), α, α_max, 1.0);
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
