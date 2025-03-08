// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/augmented_lagrangian.hpp"

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
#include "sleipnir/autodiff/expression.hpp"
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
 * Interior-point method step direction.
 */
struct Step {
  /// Primal step.
  Eigen::VectorXd p_x;
};

}  // namespace

namespace slp {

ExitStatus augmented_lagrangian(
    std::span<Variable> decision_variables,
    std::span<Variable> equality_constraints,
    std::span<Variable> inequality_constraints, Variable& f,
    std::span<std::function<bool(const IterationInfo& info)>> callbacks,
    const Options& options, Eigen::VectorXd& x) {
  Eigen::VectorXd x_0 = x;

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
  setup_profilers.emplace_back("  ↳ y,z setup").start();

  // Create autodiff variables for y for Lagrangian
  Eigen::VectorXd y = Eigen::VectorXd::Zero(equality_constraints.size());
  VariableMatrix y_ad(equality_constraints.size());
  y_ad.set_value(y);

  // Create autodiff variables for z for Lagrangian
  Eigen::VectorXd z = Eigen::VectorXd::Zero(inequality_constraints.size());
  VariableMatrix z_ad(inequality_constraints.size());
  z_ad.set_value(z);

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ L setup").start();

  // Penalty parameter
  constexpr double ρ_max = 1e10;
  Variable ρ;
  ρ.set_value(1.0);

  // Inequality constraint active set
  Eigen::VectorXd a(inequality_constraints.size());
  for (size_t row = 0; row < inequality_constraints.size(); ++row) {
    a(row) = c_i[row] <= 0.0 && z[row] == 0.0;
  }
  VariableMatrix diag_a_ad(inequality_constraints.size(),
                           inequality_constraints.size());
  for (size_t row = 0; row < inequality_constraints.size(); ++row) {
    diag_a_ad(row, row) = Variable{
        detail::make_expression_ptr<detail::DecisionVariableExpression>(
            a(row))};
  }

  // Lagrangian L
  //
  //   L(xₖ, yₖ, zₖ) = f(xₖ) − yₖᵀcₑ(xₖ) − zₖᵀcᵢ(xₖ)
  //     + 1/2ρcₑᵀcₑ + 1/2ρcᵢᵀdiag(a)cᵢ
  //
  // where diag(a) = diag(if cᵢ[i] > 0 and z[i] = 0 { 0 } else { 1 }) denotes
  // the inequality constraint active set.
  auto L = f - (y_ad.T() * c_e_ad)[0] - (z_ad.T() * c_i_ad)[0] +
           0.5 * ρ * c_e_ad.T() * c_e_ad +
           0.5 * ρ * c_i_ad.T() * diag_a_ad * c_i_ad;

  setup_profilers.back().stop();
  setup_profilers.emplace_back("  ↳ ∇²ₓₓL setup").start();

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(xₖ, yₖ, zₖ)
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

  Filter filter;

  // This should be run when the inner Newton minimization of the augmented
  // Lagrangian is complete for a given penalty parameter
  auto update_penalty_and_reset_filter = [&] {
    y -= ρ.value() * c_e;
    y_ad.set_value(y);

    z = (z - ρ.value() * c_i).cwiseMax(0.0);
    z_ad.set_value(z);

    // Increase penalty parameter
    if (ρ < ρ_max) {
      ρ.set_value(ρ.value() * 10.0);
    }

    // Update inequality contraint active set
    for (size_t row = 0; row < inequality_constraints.size(); ++row) {
      a(row) = c_i[row] <= 0.0 && z[row] == 0.0;
    }
    diag_a_ad.set_value(a.asDiagonal());

    // Reset the filter when the penalty parameter is updated
    filter.reset();
  };

  // Kept outside the loop so its storage can be reused
  small_vector<Eigen::Triplet<double>> triplets;

  RegularizedLDLT solver{decision_variables.size(), 0};

  // Variables for determining when a step is acceptable
  constexpr double α_red_factor = 0.5;
  constexpr double α_min = 1e-7;
  int acceptable_iter_counter = 0;

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
  [[maybe_unused]]
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
    if (x.lpNorm<Eigen::Infinity>() > 1e10 || !x.allFinite() ||
        c_e.lpNorm<Eigen::Infinity>() > 1e10 ||
        c_i.lpNorm<Eigen::Infinity>() > 1e10) {
      if (ρ < ρ_max) {
        // Try again from the starting point with a larger penalty parameter
        x = x_0;
        x_ad.set_value(x);

        A_e = jacobian_c_e.value();
        A_i = jacobian_c_i.value();
        g = gradient_f.value();
        H = hessian_L.value();
        c_e = c_e_ad.value();
        c_i = c_i_ad.value();

        ρ.set_value(ρ.value() * 10.0);
        continue;
      } else {
        // Report diverging iterates
        return ExitStatus::DIVERGING_ITERATES;
      }
    }

    feasibility_check_profiler.stop();
    ScopedProfiler user_callbacks_profiler{user_callbacks_prof};

    // Call user callbacks
    for (const auto& callback : callbacks) {
      if (callback({iterations, x, Eigen::VectorXd::Zero(0), g, H, A_e, A_i})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    user_callbacks_profiler.stop();
    ScopedProfiler linear_system_build_profiler{linear_system_build_prof};

    // lhs = H
    Eigen::SparseMatrix<double> lhs = H;

    // L(xₖ, yₖ, zₖ) = f(xₖ) − yₖᵀcₑ(xₖ) − zₖᵀcᵢ(xₖ) + 1/2ρcₑᵀcₑ + 1/2ρcᵢᵀIₐcᵢ
    // rhs = −∇L
    //     = −(∇f − Aₑᵀy − Aᵢᵀz + ρAₑᵀcₑ + ρAᵢᵀIₐcᵢ)
    //     = −(∇f − Aₑᵀ(y − ρcₑ) − Aᵢᵀ(z − ρIₐcᵢ))
    //     = −∇f + Aₑᵀ(y − ρcₑ) + Aᵢᵀ(z − ρIₐcᵢ)
    Eigen::SparseMatrix<double> rhs =
        -g + A_e.transpose() * (y - ρ.value() * c_e) +
        A_i.transpose() * (z - ρ.value() * a.asDiagonal() * c_i);

    linear_system_build_profiler.stop();
    ScopedProfiler linear_system_compute_profiler{linear_system_compute_prof};

    Step step;
    double α_max = 1.0;
    double α = 1.0;

    // Solve the Newton-KKT system
    //
    // Hpₖˣ = −∇f + Aₑᵀ(y − ρcₑ) + Aᵢᵀ(z − Iₐcᵢ)
    if (solver.compute(lhs).info() != Eigen::Success) [[unlikely]] {
      return ExitStatus::FACTORIZATION_FAILED;
    }

    linear_system_compute_profiler.stop();
    ScopedProfiler linear_system_solve_profiler{linear_system_solve_prof};

    step.p_x = solver.solve(rhs);

    linear_system_solve_profiler.stop();
    ScopedProfiler line_search_profiler{line_search_prof};

    // Loop until a step is accepted
    while (1) {
      Eigen::VectorXd trial_x = x + α * step.p_x;

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

      // Check whether filter accepts trial iterate
      if (filter.try_add(FilterEntry{f}, α)) {
        // Accept step
        break;
      }

      // Reduce step size
      α *= α_red_factor;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, report line search failure.
      if (α < α_min) {
        double current_kkt_error = kkt_error(g, A_e, c_e, A_i, c_i, y, z);

        trial_x = x + α_max * step.p_x;

        // Upate autodiff
        x_ad.set_value(trial_x);

        trial_c_e = c_e_ad.value();
        trial_c_i = c_i_ad.value();

        double next_kkt_error =
            kkt_error(gradient_f.value(), jacobian_c_e.value(), trial_c_e,
                      jacobian_c_i.value(), trial_c_i, y, z);

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
    x += α * step.p_x;

    // Update autodiff for Jacobians and Hessian
    x_ad.set_value(x);
    A_e = jacobian_c_e.value();
    A_i = jacobian_c_i.value();
    g = gradient_f.value();
    H = hessian_L.value();

    ScopedProfiler next_iter_prep_profiler{next_iter_prep_prof};

    c_e = c_e_ad.value();
    c_i = c_i_ad.value();

    // Update the error estimate
    E_0 = error_estimate(g, A_e, c_e, A_i, c_i, y, z, 0.0, a);
    if (E_0 < options.acceptable_tolerance) {
      ++acceptable_iter_counter;
    } else {
      acceptable_iter_counter = 0;
    }

    double E_ρ = error_estimate(g, A_e, c_e, A_i, c_i, y, z, ρ.value(), a);
    if (E_ρ < options.tolerance) {
      update_penalty_and_reset_filter();
    }

    next_iter_prep_profiler.stop();
    inner_iter_profiler.stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      print_augmented_lagrangian_iteration_diagnostics(
          iterations, IterationType::NORMAL,
          inner_iter_profiler.current_duration(), E_0, f.value(),
          c_e.lpNorm<1>() + c_i.lpNorm<1>(), z.transpose() * c_i, ρ.value(),
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
