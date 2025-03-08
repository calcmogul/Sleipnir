// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/augmented_lagrangian.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "optimization/regularized_ldlt.hpp"
#include "optimization/solver/util/error_estimate.hpp"
#include "optimization/solver/util/filter.hpp"
#include "optimization/solver/util/is_locally_infeasible.hpp"
#include "optimization/solver/util/kkt_error.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/util/assert.hpp"
#include "util/print_diagnostics.hpp"
#include "util/scope_exit.hpp"
#include "util/scoped_profiler.hpp"
#include "util/solve_profiler.hpp"

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
    const AugmentedLagrangianMatrixCallbacks& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo& info)>>
        iteration_callbacks,
    const Options& options, Eigen::VectorXd& x) {
  const auto solve_start_time = std::chrono::steady_clock::now();

  gch::small_vector<SolveProfiler> solve_profilers;
  solve_profilers.emplace_back("solver");
  solve_profilers.emplace_back("  ↳ setup");
  solve_profilers.emplace_back("  ↳ iteration");
  solve_profilers.emplace_back("    ↳ feasibility ✓");
  solve_profilers.emplace_back("    ↳ iteration callbacks");
  solve_profilers.emplace_back("    ↳ iter matrix build");
  solve_profilers.emplace_back("    ↳ iter matrix compute");
  solve_profilers.emplace_back("    ↳ iter matrix solve");
  solve_profilers.emplace_back("    ↳ line search");
  solve_profilers.emplace_back("      ↳ SOC");
  solve_profilers.emplace_back("    ↳ next iter prep");

  auto& solver_prof = solve_profilers[0];
  auto& setup_prof = solve_profilers[1];
  auto& inner_iter_prof = solve_profilers[2];
  auto& feasibility_check_prof = solve_profilers[3];
  auto& iteration_callbacks_prof = solve_profilers[4];
  auto& linear_system_build_prof = solve_profilers[5];
  auto& linear_system_compute_prof = solve_profilers[6];
  auto& linear_system_solve_prof = solve_profilers[7];
  auto& line_search_prof = solve_profilers[8];
  [[maybe_unused]]
  auto& soc_prof = solve_profilers[9];
  auto& next_iter_prep_prof = solve_profilers[10];

  // Set up profiled matrix callbacks
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  auto& f_prof = solve_profilers[11];
  auto& g_prof = solve_profilers[12];
  auto& H_prof = solve_profilers[13];
  auto& c_e_prof = solve_profilers[14];
  auto& A_e_prof = solve_profilers[15];
  auto& c_i_prof = solve_profilers[16];
  auto& A_i_prof = solve_profilers[17];

  AugmentedLagrangianMatrixCallbacks matrices{
      [&](const Eigen::VectorXd& x) -> double {
        ScopedProfiler prof{f_prof};
        return matrix_callbacks.f(x);
      },
      [&](const Eigen::VectorXd& x) -> Eigen::SparseVector<double> {
        ScopedProfiler prof{g_prof};
        return matrix_callbacks.g(x);
      },
      [&](const Eigen::VectorXd& x, const Eigen::VectorXd& y,
          const Eigen::VectorXd& z, double ρ,
          const Eigen::VectorXd& a) -> Eigen::SparseMatrix<double> {
        ScopedProfiler prof{H_prof};
        return matrix_callbacks.H(x, y, z, ρ, a);
      },
      [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        ScopedProfiler prof{c_e_prof};
        return matrix_callbacks.c_e(x);
      },
      [&](const Eigen::VectorXd& x) -> Eigen::SparseMatrix<double> {
        ScopedProfiler prof{A_e_prof};
        return matrix_callbacks.A_e(x);
      },
      [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        ScopedProfiler prof{c_i_prof};
        return matrix_callbacks.c_i(x);
      },
      [&](const Eigen::VectorXd& x) -> Eigen::SparseMatrix<double> {
        ScopedProfiler prof{A_i_prof};
        return matrix_callbacks.A_i(x);
      }};
#else
  const auto& matrices = matrix_callbacks;
#endif

  solver_prof.start();
  setup_prof.start();

  Eigen::VectorXd x_0 = x;

  double f = matrices.f(x);
  Eigen::VectorXd c_e = matrices.c_e(x);
  Eigen::VectorXd c_i = matrices.c_i(x);

  int num_decision_variables = x.rows();
  int num_equality_constraints = c_e.rows();
  int num_inequality_constraints = c_i.rows();

  // Check for overconstrained problem
  if (num_equality_constraints > num_decision_variables) {
    if (options.diagnostics) {
      print_too_few_dofs_error(c_e);
    }

    return ExitStatus::TOO_FEW_DOFS;
  }

  Eigen::SparseVector<double> g = matrices.g(x);
  Eigen::SparseMatrix<double> A_e = matrices.A_e(x);
  Eigen::SparseMatrix<double> A_i = matrices.A_i(x);

  Eigen::VectorXd y = Eigen::VectorXd::Zero(num_equality_constraints);
  Eigen::VectorXd z = Eigen::VectorXd::Zero(num_inequality_constraints);

  // Penalty parameter
  constexpr double ρ_max = 1e10;
  double ρ = 1.0;

  // Inequality constraint active set
  Eigen::VectorXd a(num_inequality_constraints);
  for (int row = 0; row < num_inequality_constraints; ++row) {
    a[row] = c_i[row] <= 0.0 && z[row] == 0.0;
  }

  Eigen::SparseMatrix<double> H = matrices.H(x, y, z, ρ, a);

  // Ensure matrix callback dimensions are consistent
  slp_assert(g.rows() == num_decision_variables);
  slp_assert(A_e.rows() == num_equality_constraints);
  slp_assert(A_e.cols() == num_decision_variables);
  slp_assert(A_i.rows() == num_inequality_constraints);
  slp_assert(A_i.cols() == num_decision_variables);
  slp_assert(H.rows() == num_decision_variables);
  slp_assert(H.cols() == num_decision_variables);

  // Check whether initial guess has finite f(xₖ), cₑ(xₖ), and cᵢ(xₖ)
  if (!std::isfinite(f) || !c_e.allFinite() || !c_i.allFinite()) {
    return ExitStatus::NONFINITE_INITIAL_COST_OR_CONSTRAINTS;
  }

  int iterations = 0;

  Filter filter;

  // This should be run when the inner Newton minimization of the augmented
  // Lagrangian is complete for a given penalty parameter
  auto update_penalty_and_reset_filter = [&] {
    y -= ρ * c_e;
    z = (z - ρ * c_i).cwiseMax(0.0);

    // Increase penalty parameter
    if (ρ < ρ_max) {
      ρ *= 10.0;
    }

    // Update inequality contraint active set
    for (int row = 0; row < num_inequality_constraints; ++row) {
      a[row] = c_i[row] <= 0.0 && z[row] == 0.0;
    }

    // Reset the filter when the penalty parameter is updated
    filter.reset();
  };

  // Kept outside the loop so its storage can be reused
  gch::small_vector<Eigen::Triplet<double>> triplets;

  RegularizedLDLT solver{num_decision_variables, 0};

  // Variables for determining when a step is acceptable
  constexpr double α_reduction_factor = 0.5;
  constexpr double α_min = 1e-7;

  // Error estimate
  double E_0 = std::numeric_limits<double>::infinity();

  setup_prof.stop();

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

  while (E_0 > options.tolerance) {
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

        c_e = matrices.c_e(x);
        c_i = matrices.c_i(x);
        A_e = matrices.A_e(x);
        A_i = matrices.A_i(x);
        g = matrices.g(x);
        H = matrices.H(x, y, z, ρ, a);

        ρ *= 10.0;
        continue;
      } else {
        // Report diverging iterates
        return ExitStatus::DIVERGING_ITERATES;
      }
    }

    feasibility_check_profiler.stop();
    ScopedProfiler iteration_callbacks_profiler{iteration_callbacks_prof};

    // Call iteration callbacks
    for (const auto& callback : iteration_callbacks) {
      if (callback({iterations, x, g, H, A_e, A_i})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    iteration_callbacks_profiler.stop();
    ScopedProfiler linear_system_build_profiler{linear_system_build_prof};

    // lhs = H
    Eigen::SparseMatrix<double> lhs = H;

    // L(xₖ, yₖ, zₖ) = f(xₖ) − yₖᵀcₑ(xₖ) − zₖᵀcᵢ(xₖ) + 1/2ρcₑᵀcₑ + 1/2ρcᵢᵀIₐcᵢ
    // rhs = −∇L
    //     = −(∇f − Aₑᵀy − Aᵢᵀz + ρAₑᵀcₑ + ρAᵢᵀIₐcᵢ)
    //     = −(∇f − Aₑᵀ(y − ρcₑ) − Aᵢᵀ(z − ρIₐcᵢ))
    //     = −∇f + Aₑᵀ(y − ρcₑ) + Aᵢᵀ(z − ρIₐcᵢ)
    Eigen::SparseMatrix<double> rhs =
        -g + A_e.transpose() * (y - ρ * c_e) +
        A_i.transpose() * (z - ρ * a.asDiagonal() * c_i);

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

      double trial_f = matrices.f(trial_x);
      Eigen::VectorXd trial_c_e = matrices.c_e(trial_x);
      Eigen::VectorXd trial_c_i = matrices.c_i(trial_x);

      // If f(xₖ + αpₖˣ), cₑ(xₖ + αpₖˣ), or cᵢ(xₖ + αpₖˣ) aren't finite, reduce
      // step size immediately
      if (!std::isfinite(trial_f) || !trial_c_e.allFinite() ||
          !trial_c_i.allFinite()) {
        // Reduce step size
        α *= α_reduction_factor;
        continue;
      }

      // Check whether filter accepts trial iterate
      if (filter.try_add(FilterEntry{f}, α)) {
        // Accept step
        break;
      }

      // Reduce step size
      α *= α_reduction_factor;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, report line search failure.
      if (α < α_min) {
        double current_kkt_error = kkt_error(g, A_e, c_e, A_i, c_i, y, z);

        trial_x = x + α_max * step.p_x;

        trial_c_e = matrices.c_e(trial_x);
        trial_c_i = matrices.c_i(trial_x);

        double next_kkt_error =
            kkt_error(matrices.g(trial_x), matrices.A_e(trial_x), trial_c_e,
                      matrices.A_i(trial_x), trial_c_i, y, z);

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

    // xₖ₊₁ = xₖ + αₖpₖˣ
    x += α * step.p_x;

    // Update autodiff for Jacobians and Hessian
    f = matrices.f(x);
    A_e = matrices.A_e(x);
    A_i = matrices.A_i(x);
    g = matrices.g(x);
    H = matrices.H(x, y, z, ρ, a);

    ScopedProfiler next_iter_prep_profiler{next_iter_prep_prof};

    c_e = matrices.c_e(x);
    c_i = matrices.c_i(x);

    // Update the error estimate
    E_0 = error_estimate(g, A_e, c_e, A_i, c_i, y, z, 0.0, a);

    double E_ρ = error_estimate(g, A_e, c_e, A_i, c_i, y, z, ρ, a);
    if (E_ρ < options.tolerance) {
      update_penalty_and_reset_filter();
    }

    next_iter_prep_profiler.stop();
    inner_iter_profiler.stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (options.diagnostics) {
      print_augmented_lagrangian_iteration_diagnostics(
          iterations, IterationType::NORMAL,
          inner_iter_profiler.current_duration(), E_0, f,
          c_e.lpNorm<1>() + c_i.lpNorm<1>(), z.transpose() * c_i, ρ,
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
  }

  return ExitStatus::SUCCESS;
}

}  // namespace slp
