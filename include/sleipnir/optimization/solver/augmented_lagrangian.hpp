// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>
#include <cmath>
#include <functional>
#include <span>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/augmented_lagrangian_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/util/all_finite.hpp"
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

namespace slp {

/// Finds the optimal solution to a nonlinear program using the augmented
/// Lagrangian method.
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
/// @param[in] iteration_callbacks The list of callbacks to call at the
///     beginning of each iteration.
/// @param[in] options Solver options.
/// @param[in,out] x The initial guess and output location for the decision
///     variables.
/// @return The exit status.
template <typename Scalar>
ExitStatus augmented_lagrangian(
    const AugmentedLagrangianMatrixCallbacks<Scalar>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options& options, Eigen::Vector<Scalar, Eigen::Dynamic>& x) {
  /// Type alias for dense vector.
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  /// Type alias for sparse vector.
  using SparseVector = Eigen::SparseVector<Scalar>;

  /// Augmented Lagrangian method step direction.
  struct Step {
    /// Primal step.
    DenseVector p_x;
  };

  using std::isfinite;

  const auto solve_start_time = std::chrono::steady_clock::now();

  gch::small_vector<SolveProfiler> solve_profilers;
  solve_profilers.emplace_back("solver");
  solve_profilers.emplace_back("↳ setup");
  solve_profilers.emplace_back("↳ iteration");
  solve_profilers.emplace_back("  ↳ feasibility ✓");
  solve_profilers.emplace_back("  ↳ iter callbacks");
  solve_profilers.emplace_back("  ↳ KKT matrix build");
  solve_profilers.emplace_back("  ↳ KKT matrix decomp");
  solve_profilers.emplace_back("  ↳ KKT system solve");
  solve_profilers.emplace_back("  ↳ line search");
  solve_profilers.emplace_back("    ↳ SOC");
  solve_profilers.emplace_back("  ↳ f(x)");
  solve_profilers.emplace_back("  ↳ ∇f(x)");
  solve_profilers.emplace_back("  ↳ ∇²ₓₓL");
  solve_profilers.emplace_back("  ↳ cₑ(x)");
  solve_profilers.emplace_back("  ↳ ∂cₑ/∂x");
  solve_profilers.emplace_back("  ↳ cᵢ(x)");
  solve_profilers.emplace_back("  ↳ ∂cᵢ/∂x");

  auto& solver_prof = solve_profilers[0];
  auto& setup_prof = solve_profilers[1];
  auto& inner_iter_prof = solve_profilers[2];
  auto& feasibility_check_prof = solve_profilers[3];
  auto& iter_callbacks_prof = solve_profilers[4];
  auto& kkt_matrix_build_prof = solve_profilers[5];
  auto& kkt_matrix_decomp_prof = solve_profilers[6];
  auto& kkt_system_solve_prof = solve_profilers[7];
  auto& line_search_prof = solve_profilers[8];
  [[maybe_unused]]
  auto& soc_prof = solve_profilers[9];

  // Set up profiled matrix callbacks
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  auto& f_prof = solve_profilers[10];
  auto& g_prof = solve_profilers[11];
  auto& H_prof = solve_profilers[12];
  auto& c_e_prof = solve_profilers[13];
  auto& A_e_prof = solve_profilers[14];
  auto& c_i_prof = solve_profilers[15];
  auto& A_i_prof = solve_profilers[16];

  AugmentedLagrangianMatrixCallbacks<Scalar> matrices{
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
      [&](const DenseVector& x, const DenseVector& y, const DenseVector& z,
          Scalar ρ, const DenseVector& a) -> SparseMatrix {
        ScopedProfiler prof{H_prof};
        return matrix_callbacks.H(x, y, z, ρ, a);
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
      }};
#else
  const auto& matrices = matrix_callbacks;
#endif

  solver_prof.start();
  setup_prof.start();

  DenseVector x_0 = x;

  Scalar f = matrices.f(x);
  SparseVector g = matrices.g(x);
  DenseVector c_e = matrices.c_e(x);
  SparseMatrix A_e = matrices.A_e(x);
  DenseVector c_i = matrices.c_i(x);
  SparseMatrix A_i = matrices.A_i(x);

  DenseVector y = DenseVector::Zero(matrices.num_equality_constraints);
  DenseVector z = DenseVector::Zero(matrices.num_inequality_constraints);

  // Penalty parameter
  constexpr Scalar ρ_max(1e10);
  Scalar ρ(1);

  // Inequality constraint active set
  DenseVector a(matrices.num_inequality_constraints);
  for (int row = 0; row < matrices.num_inequality_constraints; ++row) {
    a[row] =
        c_i[row] <= Scalar(0) && z[row] == Scalar(0) ? Scalar(1) : Scalar(0);
  }

  SparseMatrix H = matrices.H(x, y, z, ρ, a);

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

  int iterations = 0;

  Filter<Scalar> filter;

  // This should be run when the inner Newton minimization of the augmented
  // Lagrangian is complete for a given penalty parameter
  auto update_penalty_and_reset_filter = [&] {
    y -= ρ * c_e;
    z = (z - ρ * c_i).cwiseMax(Scalar(0));

    // Increase penalty parameter
    if (ρ < ρ_max) {
      ρ *= Scalar(10);
    }

    // Update inequality contraint active set
    for (int row = 0; row < matrices.num_inequality_constraints; ++row) {
      a[row] =
          c_i[row] <= Scalar(0) && z[row] == Scalar(0) ? Scalar(1) : Scalar(0);
    }

    // Reset the filter when the penalty parameter is updated
    filter.reset();
  };

  RegularizedLDLT<Scalar> solver{matrices.num_decision_variables, 0};

  // Variables for determining when a step is acceptable
  constexpr Scalar α_reduction_factor(0.5);
  constexpr Scalar α_min(1e-7);

  // Error
  Scalar E_0 = kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
      g, A_e, c_e, A_i, c_i, y, z, Scalar(0), a);

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
        c_e.template lpNorm<Eigen::Infinity>() > Scalar(1e10) ||
        c_i.template lpNorm<Eigen::Infinity>() > Scalar(1e10)) {
      if (ρ < ρ_max) {
        // Try again from the starting point with a larger penalty parameter
        x = x_0;

        c_e = matrices.c_e(x);
        c_i = matrices.c_i(x);
        A_e = matrices.A_e(x);
        A_i = matrices.A_i(x);
        g = matrices.g(x);
        H = matrices.H(x, y, z, ρ, a);

        ρ *= Scalar(10);
        continue;
      } else {
        // Report diverging iterates
        return ExitStatus::DIVERGING_ITERATES;
      }
    }

    feasibility_check_profiler.stop();
    ScopedProfiler iter_callbacks_profiler{iter_callbacks_prof};

    // Call iteration callbacks
    for (const auto& callback : iteration_callbacks) {
      if (callback({iterations, x, {}, y, z, g, H, A_e, A_i})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    iter_callbacks_profiler.stop();
    ScopedProfiler kkt_matrix_build_profiler{kkt_matrix_build_prof};

    // lhs = H
    SparseMatrix lhs = H;

    // L(xₖ, yₖ, zₖ) = f(xₖ) − yₖᵀcₑ(xₖ) − zₖᵀcᵢ(xₖ) + 1/2ρcₑᵀcₑ + 1/2ρcᵢᵀIₐcᵢ
    // rhs = −∇L
    //     = −(∇f − Aₑᵀy − Aᵢᵀz + ρAₑᵀcₑ + ρAᵢᵀIₐcᵢ)
    //     = −(∇f − Aₑᵀ(y − ρcₑ) − Aᵢᵀ(z − ρIₐcᵢ))
    //     = −∇f + Aₑᵀ(y − ρcₑ) + Aᵢᵀ(z − ρIₐcᵢ)
    SparseMatrix rhs = -g + A_e.transpose() * (y - ρ * c_e) +
                       A_i.transpose() * (z - ρ * a.asDiagonal() * c_i);

    kkt_matrix_build_profiler.stop();
    ScopedProfiler kkt_matrix_decomp_profiler{kkt_matrix_decomp_prof};

    Step step;
    Scalar α_max(1);
    Scalar α(1);

    // Solve the Newton-KKT system
    //
    // Hpₖˣ = −∇f + Aₑᵀ(y − ρcₑ) + Aᵢᵀ(z − Iₐcᵢ)
    if (solver.compute(lhs).info() != Eigen::Success) [[unlikely]] {
      return ExitStatus::FACTORIZATION_FAILED;
    }

    kkt_matrix_decomp_profiler.stop();
    ScopedProfiler kkt_system_solve_profiler{kkt_system_solve_prof};

    step.p_x = solver.solve(rhs);

    kkt_system_solve_profiler.stop();
    ScopedProfiler line_search_profiler{line_search_prof};

    const FilterEntry<Scalar> current_entry{f};

    // Loop until a step is accepted
    while (1) {
      trial_x = x + α * step.p_x;

      trial_f = matrices.f(trial_x);
      trial_c_e = matrices.c_e(trial_x);
      trial_c_i = matrices.c_i(trial_x);

      // If f(xₖ + αpₖˣ), cₑ(xₖ + αpₖˣ), or cᵢ(xₖ + αpₖˣ) aren't finite, reduce
      // step size immediately
      if (!isfinite(trial_f) || !trial_c_e.allFinite() ||
          !trial_c_i.allFinite()) {
        // Reduce step size
        α *= α_reduction_factor;
        continue;
      }

      // Check whether filter accepts trial iterate
      FilterEntry trial_entry{trial_f};
      if (filter.try_add(current_entry, trial_entry, step.p_x, g, α)) {
        // Accept step
        break;
      }

      // Reduce step size
      α *= α_reduction_factor;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, report line search failure.
      if (α < α_min) {
        Scalar current_kkt_error = kkt_error<Scalar, KKTErrorType::ONE_NORM>(
            g, A_e, c_e, A_i, c_i, y, z, ρ, a);

        trial_x = x + α_max * step.p_x;

        trial_c_e = matrices.c_e(trial_x);
        trial_c_i = matrices.c_i(trial_x);

        Scalar next_kkt_error = kkt_error<Scalar, KKTErrorType::ONE_NORM>(
            matrices.g(trial_x), matrices.A_e(trial_x), trial_c_e,
            matrices.A_i(trial_x), trial_c_i, y, z, ρ, a);

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (next_kkt_error <= Scalar(0.999) * current_kkt_error) {
          α = α_max;

          // Accept step
          break;
        }

        return ExitStatus::LOCALLY_INFEASIBLE;
      }
    }

    line_search_profiler.stop();

    // Update iterates
    x = trial_x;

    f = trial_f;
    c_e = trial_c_e;
    c_i = trial_c_i;

    // Update autodiff for Jacobians and Hessian
    A_e = matrices.A_e(x);
    A_i = matrices.A_i(x);
    g = matrices.g(x);
    H = matrices.H(x, y, z, ρ, a);

    // Update the error
    E_0 = kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
        g, A_e, c_e, A_i, c_i, y, z, Scalar(0), a);

    Scalar E_ρ = kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
        g, A_e, c_e, A_i, c_i, y, z, ρ, a);
    if (E_ρ < Scalar(options.tolerance)) {
      update_penalty_and_reset_filter();
    }

    inner_iter_profiler.stop();

    if (options.diagnostics) {
      print_augmented_lagrangian_iteration_diagnostics(
          iterations, IterationType::NORMAL,
          inner_iter_profiler.current_duration(), E_0, f,
          c_e.template lpNorm<1>() + c_i.template lpNorm<1>(),
          Scalar(z.transpose() * c_i), ρ, solver.hessian_regularization(), α,
          α_max, α_reduction_factor, Scalar(1));
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

extern template SLEIPNIR_DLLEXPORT ExitStatus augmented_lagrangian(
    const AugmentedLagrangianMatrixCallbacks<double>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options& options, Eigen::Vector<double, Eigen::Dynamic>& x);

}  // namespace slp
