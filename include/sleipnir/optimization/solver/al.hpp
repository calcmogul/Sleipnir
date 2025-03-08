// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/al_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/util/all_finite.hpp"
#include "sleipnir/optimization/solver/util/constraint_violation.hpp"
#include "sleipnir/optimization/solver/util/filter.hpp"
#include "sleipnir/optimization/solver/util/kkt_error.hpp"
#include "sleipnir/optimization/solver/util/kkt_solver.hpp"
#include "sleipnir/optimization/solver/util/time_trace.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/print_diagnostics.hpp"
#include "sleipnir/util/symbol_exports.hpp"
#include "sleipnir/util/time_tracer.hpp"

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
ExitStatus al(const ALMatrixCallbacks<Scalar>& matrix_callbacks,
              std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
                  iteration_callbacks,
              const Options& options,
              Eigen::Vector<Scalar, Eigen::Dynamic>& x) {
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

  using std::abs;
  using std::isfinite;
  using std::max;
  using std::min;

  const auto solve_start_time = std::chrono::steady_clock::now();

  using enum SolveTimeTracerID;

  // Prints final solver diagnostics when the solver exits
  SolveDiagnostics<al_time_trace> diagnostics{options.diagnostics,
                                              options.diagnostics};

  // Set up traced matrix callbacks if diagnostics are enabled
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  std::optional<ALMatrixCallbacks<Scalar>> traced_matrices;
  if (options.diagnostics) {
    traced_matrices.emplace(
        matrix_callbacks.num_decision_variables,
        matrix_callbacks.num_equality_constraints,
        matrix_callbacks.num_inequality_constraints,
        traced(diagnostics[COST], matrix_callbacks.f),
        traced(diagnostics[GRADIENT], matrix_callbacks.g),
        traced(diagnostics[HESSIAN], matrix_callbacks.H),
        traced(diagnostics[EQUALITY_CONSTRAINTS], matrix_callbacks.c_e),
        traced(diagnostics[EQUALITY_JACOBIAN], matrix_callbacks.A_e),
        traced(diagnostics[INEQUALITY_CONSTRAINTS], matrix_callbacks.c_i),
        traced(diagnostics[INEQUALITY_JACOBIAN], matrix_callbacks.A_i),
        matrix_callbacks.scaling);
  }
  const auto& matrices = traced_matrices ? *traced_matrices : matrix_callbacks;
#else
  const auto& matrices = matrix_callbacks;
#endif

  ScopedTimeTracer setup_tracer{diagnostics[SETUP]};

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
    a[row] = z[row] - ρ * c_i[row] > Scalar(0) ? Scalar(1) : Scalar(0);
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

  // Augmented Lagrangian merit function
  //
  //   ψ(x) = f(x) − yᵀcₑ(x) + ρ/2 cₑᵀcₑ
  //     + 1/(2ρ) Σᵢ (max(0, zᵢ − ρcᵢ(x))² − zᵢ²)
  auto merit = [&](Scalar f, const DenseVector& c_e,
                   const DenseVector& c_i) -> Scalar {
    return f - y.dot(c_e) + ρ / Scalar(2) * c_e.squaredNorm() +
           ((z - ρ * c_i).cwiseMax(Scalar(0)).squaredNorm() - z.squaredNorm()) /
               (Scalar(2) * ρ);
  };

  Filter<Scalar> filter;

  // Error
  Scalar E_0 = kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
      g, A_e, c_e, A_i, c_i, y, z, Scalar(0), a);

  // Stationarity tolerance for the inner minimization of ψ
  const Scalar tolerance(options.tolerance);
  Scalar ω(1);

  // Constraint violation at the previous multiplier update
  Scalar prev_violation = std::numeric_limits<Scalar>::infinity();

  // This should be run whenever y, z, or ρ change
  auto update_subproblem = [&] {
    // Update active set
    for (int row = 0; row < matrices.num_inequality_constraints; ++row) {
      a[row] = z[row] - ρ * c_i[row] > Scalar(0) ? Scalar(1) : Scalar(0);
    }

    // Update Hessian for new multipliers, penalty parameter, and active set
    H = matrices.H(x, y, z, ρ, a);

    E_0 = kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
        g, A_e, c_e, A_i, c_i, y, z, Scalar(0), a);

    // Reset the filter when the subproblem changes
    filter.reset();
  };

  // Increases the penalty parameter
  auto increase_penalty = [&] { ρ = min(Scalar(10) * ρ, ρ_max); };

  // This should be run when the inner minimization of ψ has converged to
  // within ω. The multipliers are always updated, and the penalty parameter is
  // increased if the constraint violation didn't decrease sufficiently. See
  // algorithm 4.1 of [3].
  auto update_multipliers_and_penalty = [&] {
    // ‖(cₑ, min(cᵢ, z/ρ))‖_∞
    Scalar violation =
        max(c_e.template lpNorm<Eigen::Infinity>(),
            c_i.cwiseMin(z / ρ).template lpNorm<Eigen::Infinity>());

    y -= ρ * c_e;
    z = (z - ρ * c_i).cwiseMax(Scalar(0));

    if (violation > Scalar(0.5) * prev_violation) {
      increase_penalty();
    }
    prev_violation = violation;

    ω = max(Scalar(0.1) * ω, tolerance);

    update_subproblem();
  };

  KKTSolver<Scalar> solver{
      // Use sparse solver if lower triangle fills < 25% of system
      H.nonZeros() < 0.25 * H.rows() * H.rows(),
      matrices.num_decision_variables, 0};

  // Variables for determining when a step is acceptable
  constexpr Scalar α_reduction_factor(0.5);
  constexpr Scalar α_min(1e-7);

  setup_tracer.stop();

  // Prints initial iterate diagnostics now and closes the iterate diagnostics
  // table when the solver exits
  IterateDiagnosticsTable table{options.diagnostics, [&] {
                                  print_al_initial_iterate_diagnostics(
                                      E_0, f, constraint_violation(c_e, c_i),
                                      Scalar(z.transpose() * c_i), ρ);
                                }};

  while (E_0 > Scalar(options.tolerance)) {
    ScopedTimeTracer inner_iter_tracer{diagnostics[ITERATION]};

    // Check for diverging iterates
    if (x.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !x.allFinite() ||
        c_e.template lpNorm<Eigen::Infinity>() > Scalar(1e10) ||
        c_i.template lpNorm<Eigen::Infinity>() > Scalar(1e10)) {
      if (ρ < ρ_max) {
        // Try again from the starting point with a larger penalty parameter
        x = x_0;
        increase_penalty();

        f = matrices.f(x);
        c_e = matrices.c_e(x);
        c_i = matrices.c_i(x);
        A_e = matrices.A_e(x);
        A_i = matrices.A_i(x);
        g = matrices.g(x);

        update_subproblem();

        continue;
      } else {
        // Report diverging iterates
        return ExitStatus::DIVERGING_ITERATES;
      }
    }

    PhaseTimeTracer phase_tracer{diagnostics[CALLBACKS]};

    // Call iteration callbacks
    for (const auto& callback : iteration_callbacks) {
      if (callback({iterations, x, {}, y, z, g, H, A_e, A_i})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    phase_tracer.next(diagnostics[KKT_MATRIX_BUILD]);

    // lhs = H
    SparseMatrix lhs = H;

    // rhs = −∇ψ
    //     = −(∇f − Aₑᵀ(y − ρcₑ) − Aᵢᵀ max(0, z − ρcᵢ))
    //     = −∇f + Aₑᵀ(y − ρcₑ) + Aᵢᵀ diag(a)(z − ρcᵢ)
    SparseMatrix rhs = -g + A_e.transpose() * (y - ρ * c_e) +
                       A_i.transpose() * (a.asDiagonal() * (z - ρ * c_i));

    phase_tracer.next(diagnostics[KKT_MATRIX_DECOMP]);

    Step step;
    Scalar α_max(1);
    Scalar α(1);

    // Solve the Newton-KKT system
    //
    // Hpₖˣ = −∇f + Aₑᵀ(y − ρcₑ) + Aᵢᵀ diag(a)(z − ρcᵢ)
    if (solver.compute(lhs).info() != Eigen::Success) [[unlikely]] {
      return ExitStatus::FACTORIZATION_FAILED;
    }

    phase_tracer.next(diagnostics[KKT_SYSTEM_SOLVE]);

    step.p_x = solver.solve(rhs);

    phase_tracer.next(diagnostics[LINE_SEARCH]);

    const FilterEntry<Scalar> current_entry{merit(f, c_e, c_i)};

    // Compute the directional derivative of the merit function along the
    // search direction.
    //
    //   D_ψ = ∇ψ(x)ᵀpˣ = −rhsᵀpˣ
    Scalar D_ψ = -(rhs.transpose() * step.p_x).value();

    // The inertia correction makes H positive definite, so the Newton step is
    // a descent direction in exact arithmetic. If round-off from an
    // ill-conditioned H breaks that, fall back to the steepest descent
    // direction. Otherwise, the filter would accept any step because the merit
    // function entries have no constraint violation.
    if (D_ψ >= Scalar(0)) {
      step.p_x = rhs;
      D_ψ = -step.p_x.squaredNorm();
    }

    // Start the line search at the minimizer of φ(α) = ψ(xₖ + αpₖˣ) along the
    // step. The Newton step minimizes the current piece of the piecewise
    // quadratic ψ, so when it crosses into a piece where more constraints are
    // active, backtracking by halving tends to stop short of the breakpoint
    // and the new constraint doesn't become active.
    //
    // Linearizing the constraints along the step, constraint i is active where
    //
    //   tᵢ(α) = cᵢ − zᵢ/ρ + α(Aᵢpₖˣ)ᵢ < 0
    //
    // so it enters or leaves the active set at the breakpoint
    //
    //   αᵢ = −(cᵢ − zᵢ/ρ)/(Aᵢpₖˣ)ᵢ
    //
    // Between breakpoints, φ′ is linear with slope pₖˣᵀHpₖˣ, which changes by
    // ±ρ(Aᵢpₖˣ)ᵢ² at each breakpoint. Walking the sorted breakpoints finds
    // the root of φ′. This is exact for a quadratic cost and linear
    // constraints and an estimate otherwise, so the filter still checks the
    // step and backtracking remains as a fallback.
    if (D_ψ < Scalar(0)) {
      DenseVector A_i_p = A_i * step.p_x;

      gch::small_vector<std::pair<Scalar, int>> breakpoints;
      for (int row = 0; row < matrices.num_inequality_constraints; ++row) {
        if (A_i_p[row] != Scalar(0)) {
          Scalar α_row = -(c_i[row] - z[row] / ρ) / A_i_p[row];
          if (α_row > Scalar(0) && α_row < α_max) {
            breakpoints.emplace_back(α_row, row);
          }
        }
      }
      std::sort(breakpoints.begin(), breakpoints.end());

      // φ′ and its slope at the start of the current segment
      Scalar α_start(0);
      Scalar D_φ = D_ψ;
      Scalar slope =
          step.p_x.dot(H.template selfadjointView<Eigen::Lower>() * step.p_x);

      for (const auto& [α_row, row] : breakpoints) {
        // Stop if φ′ crosses zero before this breakpoint
        Scalar D_φ_row = D_φ + slope * (α_row - α_start);
        if (D_φ_row >= Scalar(0)) {
          break;
        }

        α_start = α_row;
        D_φ = D_φ_row;

        // Constraints moving toward violation enter the active set and add
        // curvature; the rest leave it
        Scalar Δslope = ρ * A_i_p[row] * A_i_p[row];
        slope += A_i_p[row] < Scalar(0) ? Δslope : -Δslope;
      }

      if (slope > Scalar(0)) {
        α = min(α_start - D_φ / slope, α_max);
      }
    }

    // If the step's predicted decrease is below the round-off error in ψ, as
    // happens near convergence, the merit function can't distinguish good
    // steps from bad ones. In that case, accept the full step if it reduces
    // the stationarity error ‖∇ψ‖_∞ instead.
    bool step_accepted = false;
    if (-D_ψ <= Scalar(10) * std::numeric_limits<Scalar>::epsilon() *
                    abs(current_entry.cost)) {
      trial_x = x + α_max * step.p_x;

      trial_f = matrices.f(trial_x);
      trial_c_e = matrices.c_e(trial_x);
      trial_c_i = matrices.c_i(trial_x);

      if (isfinite(trial_f) && trial_c_e.allFinite() && trial_c_i.allFinite()) {
        // ∇ψ = ∇f − Aₑᵀ(y − ρcₑ) − Aᵢᵀ max(0, z − ρcᵢ)
        DenseVector trial_gψ =
            matrices.g(trial_x) -
            matrices.A_e(trial_x).transpose() * (y - ρ * trial_c_e) -
            matrices.A_i(trial_x).transpose() *
                (z - ρ * trial_c_i).cwiseMax(Scalar(0));

        if (trial_gψ.template lpNorm<Eigen::Infinity>() <
            DenseVector(rhs).template lpNorm<Eigen::Infinity>()) {
          α = α_max;
          step_accepted = true;
        }
      }
    }

    // Loop until a step is accepted
    while (!step_accepted) {
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
      FilterEntry trial_entry{merit(trial_f, trial_c_e, trial_c_i)};
      if (filter.try_add(current_entry, trial_entry, D_ψ, α)) {
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

    phase_tracer.stop();

    // Update iterates
    x = trial_x;

    f = trial_f;
    c_e = trial_c_e;
    c_i = trial_c_i;

    // Update active set
    for (int row = 0; row < matrices.num_inequality_constraints; ++row) {
      a[row] = z[row] - ρ * c_i[row] > Scalar(0) ? Scalar(1) : Scalar(0);
    }

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
    if (E_ρ <= ω) {
      update_multipliers_and_penalty();
    }

    inner_iter_tracer.stop();

    if (options.diagnostics) {
      print_al_iterate_diagnostics(
          iterations, IterationType::NORMAL,
          inner_iter_tracer.current_duration(), E_0, f,
          constraint_violation(c_e, c_i), Scalar(z.transpose() * c_i), ρ,
          solver.hessian_regularization(),
          step.p_x.template lpNorm<Eigen::Infinity>(), Scalar(0), α, α_max,
          α_reduction_factor, Scalar(1));
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

extern template SLEIPNIR_DLLEXPORT ExitStatus
al(const ALMatrixCallbacks<double>& matrix_callbacks,
   std::span<std::function<bool(const IterationInfo<double>& info)>>
       iteration_callbacks,
   const Options& options, Eigen::Vector<double, Eigen::Dynamic>& x);

}  // namespace slp
