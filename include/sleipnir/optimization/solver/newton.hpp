// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <span>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/newton_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/util/all_finite.hpp"
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

/// Finds the optimal solution to a nonlinear program using Newton's method.
///
/// A nonlinear program has the form:
///
/// ```
/// min_x f(x)
/// ```
///
/// where f(x) is the cost function.
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
ExitStatus newton(
    const NewtonMatrixCallbacks<Scalar>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<Scalar>& info)>>
        iteration_callbacks,
    const Options& options, Eigen::Vector<Scalar, Eigen::Dynamic>& x) {
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using SparseVector = Eigen::SparseVector<Scalar>;

  using std::isfinite;

  const auto solve_start_time = std::chrono::steady_clock::now();

  using enum SolveTimeTracerID;

  int iterations = 0;

  // Prints final solver diagnostics when the solver exits
  SolveDiagnostics<newton_time_trace> diagnostics{options.diagnostics,
                                                  options.diagnostics};

  // Set up traced matrix callbacks if diagnostics are enabled
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  std::optional<NewtonMatrixCallbacks<Scalar>> traced_matrices;
  if (options.diagnostics) {
    traced_matrices.emplace(matrix_callbacks.num_decision_variables,
                            traced(diagnostics[COST], matrix_callbacks.f),
                            traced(diagnostics[GRADIENT], matrix_callbacks.g),
                            traced(diagnostics[HESSIAN], matrix_callbacks.H),
                            matrix_callbacks.scaling);
  }
  const auto& matrices = traced_matrices ? *traced_matrices : matrix_callbacks;
#else
  const auto& matrices = matrix_callbacks;
#endif

  ScopedTimeTracer setup_tracer{diagnostics[SETUP]};

  Scalar f = matrices.f(x);
  SparseVector g = matrices.g(x);
  SparseMatrix H = matrices.H(x);

  // Ensure matrix callback dimensions are consistent
  slp_assert(g.rows() == matrices.num_decision_variables);
  slp_assert(H.rows() == matrices.num_decision_variables);
  slp_assert(H.cols() == matrices.num_decision_variables);

  DenseVector trial_x;

  Scalar trial_f;

  // Check whether initial guess has finite cost and derivatives
  if (!isfinite(f) || !all_finite(g) || !all_finite(H)) {
    return ExitStatus::NONFINITE_INITIAL_GUESS;
  }

  Filter<Scalar> filter;

  KKTSolver<Scalar> solver{
      // Use sparse solver if lower triangle fills < 25% of system
      H.nonZeros() < 0.25 * H.size(), matrices.num_decision_variables, 0};

  // Step size reduction factor
  constexpr Scalar α_reduction_factor(0.5);

  // The smallest KKT error of a step accepted by the KKT error reduction
  // fallback. Each fallback step must improve on it, so finitely many fallback
  // steps are taken before the solver converges.
  Scalar fallback_kkt_error = std::numeric_limits<Scalar>::infinity();

  // Error
  Scalar E_0 = unscaled_kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
      matrices.scaling, g);

  setup_tracer.stop();

  // Prints initial iterate diagnostics now and closes the iterate diagnostics
  // table when the solver exits
  IterateDiagnosticsTable table{options.diagnostics, E_0, f, Scalar(0),
                                Scalar(0)};

  while (E_0 > Scalar(options.tolerance)) {
    ScopedTimeTracer inner_iter_tracer{diagnostics[ITERATION]};

    // Check for diverging iterates
    if (x.template lpNorm<Eigen::Infinity>() > Scalar(1e10) || !x.allFinite()) {
      return ExitStatus::DIVERGING_ITERATES;
    }

    PhaseTimeTracer phase_tracer{diagnostics[CALLBACKS]};

    // Call iteration callbacks
    for (const auto& callback : iteration_callbacks) {
      if (callback({iterations, x, {}, {}, g, H, {}, {}})) {
        return ExitStatus::CALLBACK_REQUESTED_STOP;
      }
    }

    phase_tracer.next(diagnostics[KKT_MATRIX_DECOMP]);

    // Solve the Newton-KKT system
    //
    // Hpˣ = −∇f
    solver.compute(H);

    phase_tracer.next(diagnostics[KKT_SYSTEM_SOLVE]);

    DenseVector p_x = solver.solve(-g);

    phase_tracer.next(diagnostics[LINE_SEARCH]);

    constexpr Scalar α_max(1);
    Scalar α = α_max;
    const FilterEntry<Scalar> current_entry{f};
    const Scalar D_ϕ = g.transpose() * p_x;

    const Scalar α_min = filter.min_step_size(current_entry, D_ϕ);

    // α_min is zero when the search direction is a descent direction, so also
    // end the line search when the step becomes negligible relative to the
    // iterate. This bounds the number of step size reductions.
    //
    // A step αpˣ is negligible if
    //
    //   ‖αpˣ/(1 + |x|)‖_∞ < 10ε
    //
    // where the absolute value and division are elementwise. Since α > 0, it
    // factors out of the norm.
    //
    //   α‖pˣ/(1 + |x|)‖_∞ < 10ε
    //
    // Dividing both sides by the norm gives
    //
    //   α < 10ε / ‖pˣ/(1 + |x|)‖_∞
    const Scalar α_tiny = Scalar(10) * std::numeric_limits<Scalar>::epsilon() /
                          (p_x.array().abs() / (Scalar(1) + x.array().abs()))
                              .matrix()
                              .template lpNorm<Eigen::Infinity>();

    // Loop until a step is accepted
    while (1) {
      trial_x = x + α * p_x;

      trial_f = matrices.f(trial_x);

      // If f(xₖ + αpₖˣ) isn't finite, reduce step size immediately
      if (!isfinite(trial_f)) {
        // Reduce step size
        α *= α_reduction_factor;

        if (α < α_min || α < α_tiny) {
          return ExitStatus::LINE_SEARCH_FAILED;
        }
        continue;
      }

      // Check whether filter accepts trial iterate
      if (filter.try_add(current_entry, FilterEntry{trial_f}, D_ϕ, α)) {
        // Accept step
        break;
      }

      // Reduce step size
      α *= α_reduction_factor;

      // If step size hit a minimum or the step became negligible, check if the
      // KKT error was reduced. If it wasn't, report line search failure.
      if (α < α_min || α < α_tiny) {
        Scalar current_kkt_error = kkt_error<Scalar, KKTErrorType::ONE_NORM>(g);

        trial_x = x + α_max * p_x;

        trial_f = matrices.f(trial_x);

        Scalar next_kkt_error =
            kkt_error<Scalar, KKTErrorType::ONE_NORM>(matrices.g(trial_x));

        // If the step using αᵐᵃˣ reduced the KKT error below the current
        // iterate's and every previous fallback step's, accept it anyway
        if (next_kkt_error <=
            Scalar(0.999) * std::min(current_kkt_error, fallback_kkt_error)) {
          // Accept step
          fallback_kkt_error = next_kkt_error;
          α = α_max;
          break;
        }

        return ExitStatus::LINE_SEARCH_FAILED;
      }
    }

    phase_tracer.stop();

    // Update iterates
    x = trial_x;

    f = trial_f;

    // Update autodiff for Hessian
    g = matrices.g(x);
    H = matrices.H(x);

    // Update the error
    E_0 = unscaled_kkt_error<Scalar, KKTErrorType::INF_NORM_SCALED>(
        matrices.scaling, g);

    inner_iter_tracer.stop();

    if (options.diagnostics) {
      print_iterate_diagnostics(
          iterations, IterationType::NORMAL,
          inner_iter_tracer.current_duration(), E_0, f, Scalar(0), Scalar(0),
          solver.hessian_regularization(),
          solver.constraint_jacobian_regularization(),
          p_x.template lpNorm<Eigen::Infinity>(), Scalar(1), α, α_max,
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

  if (!isfinite(E_0)) {
    return ExitStatus::DIVERGING_ITERATES;
  } else {
    return ExitStatus::SUCCESS;
  }
}

extern template SLEIPNIR_DLLEXPORT ExitStatus
newton(const NewtonMatrixCallbacks<double>& matrix_callbacks,
       std::span<std::function<bool(const IterationInfo<double>& info)>>
           iteration_callbacks,
       const Options& options, Eigen::Vector<double, Eigen::Dynamic>& x);

}  // namespace slp
