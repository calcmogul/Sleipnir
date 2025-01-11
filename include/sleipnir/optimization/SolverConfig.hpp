// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>

#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * Solver configuration.
 */
struct SLEIPNIR_DLLEXPORT SolverConfig {
  /// The solver will stop once the error is below this tolerance.
  double tolerance = 1e-8;

  /// The maximum number of solver iterations before returning a solution.
  int maxIterations = 5000;

  /// The solver will stop once the error is below this tolerance for
  /// `acceptableIterations` iterations. This is useful in cases where the
  /// solver might not be able to achieve the desired level of accuracy due to
  /// floating-point round-off.
  double acceptableTolerance = 1e-6;

  /// The solver will stop once the error is below `acceptableTolerance` for
  /// this many iterations.
  int maxAcceptableIterations = 15;

  /// The maximum elapsed wall clock time before returning a solution.
  std::chrono::duration<double> timeout{
      std::numeric_limits<double>::infinity()};

  /// The number of threads to use for parallel backtracking line searches.
  uint32_t numThreads = std::min(1u, std::thread::hardware_concurrency());

  /// Enables the feasible interior-point method. When the inequality
  /// constraints are all feasible, step sizes are reduced when necessary to
  /// prevent them becoming infeasible again. This is useful when parts of the
  /// problem are ill-conditioned in infeasible regions (e.g., square root of a
  /// negative value). This can slow or prevent progress toward a solution
  /// though, so only enable it if necessary.
  bool feasibleIPM = false;

  /// Enables diagnostic prints.
  ///
  /// <table>
  ///   <tr>
  ///     <th>Heading</th>
  ///     <th>Description</th>
  ///   </tr>
  ///   <tr>
  ///     <td>iter</td>
  ///     <td>Iteration number</td>
  ///   </tr>
  ///   <tr>
  ///     <td>type</td>
  ///     <td>Iteration type (normal, accepted second-order correction, rejected
  ///     second-order correction)</td>
  ///   </tr>
  ///   <tr>
  ///     <td>time (ms)</td>
  ///     <td>Duration of iteration in milliseconds</td>
  ///   </tr>
  ///   <tr>
  ///     <td>error</td>
  ///     <td>Error estimate</td>
  ///   </tr>
  ///   <tr>
  ///     <td>cost</td>
  ///     <td>Cost function value at current iterate</td>
  ///   </tr>
  ///   <tr>
  ///     <td>infeas.</td>
  ///     <td>Constraint infeasibility at current iterate</td>
  ///   </tr>
  ///   <tr>
  ///     <td>complement.</td>
  ///     <td>Complementary slackness at current iterate (sᵀz)</td>
  ///   </tr>
  ///   <tr>
  ///     <td>reg</td>
  ///     <td>Iteration matrix regularization</td>
  ///   </tr>
  ///   <tr>
  ///     <td>primal α</td>
  ///     <td>Primal step size</td>
  ///   </tr>
  ///   <tr>
  ///     <td>dual α</td>
  ///     <td>Dual step size</td>
  ///   </tr>
  ///   <tr>
  ///     <td>↩</td>
  ///     <td>Number of line search backtracks</td>
  ///   </tr>
  /// </table>
  bool diagnostics = false;

  /// Enables writing sparsity patterns of H, Aₑ, and Aᵢ to files named H.spy,
  /// A_e.spy, and A_i.spy respectively during solve.
  ///
  /// Use tools/spy.py to plot them.
  bool spy = false;
};

}  // namespace sleipnir
