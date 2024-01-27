// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/Newton.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>

#include <Eigen/SparseCholesky>

#include "optimization/RegularizedLDLT.hpp"
#include "optimization/solver/util/ErrorEstimate.hpp"
#include "optimization/solver/util/Filter.hpp"
#include "optimization/solver/util/KKTError.hpp"
#include "sleipnir/autodiff/Gradient.hpp"
#include "sleipnir/autodiff/Hessian.hpp"
#include "sleipnir/optimization/SolverExitCondition.hpp"
#include "sleipnir/util/Print.hpp"
#include "sleipnir/util/Spy.hpp"
#include "util/ScopeExit.hpp"
#include "util/ToMilliseconds.hpp"

// See docs/algorithms.md#Works_cited for citation definitions.

namespace sleipnir {

void Newton(std::span<Variable> decisionVariables, Variable& f,
            function_ref<bool(const SolverIterationInfo& info)> callback,
            const SolverConfig& config, Eigen::VectorXd& x,
            SolverStatus* status) {
  const auto solveStartTime = std::chrono::steady_clock::now();

  // Map decision variables and constraints to VariableMatrices for Lagrangian
  VariableMatrix xAD{decisionVariables};
  xAD.SetValue(x);

  // Lagrangian L
  //
  // L(xₖ, yₖ) = f(xₖ)
  auto L = f;

  // Gradient of f ∇f
  Gradient gradientF{f, xAD};
  Eigen::SparseVector<double> g = gradientF.Value();

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(xₖ, yₖ)
  Hessian hessianL{L, xAD};
  Eigen::SparseMatrix<double> H = hessianL.Value();

  // Check whether initial guess has finite f(xₖ)
  if (!std::isfinite(f.Value())) {
    status->exitCondition =
        SolverExitCondition::kNonfiniteInitialCostOrConstraints;
    return;
  }

  // Sparsity pattern files written when spy flag is set in SolverConfig
  std::ofstream H_spy;
  if (config.spy) {
    H_spy.open("H.spy");
  }

  if (config.diagnostics) {
    sleipnir::println("Error tolerance: {}\n", config.tolerance);
  }

  std::chrono::steady_clock::time_point iterationsStartTime;

  int iterations = 0;

  // Prints final diagnostics when the solver exits
  scope_exit exit{[&] {
    status->cost = f.Value();

    if (config.diagnostics) {
      auto solveEndTime = std::chrono::steady_clock::now();

      sleipnir::println("\nSolve time: {:.3f} ms",
                        ToMilliseconds(solveEndTime - solveStartTime));
      sleipnir::println("  ↳ {:.3f} ms (solver setup)",
                        ToMilliseconds(iterationsStartTime - solveStartTime));
      if (iterations > 0) {
        sleipnir::println(
            "  ↳ {:.3f} ms ({} solver iterations; {:.3f} ms average)",
            ToMilliseconds(solveEndTime - iterationsStartTime), iterations,
            ToMilliseconds((solveEndTime - iterationsStartTime) / iterations));
      }
      sleipnir::println("");

      sleipnir::println("{:^8}   {:^10}   {:^14}   {:^6}", "autodiff",
                        "setup (ms)", "avg solve (ms)", "solves");
      sleipnir::println("{:=^47}", "");
      constexpr auto format = "{:^8}   {:10.3f}   {:14.3f}   {:6}";
      sleipnir::println(format, "∇f(x)",
                        gradientF.GetProfiler().SetupDuration(),
                        gradientF.GetProfiler().AverageSolveDuration(),
                        gradientF.GetProfiler().SolveMeasurements());
      sleipnir::println(format, "∇²ₓₓL", hessianL.GetProfiler().SetupDuration(),
                        hessianL.GetProfiler().AverageSolveDuration(),
                        hessianL.GetProfiler().SolveMeasurements());
      sleipnir::println("");
    }
  }};

  Filter filter{f};

  RegularizedLDLT solver;

  // Variables for determining when a step is acceptable
  constexpr double α_red_factor = 0.5;
  int acceptableIterCounter = 0;

  // Error estimate
  double E_0 = std::numeric_limits<double>::infinity();

  if (config.diagnostics) {
    iterationsStartTime = std::chrono::steady_clock::now();
  }

  while (E_0 > config.tolerance &&
         acceptableIterCounter < config.maxAcceptableIterations) {
    std::chrono::steady_clock::time_point innerIterStartTime;
    if (config.diagnostics) {
      innerIterStartTime = std::chrono::steady_clock::now();
    }

    // Check for diverging iterates
    if (x.lpNorm<Eigen::Infinity>() > 1e20 || !x.allFinite()) {
      status->exitCondition = SolverExitCondition::kDivergingIterates;
      return;
    }

    // Write out spy file contents if that's enabled
    if (config.spy) {
      // Gap between sparsity patterns
      if (iterations > 0) {
        H_spy << "\n";
      }

      Spy(H_spy, H);
    }

    // Call user callback
    if (callback({iterations, x, Eigen::VectorXd::Zero(0), g, H,
                  Eigen::SparseMatrix<double>{}})) {
      status->exitCondition = SolverExitCondition::kCallbackRequestedStop;
      return;
    }

    // rhs = −[∇f]
    Eigen::VectorXd rhs = -g;

    // Solve the Newton-KKT system
    //
    // [H][ pₖˣ] = −[∇f]
    solver.Compute(H);
    Eigen::VectorXd step = solver.Solve(rhs);

    // step = [ pₖˣ]
    Eigen::VectorXd p_x = step.segment(0, x.rows());

    constexpr double α_max = 1.0;
    double α = α_max;

    // Loop until a step is accepted. If a step becomes acceptable, the loop
    // will exit early.
    while (1) {
      Eigen::VectorXd trial_x = x + α * p_x;

      xAD.SetValue(trial_x);

      // If f(xₖ + αpₖˣ) isn't finite, reduce step size immediately
      if (!std::isfinite(f.Value())) {
        // Reduce step size
        α *= α_red_factor;
        continue;
      }

      // Check whether filter accepts trial iterate
      auto entry = filter.MakeEntry();
      if (filter.TryAdd(entry)) {
        // Accept step
        break;
      }

      // Reduce step size
      α *= α_red_factor;

      // Safety factor for the minimal step size
      constexpr double α_min_frac = 0.05;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, report infeasible.
      if (α < α_min_frac * Filter::γConstraint) {
        double currentKKTError = KKTError(g);

        Eigen::VectorXd trial_x = x + α_max * p_x;

        // Upate autodiff
        xAD.SetValue(trial_x);

        double nextKKTError = KKTError(gradientF.Value());

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (nextKKTError <= 0.999 * currentKKTError) {
          α = α_max;

          // Accept step
          break;
        }

        status->exitCondition = SolverExitCondition::kLocallyInfeasible;
        return;
      }
    }

    // Handle very small search directions by letting αₖ = αₖᵐᵃˣ when
    // max(|pₖˣ(i)|/(1 + |xₖ(i)|)) < 10ε_mach.
    //
    // See section 3.9 of [2].
    double maxStepScaled = 0.0;
    for (int row = 0; row < x.rows(); ++row) {
      maxStepScaled = std::max(maxStepScaled,
                               std::abs(p_x(row)) / (1.0 + std::abs(x(row))));
    }
    if (maxStepScaled < 10.0 * std::numeric_limits<double>::epsilon()) {
      α = α_max;
    }

    // xₖ₊₁ = xₖ + αₖpₖˣ
    x += α * p_x;

    // Update autodiff for Hessian
    xAD.SetValue(x);
    g = gradientF.Value();
    H = hessianL.Value();

    // Update the error estimate
    E_0 = ErrorEstimate(g);
    if (E_0 < config.acceptableTolerance) {
      ++acceptableIterCounter;
    } else {
      acceptableIterCounter = 0;
    }

    const auto innerIterEndTime = std::chrono::steady_clock::now();

    // Diagnostics for current iteration
    if (config.diagnostics) {
      if (iterations % 20 == 0) {
        sleipnir::println("{:^4}   {:^9}  {:^13}  {:^13}  {:^13}", "iter",
                          "time (ms)", "error", "cost", "infeasibility");
        sleipnir::println("{:=^61}", "");
      }

      sleipnir::println("{:4}   {:9.3f}  {:13e}  {:13e}  {:13e}", iterations,
                        ToMilliseconds(innerIterEndTime - innerIterStartTime),
                        E_0, f.Value(), 0.0);
    }

    ++iterations;

    // Check for max iterations
    if (iterations >= config.maxIterations) {
      status->exitCondition = SolverExitCondition::kMaxIterationsExceeded;
      return;
    }

    // Check for max wall clock time
    if (innerIterEndTime - solveStartTime > config.timeout) {
      status->exitCondition = SolverExitCondition::kTimeout;
      return;
    }

    // Check for solve to acceptable tolerance
    if (E_0 > config.tolerance &&
        acceptableIterCounter == config.maxAcceptableIterations) {
      status->exitCondition = SolverExitCondition::kSolvedToAcceptableTolerance;
      return;
    }
  }
}

}  // namespace sleipnir
