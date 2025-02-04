// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/SQP.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <ranges>

#include <Eigen/SparseCholesky>

#include "optimization/RegularizedLDLT.hpp"
#include "optimization/solver/util/ErrorEstimate.hpp"
#include "optimization/solver/util/Filter.hpp"
#include "optimization/solver/util/IsLocallyInfeasible.hpp"
#include "optimization/solver/util/KKTError.hpp"
#include "sleipnir/autodiff/Gradient.hpp"
#include "sleipnir/autodiff/Hessian.hpp"
#include "sleipnir/autodiff/Jacobian.hpp"
#include "sleipnir/optimization/SolverExitCondition.hpp"
#include "sleipnir/util/ScopedProfiler.hpp"
#include "sleipnir/util/SetupProfiler.hpp"
#include "sleipnir/util/SolveProfiler.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "util/ScopeExit.hpp"

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
#include "sleipnir/util/Print.hpp"
#include "sleipnir/util/Spy.hpp"
#include "util/PrintDiagnostics.hpp"
#endif

// See docs/algorithms.md#Works_cited for citation definitions.

namespace sleipnir {

void SQP(
    std::span<Variable> decisionVariables,
    std::span<Variable> equalityConstraints, Variable& f,
    std::span<std::function<bool(const SolverIterationInfo& info)>> callbacks,
    const SolverConfig& config, Eigen::VectorXd& x, SolverStatus* status) {
  const auto solveStartTime = std::chrono::steady_clock::now();

  small_vector<SetupProfiler> setupProfilers;
  setupProfilers.emplace_back("setup").Start();

  setupProfilers.emplace_back("  ↳ y setup").Start();

  // Map decision variables and constraints to VariableMatrices for Lagrangian
  VariableMatrix xAD{decisionVariables};
  xAD.SetValue(x);
  VariableMatrix c_eAD{equalityConstraints};

  // Create autodiff variables for y for Lagrangian
  VariableMatrix yAD(equalityConstraints.size());
  for (auto& y : yAD) {
    y.SetValue(0.0);
  }

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ L setup").Start();

  // Lagrangian L
  //
  // L(xₖ, yₖ) = f(xₖ) − yₖᵀcₑ(xₖ)
  auto L = f - (yAD.T() * c_eAD)(0);

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∂cₑ/∂x setup").Start();

  // Equality constraint Jacobian Aₑ
  //
  //         [∇ᵀcₑ₁(xₖ)]
  // Aₑ(x) = [∇ᵀcₑ₂(xₖ)]
  //         [    ⋮    ]
  //         [∇ᵀcₑₘ(xₖ)]
  Jacobian jacobianCe{c_eAD, xAD};

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∂cₑ/∂x init solve").Start();

  Eigen::SparseMatrix<double> A_e = jacobianCe.Value();

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∇f(x) setup").Start();

  // Gradient of f ∇f
  Gradient gradientF{f, xAD};

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∇f(x) init solve").Start();

  Eigen::SparseVector<double> g = gradientF.Value();

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∇²ₓₓL setup").Start();

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(xₖ, yₖ)
  Hessian hessianL{L, xAD, true};

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∇²ₓₓL init solve").Start();

  Eigen::SparseMatrix<double> H = hessianL.Value();

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ precondition ✓").Start();

  Eigen::VectorXd y = yAD.Value();
  Eigen::VectorXd c_e = c_eAD.Value();

  // Check for overconstrained problem
  if (equalityConstraints.size() > decisionVariables.size()) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (config.diagnostics) {
      sleipnir::println("The problem has too few degrees of freedom.");
      sleipnir::println(
          "Violated constraints (cₑ(x) = 0) in order of declaration:");
      for (int row = 0; row < c_e.rows(); ++row) {
        if (c_e(row) < 0.0) {
          sleipnir::println("  {}/{}: {} = 0", row + 1, c_e.rows(), c_e(row));
        }
      }
    }
#endif

    status->exitCondition = SolverExitCondition::kTooFewDOFs;
    return;
  }

  // Check whether initial guess has finite f(xₖ) and cₑ(xₖ)
  if (!std::isfinite(f.Value()) || !c_e.allFinite()) {
    status->exitCondition =
        SolverExitCondition::kNonfiniteInitialCostOrConstraints;
    return;
  }

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ spy setup").Start();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  // Sparsity pattern files written when spy flag is set in SolverConfig
  std::unique_ptr<Spy> H_spy;
  std::unique_ptr<Spy> A_e_spy;
  std::unique_ptr<Spy> lhs_spy;
  if (config.spy) {
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

  setupProfilers.back().Stop();

  int iterations = 0;

  Filter filter{f};

  // Kept outside the loop so its storage can be reused
  small_vector<Eigen::Triplet<double>> triplets;

  RegularizedLDLT solver;

  // Variables for determining when a step is acceptable
  constexpr double α_red_factor = 0.5;
  constexpr double α_min = 1e-20;
  int acceptableIterCounter = 0;

  int fullStepRejectedCounter = 0;

  // Error estimate
  double E_0 = std::numeric_limits<double>::infinity();

  setupProfilers[0].Stop();

  small_vector<SolveProfiler> solveProfilers;
  solveProfilers.emplace_back("solve");
  solveProfilers.emplace_back("  ↳ feasibility ✓");
  solveProfilers.emplace_back("  ↳ user callbacks");
  solveProfilers.emplace_back("  ↳ iter matrix build");
  solveProfilers.emplace_back("  ↳ iter matrix solve");
  solveProfilers.emplace_back("  ↳ line search");
  solveProfilers.emplace_back("    ↳ SOC");
  solveProfilers.emplace_back("  ↳ spy writes");
  solveProfilers.emplace_back("  ↳ next iter prep");

  auto& innerIterProf = solveProfilers[0];
  auto& feasibilityCheckProf = solveProfilers[1];
  auto& userCallbacksProf = solveProfilers[2];
  auto& linearSystemBuildProf = solveProfilers[3];
  auto& linearSystemSolveProf = solveProfilers[4];
  auto& lineSearchProf = solveProfilers[5];
  auto& socProf = solveProfilers[6];
  [[maybe_unused]]
  auto& spyWritesProf = solveProfilers[7];
  auto& nextIterPrepProf = solveProfilers[8];

  // Prints final diagnostics when the solver exits
  scope_exit exit{[&] {
    status->cost = f.Value();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (config.diagnostics) {
      // Append gradient profilers
      solveProfilers.push_back(gradientF.GetProfilers()[0]);
      solveProfilers.back().name = "  ↳ ∇f(x)";
      for (const auto& profiler :
           gradientF.GetProfilers() | std::views::drop(1)) {
        solveProfilers.push_back(profiler);
      }

      // Append Hessian profilers
      solveProfilers.push_back(hessianL.GetProfilers()[0]);
      solveProfilers.back().name = "  ↳ ∇²ₓₓL";
      for (const auto& profiler :
           hessianL.GetProfilers() | std::views::drop(1)) {
        solveProfilers.push_back(profiler);
      }

      // Append equality constraint Jacobian profilers
      solveProfilers.push_back(jacobianCe.GetProfilers()[0]);
      solveProfilers.back().name = "  ↳ ∂cₑ/∂x";
      for (const auto& profiler :
           jacobianCe.GetProfilers() | std::views::drop(1)) {
        solveProfilers.push_back(profiler);
      }

      PrintFinalDiagnostics(iterations, setupProfilers, solveProfilers);
    }
#endif
  }};

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  if (config.diagnostics) {
    sleipnir::println("Error tolerance: {}\n", config.tolerance);
  }
#endif

  while (E_0 > config.tolerance &&
         acceptableIterCounter < config.maxAcceptableIterations) {
    ScopedProfiler innerIterProfiler{innerIterProf};
    ScopedProfiler feasibilityCheckProfiler{feasibilityCheckProf};

    // Check for local equality constraint infeasibility
    if (IsEqualityLocallyInfeasible(A_e, c_e)) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
      if (config.diagnostics) {
        sleipnir::println(
            "The problem is locally infeasible due to violated equality "
            "constraints.");
        sleipnir::println(
            "Violated constraints (cₑ(x) = 0) in order of declaration:");
        for (int row = 0; row < c_e.rows(); ++row) {
          if (c_e(row) < 0.0) {
            sleipnir::println("  {}/{}: {} = 0", row + 1, c_e.rows(), c_e(row));
          }
        }
      }
#endif

      status->exitCondition = SolverExitCondition::kLocallyInfeasible;
      return;
    }

    // Check for diverging iterates
    if (x.lpNorm<Eigen::Infinity>() > 1e20 || !x.allFinite()) {
      status->exitCondition = SolverExitCondition::kDivergingIterates;
      return;
    }

    feasibilityCheckProfiler.Stop();
    ScopedProfiler userCallbacksProfiler{userCallbacksProf};

    // Call user callbacks
    for (const auto& callback : callbacks) {
      if (callback({iterations, x, Eigen::VectorXd::Zero(0), g, H, A_e,
                    Eigen::SparseMatrix<double>{}})) {
        status->exitCondition = SolverExitCondition::kCallbackRequestedStop;
        return;
      }
    }

    userCallbacksProfiler.Stop();
    ScopedProfiler linearSystemBuildProfiler{linearSystemBuildProf};

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
        decisionVariables.size() + equalityConstraints.size(),
        decisionVariables.size() + equalityConstraints.size());
    lhs.setFromSortedTriplets(triplets.begin(), triplets.end(),
                              [](const auto&, const auto& b) { return b; });

    // rhs = −[∇f − Aₑᵀy]
    //        [   cₑ    ]
    Eigen::VectorXd rhs{x.rows() + y.rows()};
    rhs.segment(0, x.rows()) = -g + A_e.transpose() * y;
    rhs.segment(x.rows(), y.rows()) = -c_e;

    linearSystemBuildProfiler.Stop();
    ScopedProfiler linearSystemSolveProfiler{linearSystemSolveProf};

    Eigen::VectorXd p_x;
    Eigen::VectorXd p_y;
    constexpr double α_max = 1.0;
    double α = 1.0;

    // Solve the Newton-KKT system
    //
    // [H   Aₑᵀ][ pₖˣ] = −[∇f − Aₑᵀy]
    // [Aₑ   0 ][−pₖʸ]    [   cₑ    ]
    solver.Compute(lhs, equalityConstraints.size(), config.tolerance / 10.0);
    if (solver.Info() != Eigen::Success) [[unlikely]] {
      status->exitCondition = SolverExitCondition::kFactorizationFailed;
      return;
    }

    Eigen::VectorXd step = solver.Solve(rhs);

    linearSystemSolveProfiler.Stop();
    ScopedProfiler lineSearchProfiler{lineSearchProf};

    // step = [ pₖˣ]
    //        [−pₖʸ]
    p_x = step.segment(0, x.rows());
    p_y = -step.segment(x.rows(), y.rows());

    α = α_max;

    // Loop until a step is accepted
    while (1) {
      Eigen::VectorXd trial_x = x + α * p_x;
      Eigen::VectorXd trial_y = y + α * p_y;

      xAD.SetValue(trial_x);

      Eigen::VectorXd trial_c_e = c_eAD.Value();

      // If f(xₖ + αpₖˣ) or cₑ(xₖ + αpₖˣ) aren't finite, reduce step size
      // immediately
      if (!std::isfinite(f.Value()) || !trial_c_e.allFinite()) {
        // Reduce step size
        α *= α_red_factor;
        continue;
      }

      // Check whether filter accepts trial iterate
      auto entry = filter.MakeEntry(trial_c_e);
      if (filter.TryAdd(entry, α)) {
        // Accept step
        break;
      }

      double prevConstraintViolation = c_e.lpNorm<1>();
      double nextConstraintViolation = trial_c_e.lpNorm<1>();

      // Second-order corrections
      //
      // If first trial point was rejected and constraint violation stayed the
      // same or went up, apply second-order corrections
      if (α == α_max && nextConstraintViolation >= prevConstraintViolation) {
        // Apply second-order corrections. See section 2.4 of [2].
        Eigen::VectorXd p_x_cor = p_x;
        Eigen::VectorXd p_y_soc = p_y;

        double α_soc = α;
        Eigen::VectorXd c_e_soc = c_e;

        bool stepAcceptable = false;
        for (int soc_iteration = 0; soc_iteration < 5 && !stepAcceptable;
             ++soc_iteration) {
          ScopedProfiler socProfiler{socProf};

          scope_exit soc_exit{[&] {
            socProfiler.Stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
            if (config.diagnostics) {
              double E = ErrorEstimate(g, A_e, trial_c_e, trial_y);
              PrintIterationDiagnostics(
                  iterations,
                  stepAcceptable ? IterationType::kAcceptedSOC
                                 : IterationType::kRejectedSOC,
                  socProfiler.CurrentDuration(), E, f.Value(),
                  trial_c_e.lpNorm<1>(), 0.0, solver.HessianRegularization(),
                  α_soc, 1.0);
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
          step = solver.Solve(rhs);

          p_x_cor = step.segment(0, x.rows());
          p_y_soc = -step.segment(x.rows(), y.rows());

          trial_x = x + α_soc * p_x_cor;
          trial_y = y + α_soc * p_y_soc;

          xAD.SetValue(trial_x);

          trial_c_e = c_eAD.Value();

          // Constraint violation scale factor for second-order corrections
          constexpr double κ_soc = 0.99;

          // If constraint violation hasn't been sufficiently reduced, stop
          // making second-order corrections
          nextConstraintViolation = trial_c_e.lpNorm<1>();
          if (nextConstraintViolation > κ_soc * prevConstraintViolation) {
            break;
          }

          // Check whether filter accepts trial iterate
          entry = filter.MakeEntry(trial_c_e);
          if (filter.TryAdd(entry, α)) {
            p_x = p_x_cor;
            p_y = p_y_soc;
            α = α_soc;
            stepAcceptable = true;
          }
        }

        if (stepAcceptable) {
          // Accept step
          break;
        }
      }

      // If we got here and α is the full step, the full step was rejected.
      // Increment the full-step rejected counter to keep track of how many full
      // steps have been rejected in a row.
      if (α == α_max) {
        ++fullStepRejectedCounter;
      }

      // If the full step was rejected enough times in a row, reset the filter
      // because it may be impeding progress.
      //
      // See section 3.2 case I of [2].
      if (fullStepRejectedCounter >= 4 &&
          filter.maxConstraintViolation > entry.constraintViolation / 10.0) {
        filter.maxConstraintViolation *= 0.1;
        filter.Reset();
        continue;
      }

      // Reduce step size
      α *= α_red_factor;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, report bad line search.
      if (α < α_min) {
        double currentKKTError = KKTError(g, A_e, c_e, y);

        trial_x = x + α_max * p_x;
        trial_y = y + α_max * p_y;

        // Upate autodiff
        xAD.SetValue(trial_x);
        yAD.SetValue(trial_y);

        trial_c_e = c_eAD.Value();

        double nextKKTError =
            KKTError(gradientF.Value(), jacobianCe.Value(), trial_c_e, trial_y);

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (nextKKTError <= 0.999 * currentKKTError) {
          α = α_max;

          // Accept step
          break;
        }

        status->exitCondition = SolverExitCondition::kLineSearchFailed;
        return;
      }
    }

    lineSearchProfiler.Stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    // Write out spy file contents if that's enabled
    if (config.spy) {
      ScopedProfiler spyWritesProfiler{spyWritesProf};
      H_spy->Add(H);
      A_e_spy->Add(A_e);
      lhs_spy->Add(lhs);
    }
#endif

    // If full step was accepted, reset full-step rejected counter
    if (α == α_max) {
      fullStepRejectedCounter = 0;
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
    // yₖ₊₁ = yₖ + αₖpₖʸ
    x += α * p_x;
    y += α * p_y;

    // Update autodiff for Jacobians and Hessian
    xAD.SetValue(x);
    yAD.SetValue(y);
    A_e = jacobianCe.Value();
    g = gradientF.Value();
    H = hessianL.Value();

    ScopedProfiler nextIterPrepProfiler{nextIterPrepProf};

    c_e = c_eAD.Value();

    // Update the error estimate
    E_0 = ErrorEstimate(g, A_e, c_e, y);
    if (E_0 < config.acceptableTolerance) {
      ++acceptableIterCounter;
    } else {
      acceptableIterCounter = 0;
    }

    nextIterPrepProfiler.Stop();
    innerIterProfiler.Stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (config.diagnostics) {
      PrintIterationDiagnostics(iterations, IterationType::kNormal,
                                innerIterProfiler.CurrentDuration(), E_0,
                                f.Value(), c_e.lpNorm<1>(), 0.0,
                                solver.HessianRegularization(), α, α);
    }
#endif

    ++iterations;

    // Check for max iterations
    if (iterations >= config.maxIterations) {
      status->exitCondition = SolverExitCondition::kMaxIterationsExceeded;
      return;
    }

    // Check for max wall clock time
    if (std::chrono::steady_clock::now() - solveStartTime > config.timeout) {
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
