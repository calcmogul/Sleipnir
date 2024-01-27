// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/InteriorPoint.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>

#include "optimization/RegularizedLDLT.hpp"
#include "optimization/solver/util/ErrorEstimate.hpp"
#include "optimization/solver/util/Filter.hpp"
#include "optimization/solver/util/FractionToTheBoundaryRule.hpp"
#include "optimization/solver/util/IsLocallyInfeasible.hpp"
#include "optimization/solver/util/KKTError.hpp"
#include "sleipnir/autodiff/Gradient.hpp"
#include "sleipnir/autodiff/Hessian.hpp"
#include "sleipnir/autodiff/Jacobian.hpp"
#include "sleipnir/optimization/SolverExitCondition.hpp"
#include "sleipnir/util/Print.hpp"
#include "sleipnir/util/Spy.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "util/PrintIterationDiagnostics.hpp"
#include "util/ScopeExit.hpp"
#include "util/ToMilliseconds.hpp"

// See docs/algorithms.md#Works_cited for citation definitions.
//
// See docs/algorithms.md#Interior-point_method for a derivation of the
// interior-point method formulation being used.

namespace sleipnir {

void InteriorPoint(std::span<Variable> decisionVariables,
                   std::span<Variable> equalityConstraints,
                   std::span<Variable> inequalityConstraints, Variable& f,
                   function_ref<bool(const SolverIterationInfo& info)> callback,
                   const SolverConfig& config, Eigen::VectorXd& x,
                   Eigen::VectorXd& s, SolverStatus* status) {
  const auto solveStartTime = std::chrono::steady_clock::now();

  // See table 1 of [1].
  constexpr double β_1 = 1e-4;
  // constexpr double β_2 = 0.01;
  // constexpr double β_3 = 0.02;
  // constexpr double β_4 = 0.2;
  // constexpr double β_5 = 0.03125;
  // constexpr double β_6 = 0.5;
  // constexpr double β_7 = 0.5;
  // constexpr double β_8 = 0.9;
  // constexpr double β_10 = 1e-4;
  // constexpr double β_11 = 1e-2;
  // constexpr double β_12 = 1e3;

  // Map decision variables and constraints to VariableMatrices for Lagrangian
  VariableMatrix xAD{decisionVariables};
  xAD.SetValue(x);
  VariableMatrix c_iAD{inequalityConstraints};

  // Create autodiff variables for s and y for Lagrangian
  VariableMatrix sAD(inequalityConstraints.size());
  sAD.SetValue(s);
  VariableMatrix yAD(inequalityConstraints.size());
  for (auto& y : yAD) {
    y.SetValue(0.0);
  }

  // Barrier parameter μ
  Variable μ = 0.1;

  // Lagrangian L
  //
  // L(xₖ, sₖ, yₖ) = f(xₖ) − (yₖ − μβ₁e)ᵀcᵢ(xₖ)
  auto L =
      f - ((yAD - μ * β_1 * VariableMatrix::Ones(s.rows(), 1)).T() * c_iAD)(0);

  // Inequality constraint Jacobian Aᵢ
  //
  //         [∇ᵀcᵢ₁(xₖ)]
  // Aᵢ(x) = [∇ᵀcᵢ₂(xₖ)]
  //         [    ⋮    ]
  //         [∇ᵀcᵢₘ(xₖ)]
  Jacobian jacobianCi{c_iAD, xAD};
  Eigen::SparseMatrix<double> A_i = jacobianCi.Value();

  // Gradient of L ∇ₓL
  Gradient gradientL{L, xAD};
  Eigen::SparseVector<double> g = gradientL.Value();

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(xₖ, sₖ, yₖ)
  Hessian hessianL{L, xAD};
  Eigen::SparseMatrix<double> H = hessianL.Value();

  Eigen::VectorXd y = yAD.Value();
  Eigen::VectorXd c_i = c_iAD.Value();

  double γ = 1.0;

  // Check for overconstrained problem
  if (equalityConstraints.size() > decisionVariables.size()) {
    if (config.diagnostics) {
      sleipnir::println("The problem has too few degrees of freedom.");
    }

    status->exitCondition = SolverExitCondition::kTooFewDOFs;
    return;
  }

  // Check whether initial guess has finite f(xₖ) and cᵢ(xₖ)
  if (!std::isfinite(f.Value()) || !c_i.allFinite()) {
    status->exitCondition =
        SolverExitCondition::kNonfiniteInitialCostOrConstraints;
    return;
  }

  // Sparsity pattern files written when spy flag is set in SolverConfig
  std::unique_ptr<Spy> H_spy;
  std::unique_ptr<Spy> A_i_spy;
  if (config.spy) {
    H_spy = std::make_unique<Spy>("H.spy", "Hessian", "Decision variables",
                                  "Decision variables", H.rows(), H.cols());
    A_i_spy = std::make_unique<Spy>("A_i.spy", "Inequality constraint Jacobian",
                                    "Constraints", "Decision variables",
                                    A_i.rows(), A_i.cols());
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
      sleipnir::println(format, "∇ₓL", gradientL.GetProfiler().SetupDuration(),
                        gradientL.GetProfiler().AverageSolveDuration(),
                        gradientL.GetProfiler().SolveMeasurements());
      sleipnir::println(format, "∇²ₓₓL", hessianL.GetProfiler().SetupDuration(),
                        hessianL.GetProfiler().AverageSolveDuration(),
                        hessianL.GetProfiler().SolveMeasurements());
      sleipnir::println(format, "∂cᵢ/∂x",
                        jacobianCi.GetProfiler().SetupDuration(),
                        jacobianCi.GetProfiler().AverageSolveDuration(),
                        jacobianCi.GetProfiler().SolveMeasurements());
      sleipnir::println("");
    }
  }};

  // Barrier parameter minimum
  const double μ_min = config.tolerance / 10.0;

  // Fraction-to-the-boundary rule scale factor minimum
  constexpr double τ_min = 0.99;

  // Fraction-to-the-boundary rule scale factor τ
  double τ = τ_min;

  Filter filter{f};

  // This should be run when the error estimate is below a desired threshold for
  // the current barrier parameter
  auto UpdateBarrierParameterAndResetFilter = [&] {
    // Barrier parameter linear decrease power in "κ_μ μ". Range of (0, 1).
    constexpr double κ_μ = 0.2;

    // Barrier parameter superlinear decrease power in "μ^(θ_μ)". Range of (1,
    // 2).
    constexpr double θ_μ = 1.5;

    // Update the barrier parameter.
    //
    //   μⱼ₊₁ = max(εₜₒₗ/10, min(κ_μ μⱼ, μⱼ^θ_μ))
    //
    // See equation (7) of [2].
    μ = std::max(μ_min, std::min(κ_μ * μ.Value(), std::pow(μ.Value(), θ_μ)));

    // Update the fraction-to-the-boundary rule scaling factor.
    //
    //   τⱼ = max(τₘᵢₙ, 1 − μⱼ)
    //
    // See equation (8) of [2].
    τ = std::max(τ_min, 1.0 - μ.Value());

    // Reset the filter when the barrier parameter is updated
    filter.Reset();
  };

  // Kept outside the loop so its storage can be reused
  small_vector<Eigen::Triplet<double>> triplets;

  RegularizedLDLT solver;

  // Variables for determining when a step is acceptable
  constexpr double α_red_factor = 0.5;
  int acceptableIterCounter = 0;

  int fullStepRejectedCounter = 0;
  int stepTooSmallCounter = 0;

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

    // Check for local inequality constraint infeasibility
    if (IsInequalityLocallyInfeasible(A_i, c_i)) {
      if (config.diagnostics) {
        sleipnir::println(
            "The problem is infeasible due to violated inequality "
            "constraints.");
        sleipnir::println(
            "Violated constraints (cᵢ(x) ≥ 0) in order of declaration:");
        for (int row = 0; row < c_i.rows(); ++row) {
          if (c_i(row) < 0.0) {
            sleipnir::println("  {}/{}: {} ≥ 0", row + 1, c_i.rows(), c_i(row));
          }
        }
      }

      status->exitCondition = SolverExitCondition::kLocallyInfeasible;
      return;
    }

    // Check for diverging iterates
    if (x.lpNorm<Eigen::Infinity>() > 1e20 || !x.allFinite() ||
        s.lpNorm<Eigen::Infinity>() > 1e20 || !s.allFinite()) {
      status->exitCondition = SolverExitCondition::kDivergingIterates;
      return;
    }

    // Write out spy file contents if that's enabled
    if (config.spy) {
      H_spy->Add(H);
      A_i_spy->Add(A_i);
    }

    // Call user callback
    if (callback({iterations, x, s, g, H, A_i})) {
      status->exitCondition = SolverExitCondition::kCallbackRequestedStop;
      return;
    }

    //     [s₁ 0 ⋯ 0 ]
    // S = [0  ⋱   ⋮ ]
    //     [⋮    ⋱ 0 ]
    //     [0  ⋯ 0 sₘ]
    Eigen::SparseMatrix<double> Sinv;
    Sinv = s.cwiseInverse().asDiagonal();

    //     [y₁ 0 ⋯ 0 ]
    // Y = [0  ⋱   ⋮ ]
    //     [⋮    ⋱ 0 ]
    //     [0  ⋯ 0 yₘ]
    const auto Y = y.asDiagonal();
    Eigen::SparseMatrix<double> Yinv;
    Yinv = y.cwiseInverse().asDiagonal();

    // M = H + AᵢᵀYS⁻¹Aᵢ
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    Eigen::SparseMatrix<double> M =
        H.triangularView<Eigen::Lower>() +
        (A_i.transpose() * Y * Sinv * A_i).triangularView<Eigen::Lower>();

    const Eigen::VectorXd e = Eigen::VectorXd::Ones(s.rows());
    const Eigen::VectorXd w = Eigen::VectorXd::Ones(s.rows());

    Eigen::VectorXd b_D = g;
    Eigen::VectorXd b_P = (1.0 - γ) * μ.Value() * w;
    Eigen::VectorXd b_C = Y * s - γ * μ.Value() * e;

    // Solve the Newton-KKT system
    solver.Compute(M);

    // rhs = −(b_D + AᵢᵀS⁻¹(Yb_P − b_C))
    Eigen::VectorXd rhs = -(b_D + A_i.transpose() * Sinv * (Y * b_P - b_C));

    Eigen::VectorXd p_x = solver.Solve(rhs);

    // pₖˢ = −(1 − γ)μw − Aᵢpₓ
    Eigen::VectorXd p_s = -(1.0 - γ) * μ.Value() * w - A_i * p_x;

    // pₖʸ = −S⁻¹Y(Aᵢpₖˣ + b_P - Y⁻¹b_C)
    Eigen::VectorXd p_y = -Sinv * Y * (A_i * p_x + b_P - Yinv * b_C);

    // αᵐᵃˣ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
    const double α_max = FractionToTheBoundaryRule(s, p_s, τ);
    double α_P = α_max;

    // α_D = max(α ∈ (0, 1] : yₖ + αpₖʸ ≥ (1−τⱼ)yₖ)
    double α_D = FractionToTheBoundaryRule(y, p_y, τ);

    // Loop until a step is accepted. If a step becomes acceptable, the loop
    // will exit early.
    while (1) {
      Eigen::VectorXd trial_x = x + α_P * p_x;
      Eigen::VectorXd trial_y = y + α_D * p_y;

      xAD.SetValue(trial_x);

      Eigen::VectorXd trial_c_i = c_iAD.Value();

      // If f(xₖ + αpₖˣ), cₑ(xₖ + αpₖˣ), or cᵢ(xₖ + αpₖˣ) aren't finite, reduce
      // step size immediately
      if (!std::isfinite(f.Value()) || !trial_c_i.allFinite()) {
        // Reduce step size
        α_P *= α_red_factor;
        continue;
      }

      Eigen::VectorXd trial_s;
      if (config.feasibleIPM && c_i.cwiseGreater(0.0).all()) {
        // If the inequality constraints are all feasible, prevent them from
        // becoming infeasible again.
        //
        // See equation (19.30) in [1].
        trial_s = trial_c_i;
      } else {
        trial_s = s + α_P * p_s;
      }

      // Check whether filter accepts trial iterate
      auto entry = filter.MakeEntry(trial_s, trial_c_i, μ.Value());
      if (filter.TryAdd(entry)) {
        // Accept step
        break;
      }

      // If we got here and α is the full step, the full step was rejected.
      // Increment the full-step rejected counter to keep track of how many full
      // steps have been rejected in a row.
      if (α_P == α_max) {
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
      α_P *= α_red_factor;

      // Safety factor for the minimal step size
      constexpr double α_min_frac = 0.05;

      // If step size hit a minimum, check if the KKT error was reduced. If it
      // wasn't, invoke feasibility restoration.
      if (α_P < α_min_frac * Filter::γConstraint) {
        double currentKKTError = KKTError(g, A_i, c_i, s, y, μ.Value());

        Eigen::VectorXd trial_x = x + α_max * p_x;
        Eigen::VectorXd trial_s = s + α_max * p_s;

        Eigen::VectorXd trial_y = y + α_D * p_y;

        // Upate autodiff
        xAD.SetValue(trial_x);
        sAD.SetValue(trial_s);
        yAD.SetValue(trial_y);

        Eigen::VectorXd trial_c_i = c_iAD.Value();

        double nextKKTError = KKTError(gradientL.Value(), jacobianCi.Value(),
                                       trial_c_i, trial_s, trial_y, μ.Value());

        // If the step using αᵐᵃˣ reduced the KKT error, accept it anyway
        if (nextKKTError <= 0.999 * currentKKTError) {
          α_P = α_max;

          // Accept step
          break;
        }
      }
    }

    // If full step was accepted, reset full-step rejected counter
    if (α_P == α_max) {
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
      α_P = α_max;
      ++stepTooSmallCounter;
    } else {
      stepTooSmallCounter = 0;
    }

    // μₖ₊₁ = (1 − (1 − γ)α_P)μₖ
    μ.SetValue((1.0 - (1.0 - γ) * α_P) * μ.Value());

    // xₖ₊₁ = xₖ + α_Pₖ pₖˣ
    x += α_P * p_x;

    // Update cᵢ
    c_i = c_iAD.Value();

    // sₖ₊₁ = μₖ₊₁w − cᵢ(xₖ₊₁)
    s = μ.Value() * w - c_i;

    // yₖ₊₁ = yₖ + α_Dₖ pₖʸ
    y += α_D * p_y;

    // Update autodiff for Jacobians and Hessian
    xAD.SetValue(x);
    sAD.SetValue(s);
    yAD.SetValue(y);
    A_i = jacobianCi.Value();
    g = gradientL.Value();
    H = hessianL.Value();

    // Update the error estimate
    E_0 = ErrorEstimate(g, A_i, c_i, s, y, 0.0);
    if (E_0 < config.acceptableTolerance) {
      ++acceptableIterCounter;
    } else {
      acceptableIterCounter = 0;
    }

    // Update the barrier parameter if necessary
    if (E_0 > config.tolerance) {
      // Barrier parameter scale factor for tolerance checks
      constexpr double κ_ε = 10.0;

      // While the error estimate is below the desired threshold for this
      // barrier parameter value, decrease the barrier parameter further
      double E_μ = ErrorEstimate(g, A_i, c_i, s, y, μ.Value());
      while (μ.Value() > μ_min && E_μ <= κ_ε * μ.Value()) {
        UpdateBarrierParameterAndResetFilter();
        E_μ = ErrorEstimate(g, A_i, c_i, s, y, μ.Value());
      }
    }

    const auto innerIterEndTime = std::chrono::steady_clock::now();

    if (config.diagnostics) {
      PrintIterationDiagnostics(
          iterations, innerIterEndTime - innerIterStartTime, E_0, f.Value(),
          (c_i - s).lpNorm<1>(), solver.HessianRegularization(), α_P);
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

    // The search direction has been very small twice, so assume the problem has
    // been solved as well as possible given finite precision and reduce the
    // barrier parameter.
    //
    // See section 3.9 of [2].
    if (stepTooSmallCounter >= 2 && μ.Value() > μ_min) {
      UpdateBarrierParameterAndResetFilter();
      continue;
    }
  }
}

}  // namespace sleipnir
