// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/InteriorPoint.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <ranges>

#include <Eigen/SparseCholesky>
#include <wpi/SmallVector.h>

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
#include "sleipnir/util/ScopedProfiler.hpp"
#include "sleipnir/util/SetupProfiler.hpp"
#include "sleipnir/util/SolveProfiler.hpp"
#include "util/ScopeExit.hpp"

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
#include "sleipnir/util/Print.hpp"
#include "sleipnir/util/Spy.hpp"
#include "util/PrintDiagnostics.hpp"
#endif

// See docs/algorithms.md#Works_cited for citation definitions.
//
// See docs/algorithms.md#Interior-point_method for a derivation of the
// interior-point method formulation being used.

namespace sleipnir {

void InteriorPoint(
    std::span<Variable> decisionVariables,
    std::span<Variable> equalityConstraints,
    std::span<Variable> inequalityConstraints, Variable& f,
    std::span<std::function<bool(const SolverIterationInfo& info)>> callbacks,
    const SolverConfig& config, Eigen::VectorXd& x, SolverStatus* status) {
  const auto solveStartTime = std::chrono::steady_clock::now();

  wpi::SmallVector<SetupProfiler> setupProfilers;
  setupProfilers.emplace_back("setup").Start();

  setupProfilers.emplace_back("  ↳ s,y,z setup").Start();

  // Map decision variables and constraints to VariableMatrices for Lagrangian
  VariableMatrix xAD{decisionVariables};
  xAD.SetValue(x);
  VariableMatrix c_eAD{equalityConstraints};
  VariableMatrix c_iAD{inequalityConstraints};

  // Create autodiff variables for s, y, and z for Lagrangian
  VariableMatrix sAD(inequalityConstraints.size());
  for (auto& s : sAD) {
    s.SetValue(1.0);
  }
  VariableMatrix yAD(equalityConstraints.size());
  for (auto& y : yAD) {
    y.SetValue(0.0);
  }
  VariableMatrix zAD(inequalityConstraints.size());
  for (auto& z : zAD) {
    z.SetValue(1.0);
  }

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ L setup").Start();

  // Lagrangian L
  //
  // L(xₖ, sₖ, yₖ, zₖ) = f(xₖ) − yₖᵀcₑ(xₖ) − zₖᵀ(cᵢ(xₖ) − sₖ)
  auto L = f - (yAD.T() * c_eAD)(0) - (zAD.T() * (c_iAD - sAD))(0);

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
  setupProfilers.emplace_back("  ↳ ∂cᵢ/∂x setup").Start();

  // Inequality constraint Jacobian Aᵢ
  //
  //         [∇ᵀcᵢ₁(xₖ)]
  // Aᵢ(x) = [∇ᵀcᵢ₂(xₖ)]
  //         [    ⋮    ]
  //         [∇ᵀcᵢₘ(xₖ)]
  Jacobian jacobianCi{c_iAD, xAD};

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∂cᵢ/∂x init solve").Start();

  Eigen::SparseMatrix<double> A_i = jacobianCi.Value();

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
  // Hₖ = ∇²ₓₓL(xₖ, sₖ, yₖ, zₖ)
  Hessian hessianL{L, xAD};

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ ∇²ₓₓL init solve").Start();

  Eigen::SparseMatrix<double> H = hessianL.Value();

  setupProfilers.back().Stop();
  setupProfilers.emplace_back("  ↳ precondition ✓").Start();

  Eigen::VectorXd s = sAD.Value();
  Eigen::VectorXd y = yAD.Value();
  Eigen::VectorXd z = zAD.Value();
  Eigen::VectorXd c_e = c_eAD.Value();
  Eigen::VectorXd c_i = c_iAD.Value();

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

  // Check whether initial guess has finite f(xₖ), cₑ(xₖ), and cᵢ(xₖ)
  if (!std::isfinite(f.Value()) || !c_e.allFinite() || !c_i.allFinite()) {
    status->exitCondition =
        SolverExitCondition::kNonfiniteInitialCostOrConstraints;
    return;
  }

  setupProfilers.back().Stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
  // Sparsity pattern files written when spy flag is set in SolverConfig
  std::unique_ptr<Spy> H_spy;
  std::unique_ptr<Spy> A_e_spy;
  std::unique_ptr<Spy> A_i_spy;
  std::unique_ptr<Spy> lhs_spy;
  if (config.spy) {
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

  // Barrier parameter minimum
  const double μ_min = config.tolerance / 10.0;

  // Barrier parameter μ
  double μ = 0.1;

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
    μ = std::max(μ_min, std::min(κ_μ * μ, std::pow(μ, θ_μ)));

    // Update the fraction-to-the-boundary rule scaling factor.
    //
    //   τⱼ = max(τₘᵢₙ, 1 − μⱼ)
    //
    // See equation (8) of [2].
    τ = std::max(τ_min, 1.0 - μ);

    // Reset the filter when the barrier parameter is updated
    filter.Reset();
  };

  // Kept outside the loop so its storage can be reused
  wpi::SmallVector<Eigen::Triplet<double>> triplets;

  RegularizedLDLT solver;

  // Variables for determining when a step is acceptable
  constexpr double α_red_factor = 0.5;
  constexpr double α_min = 1e-20;
  int acceptableIterCounter = 0;

  int fullStepRejectedCounter = 0;
  int stepTooSmallCounter = 0;

  // Error estimate
  double E_0 = std::numeric_limits<double>::infinity();

  setupProfilers[0].Stop();

  wpi::SmallVector<SolveProfiler> solveProfilers;
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

      // Append inequality constraint Jacobian profilers
      solveProfilers.push_back(jacobianCi.GetProfilers()[0]);
      solveProfilers.back().name = "  ↳ ∂cᵢ/∂x";
      for (const auto& profiler :
           jacobianCi.GetProfilers() | std::views::drop(1)) {
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

    // Check for local inequality constraint infeasibility
    if (IsInequalityLocallyInfeasible(A_i, c_i)) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
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
#endif

      status->exitCondition = SolverExitCondition::kLocallyInfeasible;
      return;
    }

    // Check for diverging iterates
    if (x.lpNorm<Eigen::Infinity>() > 1e20 || !x.allFinite() ||
        s.lpNorm<Eigen::Infinity>() > 1e20 || !s.allFinite()) {
      status->exitCondition = SolverExitCondition::kDivergingIterates;
      return;
    }

    feasibilityCheckProfiler.Stop();
    ScopedProfiler userCallbacksProfiler{userCallbacksProf};

    // Call user callbacks
    for (const auto& callback : callbacks) {
      if (callback({iterations, x, s, g, H, A_e, A_i})) {
        status->exitCondition = SolverExitCondition::kCallbackRequestedStop;
        return;
      }
    }

    userCallbacksProfiler.Stop();
    ScopedProfiler linearSystemBuildProfiler{linearSystemBuildProf};

    //     [s₁ 0 ⋯ 0 ]
    // S = [0  ⋱   ⋮ ]
    //     [⋮    ⋱ 0 ]
    //     [0  ⋯ 0 sₘ]
    //
    //     [z₁ 0 ⋯ 0 ]
    // Z = [0  ⋱   ⋮ ]
    //     [⋮    ⋱ 0 ]
    //     [0  ⋯ 0 zₘ]
    //
    // Σ = S⁻¹Z
    Eigen::SparseMatrix<double> Sinv;
    Sinv = s.cwiseInverse().asDiagonal();
    const Eigen::SparseMatrix<double> Σ = Sinv * z.asDiagonal();

    // lhs = [H + AᵢᵀΣAᵢ  Aₑᵀ]
    //       [    Aₑ       0 ]
    //
    // Don't assign upper triangle because solver only uses lower triangle.
    const Eigen::SparseMatrix<double> topLeft =
        H.triangularView<Eigen::Lower>() +
        (A_i.transpose() * Σ * A_i).triangularView<Eigen::Lower>();
    triplets.clear();
    triplets.reserve(topLeft.nonZeros() + A_e.nonZeros());
    for (int col = 0; col < H.cols(); ++col) {
      // Append column of H + AᵢᵀΣAᵢ lower triangle in top-left quadrant
      for (Eigen::SparseMatrix<double>::InnerIterator it{topLeft, col}; it;
           ++it) {
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

    const Eigen::VectorXd e = Eigen::VectorXd::Ones(s.rows());

    // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(−Σcᵢ + μS⁻¹e + z)]
    //        [               cₑ                ]
    Eigen::VectorXd rhs{x.rows() + y.rows()};
    rhs.segment(0, x.rows()) = -g + A_e.transpose() * y +
                               A_i.transpose() * (-Σ * c_i + μ * Sinv * e + z);
    rhs.segment(x.rows(), y.rows()) = -c_e;

    linearSystemBuildProfiler.Stop();
    ScopedProfiler linearSystemSolveProfiler{linearSystemSolveProf};

    Eigen::VectorXd p_x;
    Eigen::VectorXd p_s;
    Eigen::VectorXd p_y;
    Eigen::VectorXd p_z;
    double α_max = 1.0;
    double α = 1.0;
    double α_z = 1.0;

    // Solve the Newton-KKT system
    //
    // [H + AᵢᵀΣAᵢ  Aₑᵀ][ pₖˣ] = −[∇f − Aₑᵀy + Aᵢᵀ(S⁻¹(Zcᵢ − μe) − z)]
    // [    Aₑ       0 ][−pₖʸ]    [                cₑ                ]
    solver.Compute(lhs, equalityConstraints.size(), μ);
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

    // pₖˢ = cᵢ − s + Aᵢpₖˣ
    p_s = c_i - s + A_i * p_x;

    // pₖᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ
    p_z = -Σ * c_i + μ * Sinv * e - Σ * A_i * p_x;

    // αᵐᵃˣ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
    α_max = FractionToTheBoundaryRule(s, p_s, τ);
    α = α_max;

    // αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
    α_z = FractionToTheBoundaryRule(z, p_z, τ);

    // Loop until a step is accepted
    while (1) {
      Eigen::VectorXd trial_x = x + α * p_x;
      Eigen::VectorXd trial_y = y + α_z * p_y;
      Eigen::VectorXd trial_z = z + α_z * p_z;

      xAD.SetValue(trial_x);

      Eigen::VectorXd trial_c_e = c_eAD.Value();
      Eigen::VectorXd trial_c_i = c_iAD.Value();

      // If f(xₖ + αpₖˣ), cₑ(xₖ + αpₖˣ), or cᵢ(xₖ + αpₖˣ) aren't finite, reduce
      // step size immediately
      if (!std::isfinite(f.Value()) || !trial_c_e.allFinite() ||
          !trial_c_i.allFinite()) {
        // Reduce step size
        α *= α_red_factor;
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
        trial_s = s + α * p_s;
      }

      // Check whether filter accepts trial iterate
      auto entry = filter.MakeEntry(trial_s, trial_c_e, trial_c_i, μ);
      if (filter.TryAdd(entry, α)) {
        // Accept step
        break;
      }

      double prevConstraintViolation = c_e.lpNorm<1>() + (c_i - s).lpNorm<1>();
      double nextConstraintViolation =
          trial_c_e.lpNorm<1>() + (trial_c_i - trial_s).lpNorm<1>();

      // Second-order corrections
      //
      // If first trial point was rejected and constraint violation stayed the
      // same or went up, apply second-order corrections
      if (α == α_max && nextConstraintViolation >= prevConstraintViolation) {
        // Apply second-order corrections. See section 2.4 of [2].
        Eigen::VectorXd p_x_cor = p_x;
        Eigen::VectorXd p_y_soc = p_y;
        Eigen::VectorXd p_z_soc = p_z;
        Eigen::VectorXd p_s_soc = p_s;

        double α_soc = α;
        double α_z_soc = α_z;
        Eigen::VectorXd c_e_soc = c_e;

        bool stepAcceptable = false;
        for (int soc_iteration = 0; soc_iteration < 5 && !stepAcceptable;
             ++soc_iteration) {
          ScopedProfiler socProfiler{socProf};

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
          scope_exit soc_exit{[&] {
            socProfiler.Stop();

            if (config.diagnostics) {
              double E = ErrorEstimate(g, A_e, trial_c_e, trial_y);
              PrintIterationDiagnostics(
                  iterations,
                  stepAcceptable ? IterationType::kAcceptedSOC
                                 : IterationType::kRejectedSOC,
                  socProfiler.CurrentDuration(), E, f.Value(),
                  trial_c_e.lpNorm<1>() + (trial_c_i - trial_s).lpNorm<1>(),
                  trial_s.dot(trial_z), solver.HessianRegularization(), α_soc,
                  α_z_soc);
            }
          }};
#endif

          // Rebuild Newton-KKT rhs with updated constraint values.
          //
          // rhs = −[∇f − Aₑᵀy − Aᵢᵀ(−Σcᵢ + μS⁻¹e + z)]
          //        [              cₑˢᵒᶜ              ]
          //
          // where cₑˢᵒᶜ = αc(xₖ) + c(xₖ + αpₖˣ)
          c_e_soc = α_soc * c_e_soc + trial_c_e;
          rhs.bottomRows(y.rows()) = -c_e_soc;

          // Solve the Newton-KKT system
          step = solver.Solve(rhs);

          p_x_cor = step.segment(0, x.rows());
          p_y_soc = -step.segment(x.rows(), y.rows());

          // pₖˢ = cᵢ − s + Aᵢpₖˣ
          p_s_soc = c_i - s + A_i * p_x_cor;

          // pₖᶻ = −Σcᵢ + μS⁻¹e − ΣAᵢpₖˣ
          p_z_soc = -Σ * c_i + μ * Sinv * e - Σ * A_i * p_x_cor;

          // αˢᵒᶜ = max(α ∈ (0, 1] : sₖ + αpₖˢ ≥ (1−τⱼ)sₖ)
          α_soc = FractionToTheBoundaryRule(s, p_s_soc, τ);
          trial_x = x + α_soc * p_x_cor;
          trial_s = s + α_soc * p_s_soc;

          // αₖᶻ = max(α ∈ (0, 1] : zₖ + αpₖᶻ ≥ (1−τⱼ)zₖ)
          α_z_soc = FractionToTheBoundaryRule(z, p_z_soc, τ);
          trial_y = y + α_z_soc * p_y_soc;
          trial_z = z + α_z_soc * p_z_soc;

          xAD.SetValue(trial_x);

          trial_c_e = c_eAD.Value();
          trial_c_i = c_iAD.Value();

          // Constraint violation scale factor for second-order corrections
          constexpr double κ_soc = 0.99;

          // If constraint violation hasn't been sufficiently reduced, stop
          // making second-order corrections
          nextConstraintViolation =
              trial_c_e.lpNorm<1>() + (trial_c_i - trial_s).lpNorm<1>();
          if (nextConstraintViolation > κ_soc * prevConstraintViolation) {
            break;
          }

          // Check whether filter accepts trial iterate
          entry = filter.MakeEntry(trial_s, trial_c_e, trial_c_i, μ);
          if (filter.TryAdd(entry, α)) {
            p_x = p_x_cor;
            p_y = p_y_soc;
            p_s = p_s_soc;
            p_z = p_z_soc;
            α = α_soc;
            α_z = α_z_soc;
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
        double currentKKTError = KKTError(g, A_e, c_e, A_i, c_i, s, y, z, μ);

        trial_x = x + α_max * p_x;
        trial_s = s + α_max * p_s;

        trial_y = y + α_z * p_y;
        trial_z = z + α_z * p_z;

        // Upate autodiff
        xAD.SetValue(trial_x);
        sAD.SetValue(trial_s);
        yAD.SetValue(trial_y);
        zAD.SetValue(trial_z);

        trial_c_e = c_eAD.Value();
        trial_c_i = c_iAD.Value();

        double nextKKTError = KKTError(gradientF.Value(), jacobianCe.Value(),
                                       trial_c_e, jacobianCi.Value(), trial_c_i,
                                       trial_s, trial_y, trial_z, μ);

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
      A_i_spy->Add(A_i);
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
      ++stepTooSmallCounter;
    } else {
      stepTooSmallCounter = 0;
    }

    // xₖ₊₁ = xₖ + αₖpₖˣ
    // sₖ₊₁ = sₖ + αₖpₖˢ
    // yₖ₊₁ = yₖ + αₖᶻpₖʸ
    // zₖ₊₁ = zₖ + αₖᶻpₖᶻ
    x += α * p_x;
    s += α * p_s;
    y += α_z * p_y;
    z += α_z * p_z;

    // A requirement for the convergence proof is that the "primal-dual barrier
    // term Hessian" Σₖ does not deviate arbitrarily much from the "primal
    // Hessian" μⱼSₖ⁻². We ensure this by resetting
    //
    //   zₖ₊₁⁽ⁱ⁾ = max(min(zₖ₊₁⁽ⁱ⁾, κ_Σ μⱼ/sₖ₊₁⁽ⁱ⁾), μⱼ/(κ_Σ sₖ₊₁⁽ⁱ⁾))
    //
    // for some fixed κ_Σ ≥ 1 after each step. See equation (16) of [2].
    for (int row = 0; row < z.rows(); ++row) {
      // Barrier parameter scale factor for inequality constraint Lagrange
      // multiplier safeguard
      constexpr double κ_Σ = 1e10;

      z(row) = std::max(std::min(z(row), κ_Σ * μ / s(row)), μ / (κ_Σ * s(row)));
    }

    // Update autodiff for Jacobians and Hessian
    xAD.SetValue(x);
    sAD.SetValue(s);
    yAD.SetValue(y);
    zAD.SetValue(z);
    A_e = jacobianCe.Value();
    A_i = jacobianCi.Value();
    g = gradientF.Value();
    H = hessianL.Value();

    ScopedProfiler nextIterPrepProfiler{nextIterPrepProf};

    c_e = c_eAD.Value();
    c_i = c_iAD.Value();

    // Update the error estimate
    E_0 = ErrorEstimate(g, A_e, c_e, A_i, c_i, s, y, z, 0.0);
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
      double E_μ = ErrorEstimate(g, A_e, c_e, A_i, c_i, s, y, z, μ);
      while (μ > μ_min && E_μ <= κ_ε * μ) {
        UpdateBarrierParameterAndResetFilter();
        E_μ = ErrorEstimate(g, A_e, c_e, A_i, c_i, s, y, z, μ);
      }
    }

    nextIterPrepProfiler.Stop();
    innerIterProfiler.Stop();

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (config.diagnostics) {
      PrintIterationDiagnostics(
          iterations, IterationType::kNormal,
          innerIterProfiler.CurrentDuration(), E_0, f.Value(),
          c_e.lpNorm<1>() + (c_i - s).lpNorm<1>(), s.dot(z),
          solver.HessianRegularization(), α, α_z);
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

    // The search direction has been very small twice, so assume the problem has
    // been solved as well as possible given finite precision and reduce the
    // barrier parameter.
    //
    // See section 3.9 of [2].
    if (stepTooSmallCounter >= 2 && μ > μ_min) {
      UpdateBarrierParameterAndResetFilter();
      continue;
    }
  }
}

}  // namespace sleipnir
