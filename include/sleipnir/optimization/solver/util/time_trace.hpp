// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <array>

#include "sleipnir/util/time_tracer.hpp"

namespace slp {

/// Autodiff setup tracer IDs.
///
/// Time traces list these in declaration order.
enum class SetupTimeTracerID : uint8_t {
  /// Total autodiff setup.
  SETUP,
  /// Cost gradient setup.
  GRADIENT,
  /// Lagrangian Hessian setup.
  HESSIAN,
  /// Cost part of Lagrangian Hessian setup.
  HESSIAN_COST,
  /// Constraint part of Lagrangian Hessian setup.
  HESSIAN_CONSTRAINTS,
  /// Equality constraint Jacobian setup.
  EQUALITY_JACOBIAN,
  /// Inequality constraint Jacobian setup.
  INEQUALITY_JACOBIAN,
};

/// Solver tracer IDs.
///
/// Time traces list these in declaration order.
enum class SolveTimeTracerID : uint8_t {
  /// Total solve.
  SOLVER,
  /// Solver setup.
  SETUP,
  /// Solver iteration.
  ITERATION,
  /// Iteration callbacks.
  CALLBACKS,
  /// Barrier parameter update.
  BARRIER_PARAMETER_UPDATE,
  /// KKT matrix build.
  KKT_MATRIX_BUILD,
  /// KKT matrix decomposition.
  KKT_MATRIX_DECOMP,
  /// KKT system solve.
  KKT_SYSTEM_SOLVE,
  /// Line search.
  LINE_SEARCH,
  /// Second-order corrections.
  SOC,
  /// Feasibility restoration.
  FEASIBILITY_RESTORATION,
  /// Cost evaluation.
  COST,
  /// Cost gradient evaluation.
  GRADIENT,
  /// Lagrangian Hessian evaluation.
  HESSIAN,
  /// Constraint part of Lagrangian Hessian evaluation.
  HESSIAN_CONSTRAINTS,
  /// Equality constraint evaluation.
  EQUALITY_CONSTRAINTS,
  /// Equality constraint Jacobian evaluation.
  EQUALITY_JACOBIAN,
  /// Inequality constraint evaluation.
  INEQUALITY_CONSTRAINTS,
  /// Inequality constraint Jacobian evaluation.
  INEQUALITY_JACOBIAN,
};

/// Autodiff setup time trace for unconstrained problems.
inline constexpr std::array newton_setup_time_trace{
    TimeTraceEntry{SetupTimeTracerID::SETUP, "setup", 0},
    TimeTraceEntry{SetupTimeTracerID::GRADIENT, "∇f(x)", 1},
    TimeTraceEntry{SetupTimeTracerID::HESSIAN, "∇²ₓₓL", 1},
};

/// Autodiff setup time trace for equality-constrained problems.
inline constexpr std::array sqp_setup_time_trace{
    TimeTraceEntry{SetupTimeTracerID::SETUP, "setup", 0},
    TimeTraceEntry{SetupTimeTracerID::GRADIENT, "∇f(x)", 1},
    TimeTraceEntry{SetupTimeTracerID::HESSIAN, "∇²ₓₓL", 1},
    TimeTraceEntry{SetupTimeTracerID::HESSIAN_COST, "∇²ₓₓL_f", 2},
    TimeTraceEntry{SetupTimeTracerID::HESSIAN_CONSTRAINTS, "∇²ₓₓL_c", 2},
    TimeTraceEntry{SetupTimeTracerID::EQUALITY_JACOBIAN, "∂cₑ/∂x", 1},
};

/// Autodiff setup time trace for inequality-constrained problems.
inline constexpr std::array ipm_setup_time_trace{
    TimeTraceEntry{SetupTimeTracerID::SETUP, "setup", 0},
    TimeTraceEntry{SetupTimeTracerID::GRADIENT, "∇f(x)", 1},
    TimeTraceEntry{SetupTimeTracerID::HESSIAN, "∇²ₓₓL", 1},
    TimeTraceEntry{SetupTimeTracerID::HESSIAN_COST, "∇²ₓₓL_f", 2},
    TimeTraceEntry{SetupTimeTracerID::HESSIAN_CONSTRAINTS, "∇²ₓₓL_c", 2},
    TimeTraceEntry{SetupTimeTracerID::INEQUALITY_JACOBIAN, "∂cᵢ/∂x", 1},
};

/// Newton's method time trace.
inline constexpr std::array newton_time_trace{
    TimeTraceEntry{SolveTimeTracerID::SOLVER, "solver", 0},
    TimeTraceEntry{SolveTimeTracerID::SETUP, "setup", 1},
    TimeTraceEntry{SolveTimeTracerID::ITERATION, "iteration", 1},
    TimeTraceEntry{SolveTimeTracerID::CALLBACKS, "callbacks", 2},
    TimeTraceEntry{SolveTimeTracerID::KKT_MATRIX_DECOMP, "KKT matrix decomp",
                   2},
    TimeTraceEntry{SolveTimeTracerID::KKT_SYSTEM_SOLVE, "KKT system solve", 2},
    TimeTraceEntry{SolveTimeTracerID::LINE_SEARCH, "line search", 2},
    TimeTraceEntry{SolveTimeTracerID::COST, "f(x)", 2},
    TimeTraceEntry{SolveTimeTracerID::GRADIENT, "∇f(x)", 2},
    TimeTraceEntry{SolveTimeTracerID::HESSIAN, "∇²ₓₓL", 2},
};

/// Sequential quadratic programming time trace.
inline constexpr std::array sqp_time_trace{
    TimeTraceEntry{SolveTimeTracerID::SOLVER, "solver", 0},
    TimeTraceEntry{SolveTimeTracerID::SETUP, "setup", 1},
    TimeTraceEntry{SolveTimeTracerID::ITERATION, "iteration", 1},
    TimeTraceEntry{SolveTimeTracerID::CALLBACKS, "callbacks", 2},
    TimeTraceEntry{SolveTimeTracerID::KKT_MATRIX_BUILD, "KKT matrix build", 2},
    TimeTraceEntry{SolveTimeTracerID::KKT_MATRIX_DECOMP, "KKT matrix decomp",
                   2},
    TimeTraceEntry{SolveTimeTracerID::KKT_SYSTEM_SOLVE, "KKT system solve", 2},
    TimeTraceEntry{SolveTimeTracerID::LINE_SEARCH, "line search", 2},
    TimeTraceEntry{SolveTimeTracerID::SOC, "SOC", 3},
    TimeTraceEntry{SolveTimeTracerID::FEASIBILITY_RESTORATION,
                   "feas. restoration", 2},
    TimeTraceEntry{SolveTimeTracerID::COST, "f(x)", 2},
    TimeTraceEntry{SolveTimeTracerID::GRADIENT, "∇f(x)", 2},
    TimeTraceEntry{SolveTimeTracerID::HESSIAN, "∇²ₓₓL", 2},
    TimeTraceEntry{SolveTimeTracerID::HESSIAN_CONSTRAINTS, "∇²ₓₓL_c", 2},
    TimeTraceEntry{SolveTimeTracerID::EQUALITY_CONSTRAINTS, "cₑ(x)", 2},
    TimeTraceEntry{SolveTimeTracerID::EQUALITY_JACOBIAN, "∂cₑ/∂x", 2},
};

/// Interior-point method time trace.
inline constexpr std::array ipm_time_trace{
    TimeTraceEntry{SolveTimeTracerID::SOLVER, "solver", 0},
    TimeTraceEntry{SolveTimeTracerID::SETUP, "setup", 1},
    TimeTraceEntry{SolveTimeTracerID::ITERATION, "iteration", 1},
    TimeTraceEntry{SolveTimeTracerID::CALLBACKS, "callbacks", 2},
    TimeTraceEntry{SolveTimeTracerID::BARRIER_PARAMETER_UPDATE, "μ update", 2},
    TimeTraceEntry{SolveTimeTracerID::KKT_MATRIX_BUILD, "KKT matrix build", 2},
    TimeTraceEntry{SolveTimeTracerID::KKT_MATRIX_DECOMP, "KKT matrix decomp",
                   2},
    TimeTraceEntry{SolveTimeTracerID::KKT_SYSTEM_SOLVE, "KKT system solve", 2},
    TimeTraceEntry{SolveTimeTracerID::LINE_SEARCH, "line search", 2},
    TimeTraceEntry{SolveTimeTracerID::SOC, "SOC", 3},
    TimeTraceEntry{SolveTimeTracerID::FEASIBILITY_RESTORATION,
                   "feas. restoration", 2},
    TimeTraceEntry{SolveTimeTracerID::COST, "f(x)", 2},
    TimeTraceEntry{SolveTimeTracerID::GRADIENT, "∇f(x)", 2},
    TimeTraceEntry{SolveTimeTracerID::HESSIAN, "∇²ₓₓL", 2},
    TimeTraceEntry{SolveTimeTracerID::HESSIAN_CONSTRAINTS, "∇²ₓₓL_c", 2},
    TimeTraceEntry{SolveTimeTracerID::INEQUALITY_CONSTRAINTS, "cᵢ(x)", 2},
    TimeTraceEntry{SolveTimeTracerID::INEQUALITY_JACOBIAN, "∂cᵢ/∂x", 2},
};

}  // namespace slp
