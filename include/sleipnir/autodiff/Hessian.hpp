// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/SparseCore>
#include <wpi/SmallVector.h>

#include "sleipnir/autodiff/AdjointExpressionGraph.hpp"
#include "sleipnir/autodiff/Jacobian.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/util/SolveProfiler.hpp"
#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * This class calculates the Hessian of a variable with respect to a vector of
 * variables.
 *
 * The gradient tree is cached so subsequent Hessian calculations are faster,
 * and the Hessian is only recomputed if the variable expression is nonlinear.
 */
class SLEIPNIR_DLLEXPORT Hessian {
 public:
  /**
   * Constructs a Hessian object.
   *
   * @param variable Variable of which to compute the Hessian.
   * @param wrt Vector of variables with respect to which to compute the
   *   Hessian.
   */
  Hessian(Variable variable, VariableMatrix wrt) noexcept
      : m_jacobian{
            detail::AdjointExpressionGraph{variable}.GenerateGradientTree(wrt),
            wrt} {}

  /**
   * Returns the Hessian as a VariableMatrix.
   *
   * This is useful when constructing optimization problems with derivatives in
   * them.
   */
  VariableMatrix Get() const { return m_jacobian.Get(); }

  /**
   * Evaluates the Hessian at wrt's value.
   */
  const Eigen::SparseMatrix<double>& Value() { return m_jacobian.Value(); }

  /**
   * Returns the profilers.
   */
  const wpi::SmallVector<SolveProfiler>& GetProfilers() const {
    return m_jacobian.GetProfilers();
  }

 private:
  Jacobian m_jacobian;
};

}  // namespace sleipnir
