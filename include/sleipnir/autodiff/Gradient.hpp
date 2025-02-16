// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include <Eigen/SparseCore>

#include "sleipnir/autodiff/Jacobian.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/util/SolveProfiler.hpp"
#include "sleipnir/util/SymbolExports.hpp"
#include "sleipnir/util/small_vector.hpp"

namespace sleipnir {

/**
 * This class calculates the gradient of a a variable with respect to a vector
 * of variables.
 *
 * The gradient is only recomputed if the variable expression is quadratic or
 * higher order.
 */
class SLEIPNIR_DLLEXPORT Gradient {
 public:
  /**
   * Constructs a Gradient object.
   *
   * @param variable Variable of which to compute the gradient.
   * @param wrt Variable with respect to which to compute the gradient.
   */
  Gradient(Variable variable, Variable wrt) noexcept
      : m_jacobian{std::move(variable), VariableMatrix{std::move(wrt)}} {}

  /**
   * Constructs a Gradient object.
   *
   * @param variable Variable of which to compute the gradient.
   * @param wrt Vector of variables with respect to which to compute the
   *   gradient.
   */
  Gradient(Variable variable, VariableMatrix wrt) noexcept
      : m_jacobian{std::move(variable), std::move(wrt)} {}

  /**
   * Returns the gradient as a VariableMatrix.
   *
   * This is useful when constructing optimization problems with derivatives in
   * them.
   *
   * @return The gradient as a VariableMatrix.
   */
  VariableMatrix Get() const { return m_jacobian.Get().T(); }

  /**
   * Evaluates the gradient at wrt's value.
   *
   * @return The gradient at wrt's value.
   */
  const Eigen::SparseVector<double>& Value() {
    m_g = m_jacobian.Value();

    return m_g;
  }

  /**
   * Returns the profiler.
   *
   * @return The profiler.
   */
  const small_vector<SolveProfiler>& GetProfilers() const {
    return m_jacobian.GetProfilers();
  }

 private:
  Eigen::SparseVector<double> m_g;

  Jacobian m_jacobian;
};

}  // namespace sleipnir
