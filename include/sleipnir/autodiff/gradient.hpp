// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include <Eigen/SparseCore>

#include "sleipnir/autodiff/jacobian.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "sleipnir/util/solve_profiler.hpp"
#include "sleipnir/util/symbol_exports.hpp"

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
  Gradient(Variable variable, SleipnirMatrixLike auto wrt) noexcept
      : m_jacobian{std::move(variable), std::move(wrt)} {}

  /**
   * Returns the gradient as a VariableMatrix.
   *
   * This is useful when constructing optimization problems with derivatives in
   * them.
   *
   * @return The gradient as a VariableMatrix.
   */
  VariableMatrix get() const { return m_jacobian.get().T(); }

  /**
   * Evaluates the gradient at wrt's value.
   *
   * @return The gradient at wrt's value.
   */
  const Eigen::SparseVector<double>& value() {
    m_g = m_jacobian.value();

    return m_g;
  }

  /**
   * Returns the profiler.
   *
   * @return The profiler.
   */
  const small_vector<SolveProfiler>& get_profilers() const {
    return m_jacobian.get_profilers();
  }

 private:
  Eigen::SparseVector<double> m_g;

  Jacobian m_jacobian;
};

}  // namespace sleipnir
