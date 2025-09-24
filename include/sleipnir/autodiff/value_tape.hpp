// Copyright (c) Sleipnir contributors

#pragma once

#include <ranges>

#include <Eigen/Core>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/adjoint_tape.hpp"
#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/variable.hpp"

namespace slp::detail {

/// Evaluates the values of several variables with a tape instead of
/// traversing their expression graphs.
///
/// This is faster than calling Variable::value() on each variable when the
/// values are needed repeatedly, since the tape stores the expression graphs as
/// flat arrays and evaluates nodes shared between variables once.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class ValueTape {
 public:
  /// Constructs a ValueTape.
  ///
  /// The variables's expression graphs must outlive the ValueTape, and their
  /// structure must not change while it's in use.
  ///
  /// @param variables The variables whose values to evaluate.
  explicit ValueTape(const std::ranges::input_range auto& variables) {
    gch::small_vector<Expression<Scalar>*> roots;
    for (const auto& variable : variables) {
      roots.emplace_back(variable.expr.get());
    }
    m_tape = AdjointTape<Scalar>{roots, {}};
    m_values.resize(roots.size());
  }

  /// Evaluates the variables's values from the current values of the leaves
  /// they depend on.
  ///
  /// @return The variables's values.
  const Eigen::Vector<Scalar, Eigen::Dynamic>& value() {
    m_tape.template update_values<0>();

    for (int i = 0; i < m_values.rows(); ++i) {
      m_values[i] = m_tape.root_value(i);
    }
    return m_values;
  }

 private:
  AdjointTape<Scalar> m_tape;
  Eigen::Vector<Scalar, Eigen::Dynamic> m_values;
};

}  // namespace slp::detail
