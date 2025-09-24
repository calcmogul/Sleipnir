// Copyright (c) Sleipnir contributors

#pragma once

#include <cstddef>
#include <utility>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/adjoint_tape.hpp"
#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/// This class calculates the Jacobian of a vector of variables with respect to
/// a vector of variables.
///
/// The Jacobian is only recomputed if the variable expression is quadratic or
/// higher order.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class Jacobian {
 public:
  /// Constructs a Jacobian object.
  ///
  /// @param variable Variable of which to compute the Jacobian.
  /// @param wrt Variable with respect to which to compute the Jacobian.
  Jacobian(Variable<Scalar> variable, Variable<Scalar> wrt)
      : Jacobian{VariableMatrix<Scalar>{std::move(variable)},
                 VariableMatrix<Scalar>{std::move(wrt)}} {}

  /// Constructs a Jacobian object.
  ///
  /// @param variable Variable of which to compute the Jacobian.
  /// @param wrt Vector of variables with respect to which to compute the
  ///     Jacobian.
  Jacobian(Variable<Scalar> variable, SleipnirMatrixLike<Scalar> auto wrt)
      : Jacobian{VariableMatrix<Scalar>{std::move(variable)}, std::move(wrt)} {}

  /// Constructs a Jacobian object.
  ///
  /// @param variables Vector of variables of which to compute the Jacobian.
  /// @param wrt Vector of variables with respect to which to compute the
  ///     Jacobian.
  Jacobian(VariableMatrix<Scalar> variables,
           SleipnirMatrixLike<Scalar> auto wrt)
      : m_variables{std::move(variables)}, m_wrt{std::move(wrt)} {
    slp_assert(m_variables.cols() == 1);
    slp_assert(m_wrt.cols() == 1);

    gch::small_vector<detail::Expression<Scalar>*> wrt_exprs;
    for (auto& elem : m_wrt) {
      wrt_exprs.emplace_back(elem.expr.get());
    }

    // Split rows into linear and nonlinear rows. Constant rows are ignored
    // because their gradients have no nonzero entries.
    gch::small_vector<detail::Expression<Scalar>*> linear_roots;
    gch::small_vector<detail::Expression<Scalar>*> nonlinear_roots;
    for (int row = 0; row < m_variables.rows(); ++row) {
      if (m_variables[row].expr == nullptr) {
        continue;
      }

      if (m_variables[row].type() == ExpressionType::LINEAR) {
        linear_roots.emplace_back(m_variables[row].expr.get());
        m_linear_rows.emplace_back(row);
      } else if (m_variables[row].type() > ExpressionType::LINEAR) {
        nonlinear_roots.emplace_back(m_variables[row].expr.get());
        m_nonlinear_rows.emplace_back(row);
      }
    }

    m_linear_tape = detail::AdjointTape<Scalar>{linear_roots, wrt_exprs};
    m_nonlinear_tape = detail::AdjointTape<Scalar>{nonlinear_roots, wrt_exprs};

    // Map the tapes's adjoints of wrt to m_J's nonzeros. A tape's entry rows
    // are indices into its list of roots.
    //
    // Both tapes's entries go through one make_slot_map() call because
    // setFromTriplets() replaces m_J's entire sparsity pattern.
    gch::small_vector<detail::TapeEntry> entries;
    for (auto entry : m_linear_tape.entries()) {
      entry.row = m_linear_rows[entry.row];
      entries.push_back(entry);
    }
    for (auto entry : m_nonlinear_tape.entries()) {
      entry.row = m_nonlinear_rows[entry.row];
      entries.push_back(entry);
    }
    auto slot_map = detail::make_slot_map(m_J, entries);

    // Split slot map into linear and nonlinear slots
    size_t num_linear = m_linear_tape.entries().size();
    m_linear_slot_map.assign(slot_map.begin(), slot_map.begin() + num_linear);
    m_nonlinear_slot_map.assign(slot_map.begin() + num_linear, slot_map.end());

    // Linear rows have constant gradients, so compute them once here
    m_linear_tape.update_values();
    m_linear_tape.replay();
    detail::copy_slots(m_J, m_linear_slot_map, m_linear_tape.adjoints());
  }

  /// Returns the Jacobian as a VariableMatrix.
  ///
  /// This is useful when constructing optimization problems with derivatives in
  /// them.
  ///
  /// @return The Jacobian as a VariableMatrix.
  VariableMatrix<Scalar> get() const {
    VariableMatrix<Scalar> result{detail::empty, m_variables.rows(),
                                  m_wrt.rows()};

    for (int row = 0; row < m_variables.rows(); ++row) {
      auto grad =
          gradient_tree(detail::topological_sort(m_variables[row].expr));
      for (int col = 0; col < m_wrt.rows(); ++col) {
        if (grad[col].expr != nullptr) {
          result[row, col] = std::move(grad[col]);
        } else {
          result[row, col] = Variable{Scalar(0)};
        }
      }
    }

    return result;
  }

  /// Evaluates the Jacobian at wrt's value.
  ///
  /// @return The Jacobian at wrt's value.
  const Eigen::SparseMatrix<Scalar>& value() {
    if (m_nonlinear_rows.empty()) {
      return m_J;
    }

    m_nonlinear_tape.update_values();
    m_nonlinear_tape.replay();
    detail::copy_slots(m_J, m_nonlinear_slot_map, m_nonlinear_tape.adjoints());

    return m_J;
  }

 private:
  VariableMatrix<Scalar> m_variables;
  VariableMatrix<Scalar> m_wrt;

  /// Row indices of linear rows, whose gradients are computed once on
  /// construction
  gch::small_vector<int> m_linear_rows;

  /// Row indices of quadratic or nonlinear rows, whose gradients are computed
  /// in value()
  gch::small_vector<int> m_nonlinear_rows;

  detail::AdjointTape<Scalar> m_linear_tape;
  detail::AdjointTape<Scalar> m_nonlinear_tape;

  gch::small_vector<detail::SlotMapEntry> m_linear_slot_map;
  gch::small_vector<detail::SlotMapEntry> m_nonlinear_slot_map;

  Eigen::SparseMatrix<Scalar> m_J{m_variables.rows(), m_wrt.rows()};

  /// Returns the variable's gradient tree.
  ///
  /// This function lazily allocates variables, so elements of the returned
  /// VariableMatrix will be empty if the corresponding element of wrt had no
  /// adjoint. Ensure Variable::expr isn't nullptr before calling member
  /// functions.
  ///
  /// @param top_list Topologically sorted graph from parent to child.
  /// @return The variable's gradient tree.
  VariableMatrix<Scalar> gradient_tree(
      const detail::ExpressionGraph<Scalar>& top_list) const {
    slp_assert(m_wrt.cols() == 1);

    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    if (top_list.empty()) {
      return VariableMatrix<Scalar>{detail::empty, m_wrt.rows(), 1};
    }

    // Temporarily store each node's index in top_list in scratch
    for (size_t i = 0; i < top_list.size(); ++i) {
      top_list[i]->scratch = i;
    }

    // Adjoint of each node in top_list
    gch::small_vector<detail::ExpressionPtr<Scalar>> adjoints(top_list.size());

    // Set root node's adjoint to 1 since df/df is 1
    adjoints[0] = detail::constant_ptr(Scalar(1));

    // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
    // multiplied by dy/dx. If there are multiple "paths" from the root node to
    // variable; the variable's adjoint is the sum of each path's adjoint
    // contribution.
    for (size_t i = 0; i < top_list.size(); ++i) {
      const auto& node = top_list[i];

      // Leaf nodes have no children to propagate adjoints to, and a null (zero)
      // adjoint has nothing to contribute
      if (node->is_leaf || adjoints[i] == nullptr) {
        continue;
      }

      detail::ArgArray<detail::ExpressionPtr<Scalar>> g;
      node->grad_expr(g);

      // Constant arguments are skipped since their adjoints are never used
      size_t k = 0;
      node->visit_args([&](auto* arg) {
        if (g[k] != nullptr && arg->type() != ExpressionType::CONSTANT) {
          adjoints[arg->scratch] += g[k] * adjoints[i];
        }
        ++k;
      });
    }

    // Move gradient tree to return value. Elements of wrt that aren't in the
    // graph have no adjoint.
    VariableMatrix<Scalar> grad{detail::empty, m_wrt.rows(), 1};
    for (int row = 0; row < grad.rows(); ++row) {
      if (int idx = m_wrt[row].expr->scratch; idx != -1) {
        grad[row] = Variable{std::move(adjoints[idx])};
      }
    }

    for (auto& node : top_list) {
      node->scratch = -1;
    }

    return grad;
  }
};

extern template class EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
Jacobian<double>;

}  // namespace slp
