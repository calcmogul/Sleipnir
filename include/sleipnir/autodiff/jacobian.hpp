// Copyright (c) Sleipnir contributors

#pragma once

#include <cstddef>
#include <span>
#include <utility>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/adjoint_tape.hpp"
#include "sleipnir/autodiff/expression.hpp"
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

    // A tape's entry rows are indices into its list of roots
    auto set_entries = [&](const detail::AdjointTape<Scalar>& tape,
                           std::span<const int> rows) {
      auto adjoints = tape.symbolic_replay(tape.symbolic_partials());
      for (const auto& entry : tape.entries()) {
        result[rows[entry.row], entry.col] =
            Variable{std::move(adjoints[entry.slot])};
      }
    };
    set_entries(m_linear_tape, m_linear_rows);
    set_entries(m_nonlinear_tape, m_nonlinear_rows);

    // The tapes only have entries for nonzero adjoints
    for (int row = 0; row < m_variables.rows(); ++row) {
      for (int col = 0; col < m_wrt.rows(); ++col) {
        if (result[row, col].expr == nullptr) {
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
};

extern template class EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
Jacobian<double>;

}  // namespace slp
