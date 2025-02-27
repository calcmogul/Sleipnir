// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include <Eigen/SparseCore>

#include "sleipnir/autodiff/adjoint_expression_graph.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/scoped_profiler.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "sleipnir/util/solve_profiler.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace sleipnir {

/**
 * This class calculates the Jacobian of a vector of variables with respect to a
 * vector of variables.
 *
 * The Jacobian is only recomputed if the variable expression is quadratic or
 * higher order.
 */
class SLEIPNIR_DLLEXPORT Jacobian {
 public:
  /**
   * Constructs a Jacobian object.
   *
   * @param variables Vector of variables of which to compute the Jacobian.
   * @param wrt Vector of variables with respect to which to compute the
   *   Jacobian.
   */
  Jacobian(VariableMatrix variables, SleipnirMatrixLike auto wrt) noexcept
      : m_variables{std::move(variables)}, m_wrt{std::move(wrt)} {
    // Initialize column each expression's adjoint occupies in the Jacobian
    for (size_t col = 0; col < m_wrt.size(); ++col) {
      m_wrt[col].expr->col = col;
    }

    for (auto& variable : m_variables) {
      m_graphs.emplace_back(variable);
    }

    // Reset col to -1
    for (auto& node : m_wrt) {
      node.expr->col = -1;
    }

    for (int row = 0; row < m_variables.rows(); ++row) {
      if (m_variables[row].expr == nullptr) {
        continue;
      }

      if (m_variables[row].type() == ExpressionType::LINEAR) {
        // If the row is linear, compute its gradient once here and cache its
        // triplets. Constant rows are ignored because their gradients have no
        // nonzero triplets.
        m_graphs[row].append_adjoint_triplets(m_cached_triplets, row, m_wrt);
      } else if (m_variables[row].type() > ExpressionType::LINEAR) {
        // If the row is quadratic or nonlinear, add it to the list of nonlinear
        // rows to be recomputed in Value().
        m_nonlinear_rows.emplace_back(row);
      }
    }

    if (m_nonlinear_rows.empty()) {
      m_J.setFromTriplets(m_cached_triplets.begin(), m_cached_triplets.end());
    }

    m_profilers.emplace_back("");
    m_profilers.emplace_back("    ↳ graph update");
    m_profilers.emplace_back("    ↳ adjoints");
    m_profilers.emplace_back("    ↳ matrix build");
  }

  /**
   * Returns the Jacobian as a VariableMatrix.
   *
   * This is useful when constructing optimization problems with derivatives in
   * them.
   *
   * @return The Jacobian as a VariableMatrix.
   */
  VariableMatrix get() const {
    VariableMatrix result{VariableMatrix::empty, m_variables.rows(),
                          m_wrt.rows()};

    for (int row = 0; row < m_variables.rows(); ++row) {
      auto grad = m_graphs[row].generate_gradient_tree(m_wrt);
      for (int col = 0; col < m_wrt.rows(); ++col) {
        if (grad[col].expr != nullptr) {
          result(row, col) = std::move(grad[col]);
        } else {
          result(row, col) = Variable{0.0};
        }
      }
    }

    return result;
  }

  /**
   * Evaluates the Jacobian at wrt's value.
   *
   * @return The Jacobian at wrt's value.
   */
  const Eigen::SparseMatrix<double>& value() {
    ScopedProfiler value_profiler{m_profilers[0]};

    if (m_nonlinear_rows.empty()) {
      return m_J;
    }

    ScopedProfiler graph_update_profiler{m_profilers[1]};

    for (auto& graph : m_graphs) {
      graph.update_values();
    }

    graph_update_profiler.stop();
    ScopedProfiler adjoints_profiler{m_profilers[2]};

    // Copy the cached triplets so triplets added for the nonlinear rows are
    // thrown away at the end of the function
    auto triplets = m_cached_triplets;

    // Compute each nonlinear row of the Jacobian
    for (int row : m_nonlinear_rows) {
      m_graphs[row].append_adjoint_triplets(triplets, row, m_wrt);
    }

    adjoints_profiler.stop();
    ScopedProfiler matrix_build_profiler{m_profilers[3]};

    if (!triplets.empty()) {
      m_J.setFromTriplets(triplets.begin(), triplets.end());
    } else {
      // setFromTriplets() is a no-op on empty triplets, so explicitly zero out
      // the storage
      m_J.setZero();
    }

    return m_J;
  }

  /**
   * Returns the profilers.
   *
   * @return The profilers.
   */
  const small_vector<SolveProfiler>& get_profilers() const {
    return m_profilers;
  }

 private:
  VariableMatrix m_variables;
  VariableMatrix m_wrt;

  small_vector<detail::AdjointExpressionGraph> m_graphs;

  Eigen::SparseMatrix<double> m_J{m_variables.rows(), m_wrt.rows()};

  // Cached triplets for gradients of linear rows
  small_vector<Eigen::Triplet<double>> m_cached_triplets;

  // List of row indices for nonlinear rows whose graients will be computed in
  // Value()
  small_vector<int> m_nonlinear_rows;

  small_vector<SolveProfiler> m_profilers;
};

}  // namespace sleipnir
