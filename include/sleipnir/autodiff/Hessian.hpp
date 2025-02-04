// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include <Eigen/SparseCore>

#include "sleipnir/autodiff/AdjointExpressionGraph.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/util/ScopedProfiler.hpp"
#include "sleipnir/util/SolveProfiler.hpp"
#include "sleipnir/util/SymbolExports.hpp"
#include "sleipnir/util/small_vector.hpp"

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
   * @param onlyLower Only compute lower triangle of Hessian.
   */
  Hessian(Variable variable, VariableMatrix wrt,
          bool onlyLower = false) noexcept
      : m_variables{detail::AdjointExpressionGraph{variable}
                        .GenerateGradientTree(wrt)},
        m_wrt{wrt} {
    // Initialize column each expression's adjoint occupies in the Hessian
    for (size_t col = 0; col < m_wrt.size(); ++col) {
      m_wrt(col).expr->col = col;
    }

    if (onlyLower) {
      for (size_t col = 0; col < m_wrt.size(); ++col) {
        m_graphs.emplace_back(m_variables(col));
      }
    } else {
      for (size_t col = 0; col < m_wrt.size(); ++col) {
        m_graphs.emplace_back(m_variables(col));
      }
    }

    // Reset col to -1
    for (auto& node : m_wrt) {
      node.expr->col = -1;
    }

    for (int row = 0; row < m_variables.Rows(); ++row) {
      if (m_variables(row).expr == nullptr) {
        continue;
      }

      if (m_variables(row).Type() == ExpressionType::kLinear) {
        // If the row is linear, compute its gradient once here and cache its
        // triplets. Constant rows are ignored because their gradients have no
        // nonzero triplets.
        m_graphs[row].AppendAdjointTriplets(m_cachedTriplets, row);
      } else if (m_variables(row).Type() > ExpressionType::kLinear) {
        // If the row is quadratic or nonlinear, add it to the list of nonlinear
        // rows to be recomputed in Value().
        m_nonlinearRows.emplace_back(row);
      }
    }

    if (m_nonlinearRows.empty()) {
      m_H.setFromTriplets(m_cachedTriplets.begin(), m_cachedTriplets.end());
    }

    m_profilers.emplace_back("");
    m_profilers.emplace_back("    ↳ graph update");
    m_profilers.emplace_back("    ↳ adjoints");
    m_profilers.emplace_back("    ↳ matrix build");
  }

  /**
   * Returns the Hessian as a VariableMatrix.
   *
   * This is useful when constructing optimization problems with derivatives in
   * them.
   */
  VariableMatrix Get() const {
    VariableMatrix result{VariableMatrix::empty, m_variables.Rows(),
                          m_wrt.Rows()};

    for (int row = 0; row < m_variables.Rows(); ++row) {
      auto grad = m_graphs[row].GenerateGradientTree(m_wrt);
      for (int col = 0; col < m_wrt.Rows(); ++col) {
        if (grad(col).expr != nullptr) {
          result(row, col) = std::move(grad(col));
        } else {
          result(row, col) = Variable{0.0};
        }
      }
    }

    return result;
  }

  /**
   * Evaluates the Hessian at wrt's value.
   */
  const Eigen::SparseMatrix<double>& Value() {
    ScopedProfiler valueProfiler{m_profilers[0]};

    if (m_nonlinearRows.empty()) {
      return m_H;
    }

    ScopedProfiler graphUpdateProfiler{m_profilers[1]};

    for (auto& graph : m_graphs) {
      graph.UpdateValues();
    }

    graphUpdateProfiler.Stop();
    ScopedProfiler adjointsProfiler{m_profilers[2]};

    // Copy the cached triplets so triplets added for the nonlinear rows are
    // thrown away at the end of the function
    auto triplets = m_cachedTriplets;

    // Compute each nonlinear row of the Hessian
    for (int row : m_nonlinearRows) {
      m_graphs[row].AppendAdjointTriplets(triplets, row);
    }

    adjointsProfiler.Stop();
    ScopedProfiler matrixBuildProfiler{m_profilers[3]};

    if (!triplets.empty()) {
      m_H.setFromTriplets(triplets.begin(), triplets.end());
    } else {
      // setFromTriplets() is a no-op on empty triplets, so explicitly zero out
      // the storage
      m_H.setZero();
    }

    return m_H;
  }

  /**
   * Returns the profilers.
   */
  const small_vector<SolveProfiler>& GetProfilers() const {
    return m_profilers;
  }

 private:
  VariableMatrix m_variables;
  VariableMatrix m_wrt;

  small_vector<detail::AdjointExpressionGraph> m_graphs;

  Eigen::SparseMatrix<double> m_H{m_variables.Rows(), m_wrt.Rows()};

  // Cached triplets for gradients of linear rows
  small_vector<Eigen::Triplet<double>> m_cachedTriplets;

  // List of row indices for nonlinear rows whose graients will be computed in
  // Value()
  small_vector<int> m_nonlinearRows;

  small_vector<SolveProfiler> m_profilers;
};

}  // namespace sleipnir
