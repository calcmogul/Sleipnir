// Copyright (c) Sleipnir contributors

#pragma once

#include <ranges>
#include <utility>

#include <Eigen/SparseCore>
#include <wpi/SmallVector.h>

#include "sleipnir/autodiff/ExpressionGraph.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"

namespace sleipnir::detail {

/**
 * This class is an adaptor type that performs value updates of an expression's
 * adjoint graph.
 */
class AdjointExpressionGraph {
 public:
  /**
   * Generates the adjoint graph for the given expression.
   *
   * @param root The root node of the expression.
   */
  explicit AdjointExpressionGraph(const Variable& root)
      : m_topList{TopologicalSort(root.expr)} {
    for (const auto& node : m_topList) {
      m_colList.emplace_back(node->col);
    }
  }

  /**
   * Update the values of all nodes in this adjoint graph based on the values of
   * their dependent nodes.
   */
  void UpdateValues() { detail::UpdateValues(m_topList); }

  /**
   * Returns the variable's gradient tree.
   *
   * This function lazily allocates variables, so elements of the returned
   * VariableMatrix will be empty if the corresponding element of wrt had no
   * adjoint. Ensure Variable::expr isn't nullptr before calling member
   * functions.
   *
   * @param wrt Variables with respect to which to compute the gradient.
   * @return The variable's gradient tree.
   */
  VariableMatrix GenerateGradientTree(const VariableMatrix& wrt) const {
    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    if (m_topList.empty()) {
      return VariableMatrix{VariableMatrix::empty, wrt.Rows(), 1};
    }

    // Set root node's adjoint to 1 since df/df is 1
    m_topList[0]->adjointExpr = MakeExpressionPtr<ConstExpression>(1.0);

    // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
    // multiplied by dy/dx. If there are multiple "paths" from the root node to
    // variable; the variable's adjoint is the sum of each path's adjoint
    // contribution.
    for (auto& node : m_topList) {
      auto& lhs = node->args[0];
      auto& rhs = node->args[1];

      if (lhs != nullptr) {
        lhs->adjointExpr += node->GradExprL(lhs, rhs, node->adjointExpr);
        if (rhs != nullptr) {
          rhs->adjointExpr += node->GradExprR(lhs, rhs, node->adjointExpr);
        }
      }
    }

    // Move gradient tree to return value
    VariableMatrix grad{VariableMatrix::empty, wrt.Rows(), 1};
    for (int row = 0; row < grad.Rows(); ++row) {
      grad(row) = Variable{std::move(wrt(row).expr->adjointExpr)};
    }

    // Unlink adjoints to avoid circular references between them and their
    // parent expressions. This ensures all expressions are returned to the free
    // list.
    for (auto& node : m_topList) {
      node->adjointExpr = nullptr;
    }

    return grad;
  }

  /**
   * Updates the adjoints in the expression graph (computes the gradient) then
   * appends the adjoints of wrt to the sparse matrix triplets.
   *
   * @param triplets The sparse matrix triplets.
   * @param row The row of wrt.
   * @param wrt Vector of variables with respect to which to compute the
   *   Jacobian.
   */
  void AppendAdjointTriplets(wpi::SmallVector<Eigen::Triplet<double>>& triplets,
                             int row, const VariableMatrix& wrt) const {
    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    // If wrt has fewer nodes than graph, zero wrt's adjoints
    if (static_cast<size_t>(wrt.Rows()) < m_topList.size()) {
      for (const auto& elem : wrt) {
        elem.expr->adjoint = 0.0;
      }
    }

    if (m_topList.empty()) {
      return;
    }

    // Set root node's adjoint to 1 since df/df is 1
    m_topList[0]->adjoint = 1.0;

    // Zero the rest of the adjoints
    for (auto& node : m_topList | std::views::drop(1)) {
      node->adjoint = 0.0;
    }

    // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
    // multiplied by dy/dx. If there are multiple "paths" from the root node to
    // variable; the variable's adjoint is the sum of each path's adjoint
    // contribution.
    for (const auto& node : m_topList) {
      auto& lhs = node->args[0];
      auto& rhs = node->args[1];

      if (lhs != nullptr) {
        if (rhs != nullptr) {
          lhs->adjoint += node->GradL(lhs->value, rhs->value, node->adjoint);
          rhs->adjoint += node->GradR(lhs->value, rhs->value, node->adjoint);
        } else {
          lhs->adjoint += node->GradL(lhs->value, 0.0, node->adjoint);
        }
      }
    }

    // If wrt has fewer nodes than graph, iterate over wrt
    if (static_cast<size_t>(wrt.Rows()) < m_topList.size()) {
      for (int col = 0; col < wrt.Rows(); ++col) {
        const auto& node = wrt(col).expr;

        // Append adjoints of wrt to sparse matrix triplets
        if (node->adjoint != 0.0) {
          triplets.emplace_back(row, col, node->adjoint);
        }
      }
    } else {
      for (const auto& [col, node] : std::views::zip(m_colList, m_topList)) {
        // Append adjoints of wrt to sparse matrix triplets
        if (col != -1 && node->adjoint != 0.0) {
          triplets.emplace_back(row, col, node->adjoint);
        }
      }
    }
  }

 private:
  // Topological sort of graph from parent to child
  wpi::SmallVector<Expression*> m_topList;

  // List that maps nodes to their respective column
  wpi::SmallVector<int> m_colList;
};

}  // namespace sleipnir::detail
