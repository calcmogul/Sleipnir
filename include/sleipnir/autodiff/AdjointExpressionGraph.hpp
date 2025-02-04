// Copyright (c) Sleipnir contributors

#pragma once

#include <ranges>
#include <utility>

#include <Eigen/SparseCore>

#include "sleipnir/autodiff/ExpressionGraph.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/util/small_vector.hpp"

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
   * @param wrt Vector of variables with respect to which to compute the
   *   adjoints.
   */
  explicit AdjointExpressionGraph(const Variable& root,
                                  const VariableMatrix& wrt)
      : m_topList{TopologicalSort(root.expr)} {
    // Mark nodes that are parents of wrt
    for (auto& node : wrt) {
      node.expr->visited = true;
    }
    for (auto& node : m_topList | std::views::reverse) {
      for (auto& arg : node->args) {
        node->visited |= arg != nullptr && arg->visited;
      }
    }

    // Prune unnecessary nodes
    // erase_if(m_topList, [](const auto& elem) { return !elem->visited; });

    // Unmark nodes
    for (auto& node : m_topList) {
      node->visited = false;
    }
    for (auto& node : wrt) {
      node.expr->visited = false;
    }

    for (const auto& node : m_topList) {
      // Build column index list
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
   */
  void AppendAdjointTriplets(small_vector<Eigen::Triplet<double>>& triplets,
                             int row) const {
    detail::UpdateAdjoints(m_topList);

    for (const auto& [node, col] : std::views::zip(m_topList, m_colList)) {
      // Append adjoints of wrt to sparse matrix triplets
      if (col != -1 && node->adjoint != 0.0) {
        triplets.emplace_back(row, col, node->adjoint);
      }
    }
  }

 private:
  // Topological sort of graph from parent to child
  small_vector<Expression*> m_topList;

  // List that maps nodes to their respective column
  small_vector<int> m_colList;
};

}  // namespace sleipnir::detail
