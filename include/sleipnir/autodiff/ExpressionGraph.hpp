// Copyright (c) Sleipnir contributors

#pragma once

#include <ranges>

#include <wpi/SmallVector.h>

#include "sleipnir/autodiff/Expression.hpp"

namespace sleipnir::detail {

/**
 * Generate a topological sort of an expression graph from parent to child.
 *
 * https://en.wikipedia.org/wiki/Topological_sorting
 *
 * @param root The root node of the expression.
 */
inline wpi::SmallVector<Expression*> TopologicalSort(
    const ExpressionPtr& root) {
  wpi::SmallVector<Expression*> list;

  // If the root type is a constant, Update() is a no-op, so there's no work
  // to do
  if (root == nullptr || root->Type() == ExpressionType::kConstant) {
    return list;
  }

  // Stack of nodes to explore
  wpi::SmallVector<Expression*> stack;

  // Enumerate incoming edges for each node via depth-first search
  stack.emplace_back(root.Get());
  while (!stack.empty()) {
    auto node = stack.back();
    stack.pop_back();

    for (auto& arg : node->args) {
      // If the node hasn't been explored yet, add it to the stack
      if (arg != nullptr && ++arg->incomingEdges == 1) {
        stack.push_back(arg.Get());
      }
    }
  }

  // Generate topological sort of graph from parent to child.
  //
  // A node is only added to the stack after all its incoming edges have been
  // traversed. Expression::incomingEdges is a decrementing counter for
  // tracking this.
  //
  // https://en.wikipedia.org/wiki/Topological_sorting
  stack.emplace_back(root.Get());
  while (!stack.empty()) {
    auto node = stack.back();
    stack.pop_back();

    list.emplace_back(node);

    for (auto& arg : node->args) {
      // If we traversed all this node's incoming edges, add it to the stack
      if (arg != nullptr && --arg->incomingEdges == 0) {
        stack.push_back(arg.Get());
      }
    }
  }

  return list;
}

/**
 * Update the values of all nodes in this graph based on the values of
 * their dependent nodes.
 *
 * @param list Topological sort of graph from parent to child.
 */
inline void UpdateValues(const wpi::SmallVector<Expression*>& list) {
  // Traverse graph from child to parent and update values
  for (auto& node : list | std::views::reverse) {
    auto& lhs = node->args[0];
    auto& rhs = node->args[1];

    if (lhs != nullptr) {
      if (rhs != nullptr) {
        node->value = node->Value(lhs->value, rhs->value);
      } else {
        node->value = node->Value(lhs->value, 0.0);
      }
    }
  }
}

/**
 * Updates the adjoints in the expression graph (computes the gradient).
 *
 * @param list Topological sort of graph from parent to child.
 */
inline void UpdateAdjoints(const wpi::SmallVector<Expression*>& list) {
  // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
  // for background on reverse accumulation automatic differentiation.

  if (list.empty()) {
    return;
  }

  // Set root node's adjoint to 1 since df/df is 1
  list[0]->adjoint = 1.0;

  // Zero the rest of the adjoints
  for (auto& node : list | std::views::drop(1)) {
    node->adjoint = 0.0;
  }

  // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
  // multiplied by dy/dx. If there are multiple "paths" from the root node to
  // variable; the variable's adjoint is the sum of each path's adjoint
  // contribution.
  for (const auto& node : list) {
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
}

}  // namespace sleipnir::detail
