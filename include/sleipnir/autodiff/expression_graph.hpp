// Copyright (c) Sleipnir contributors

#pragma once

#include <ranges>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"

namespace slp::detail {

/// Typedef for topologically sorted expression graph from parent to child.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
using ExpressionGraph = gch::small_vector<Expression<Scalar>*>;

/// Generates a topological sort of an expression graph from parent to child.
///
/// https://en.wikipedia.org/wiki/Topological_sorting
///
/// @tparam Scalar Scalar type.
/// @param root The root node of the expression.
template <typename Scalar>
ExpressionGraph<Scalar> topological_sort(const ExpressionPtr<Scalar>& root) {
  ExpressionGraph<Scalar> list;

  // If the root type is constant, updates are a no-op, so return an empty list
  if (root == nullptr || root->type() == ExpressionType::CONSTANT) {
    return list;
  }

  // Stack of nodes to explore
  gch::small_vector<Expression<Scalar>*> stack;

  // Enumerate incoming edges for each node via depth-first search
  //
  // NOTE: scratch counts incoming edges, offset by -1 so -1 means no edges.
  stack.emplace_back(root.get());
  while (!stack.empty()) {
    auto node = stack.back();
    stack.pop_back();

    node->visit_args([&stack](const auto& arg) {
      // If the node hasn't been explored yet, add it to the stack
      if (++arg->scratch == 0) {
        stack.push_back(arg);
      }
    });
  }

  // Generate topological sort of graph from parent to child.
  //
  // A node is only added to the stack after all its incoming edges have been
  // traversed. Expression::scratch is a decrementing counter for tracking this.
  //
  // https://en.wikipedia.org/wiki/Topological_sorting
  stack.emplace_back(root.get());
  while (!stack.empty()) {
    auto node = stack.back();
    stack.pop_back();

    list.emplace_back(node);

    node->visit_args([&stack](const auto& arg) {
      // If we traversed all this node's incoming edges, add it to the stack
      if (--arg->scratch == -1) {
        stack.push_back(arg);
      }
    });
  }

  return list;
}

/// Updates the values of all nodes in this graph based on the values of
/// their dependent nodes.
///
/// @tparam Scalar Scalar type.
/// @param list Topological sort of graph from parent to child.
template <typename Scalar>
void update_values(const ExpressionGraph<Scalar>& list) {
  // Traverse graph from child to parent and update values
  for (auto& node : list | std::views::reverse) {
    node->val = node->value();
  }
}

/// Updates the adjoints in the expression graph (computes the gradient) then
/// appends the adjoints of wrt to the sparse matrix triplets.
///
/// @tparam Scalar Scalar type.
/// @param top_list Topologically sorted graph from parent to child.
/// @param output_list Output row as column-node pairs.
/// @param triplets The sparse matrix triplets.
/// @param row The row of wrt.
template <typename Scalar>
void append_triplets(
    const ExpressionGraph<Scalar>& top_list,
    const gch::small_vector<std::pair<int, detail::Expression<Scalar>*>>&
        output_list,
    gch::small_vector<Eigen::Triplet<Scalar>>& triplets, int row) {
  // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
  // for background on reverse accumulation automatic differentiation.

  if (top_list.empty()) {
    return;
  }

  // Set root node's adjoint to 1 since df/df is 1
  top_list[0]->adjoint = Scalar(1);

  // Zero the rest of the adjoints
  for (auto& node : top_list | std::views::drop(1)) {
    node->adjoint = Scalar(0);
  }

  // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
  // multiplied by dy/dx. If there are multiple "paths" from the root node to
  // variable; the variable's adjoint is the sum of each path's adjoint
  // contribution.
  for (const auto& node : top_list) {
    node->accumulate_adjoints();
  }

  // Exploit the row's sparsity pattern by only appending wrt adjoints that
  // appear in the expression graph
  for (const auto& [col, node] : output_list) {
    // Append adjoints of wrt to sparse matrix triplets
    triplets.emplace_back(row, col, node->adjoint);
  }
}

}  // namespace slp::detail
