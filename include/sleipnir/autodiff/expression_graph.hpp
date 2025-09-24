// Copyright (c) Sleipnir contributors

#pragma once

#include <ranges>
#include <span>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_type.hpp"

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

/// Generates a topological sort of the union of several expression graphs from
/// parent to child.
///
/// Each node appears once, even if it's reachable from multiple roots. Null and
/// constant roots are skipped.
///
/// https://en.wikipedia.org/wiki/Topological_sorting
///
/// @tparam Scalar Scalar type.
/// @param roots The root nodes of the expressions.
template <typename Scalar>
ExpressionGraph<Scalar> topological_sort(
    std::span<Expression<Scalar>* const> roots) {
  ExpressionGraph<Scalar> list;

  auto is_root = [](const Expression<Scalar>* root) {
    return root != nullptr && root->type() != ExpressionType::CONSTANT;
  };

  // Stack of nodes to explore
  gch::small_vector<Expression<Scalar>*> stack;

  // Enumerate incoming edges for each node via depth-first search. The roots
  // are treated as children of a virtual root node, so a root that's also a
  // descendant of another root is only listed after all its parents.
  //
  // NOTE: scratch counts incoming edges, offset by -1 so -1 means no edges.
  for (auto root : roots) {
    if (is_root(root) && ++root->scratch == 0) {
      stack.emplace_back(root);
    }
  }
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

  // Generate topological sort of graph from parent to child, starting with the
  // virtual root node's edges.
  //
  // A node is only added to the stack after all its incoming edges have been
  // traversed. Expression::scratch is a decrementing counter for tracking this.
  for (auto root : roots) {
    if (is_root(root) && --root->scratch == -1) {
      stack.emplace_back(root);
    }
  }
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
    // Leaf nodes don't need updating
    if (!node->is_leaf) {
      node->val = node->value();
    }
  }
}

}  // namespace slp::detail
