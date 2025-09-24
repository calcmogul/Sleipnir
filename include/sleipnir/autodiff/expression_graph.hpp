// Copyright (c) Sleipnir contributors

#pragma once

#include <span>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_eval.hpp"
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

/// Returns true if the partial derivatives of an expression graph's operators
/// up to a given type depend on parameters.
///
/// @tparam Scalar Scalar type.
/// @param list Topological sort of graph from parent to child.
/// @param max_type The largest operator type to check.
/// @return True if the partial derivatives of an operator with a type of at
///     most max_type depend on parameters.
template <typename Scalar>
bool has_parameter_partials(const ExpressionGraph<Scalar>& list,
                            ExpressionType max_type) {
  using enum ExpressionType;

  for (const auto& node : list) {
    // Operators on only constants and parameters (e.g., p₁ * p₂ in
    // x + p₁ * p₂) are skipped because their partial derivatives are with
    // respect to parameters, which aren't decision variables.
    if (!node->is_leaf && node->type() >= LINEAR && node->type() <= max_type &&
        !constant_partials(*node)) {
      bool has_parameter_arg = false;
      node->visit_args([&](const auto& arg) {
        has_parameter_arg |= arg->type() == PARAMETER;
      });

      // A parameter's value can change between solves
      if (has_parameter_arg) {
        return true;
      }
    }
  }

  return false;
}

/// Returns true if the gradient of an expression graph is constant.
///
/// @tparam Scalar Scalar type.
/// @param list Topological sort of graph from parent to child.
/// @return True if the gradient of the expression graph is constant.
template <typename Scalar>
bool has_constant_gradient(const ExpressionGraph<Scalar>& list) {
  using enum ExpressionType;

  // The gradient of a constant or parameter expression is zero
  if (list.empty() || list[0]->type() < LINEAR) {
    return true;
  }

  // The gradient of a quadratic or nonlinear expression is never constant
  if (list[0]->type() > LINEAR) {
    return false;
  }

  // The gradient of a linear expression is composed of the partial derivatives
  // of its linear operators. A linear operator's partial derivatives can't
  // depend on decision variables, or the operator wouldn't be linear, so they
  // can only vary with parameters.
  return !has_parameter_partials(list, LINEAR);
}

/// Returns true if the Hessian of an expression graph is constant.
///
/// @tparam Scalar Scalar type.
/// @param list Topological sort of graph from parent to child.
/// @return True if the Hessian of the expression graph is constant.
template <typename Scalar>
bool has_constant_hessian(const ExpressionGraph<Scalar>& list) {
  using enum ExpressionType;

  // The Hessian of a constant, parameter, or linear expression is zero
  if (list.empty() || list[0]->type() < QUADRATIC) {
    return true;
  }

  // The Hessian of a nonlinear expression is never constant
  if (list[0]->type() > QUADRATIC) {
    return false;
  }

  // The Hessian of a quadratic expression is composed of the partial
  // derivatives of its linear and quadratic operators (a quadratic operator's
  // partial derivatives that vary with decision variables are only multiplied
  // by its constant second partial derivatives), so only parameters can change
  // it.
  return !has_parameter_partials(list, QUADRATIC);
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

}  // namespace slp::detail
