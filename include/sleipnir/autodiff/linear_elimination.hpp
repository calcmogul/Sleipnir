// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <cstddef>
#include <optional>
#include <span>
#include <utility>

#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/adjoint_tape.hpp"
#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_op.hpp"
#include "sleipnir/autodiff/expression_type.hpp"

namespace slp::detail {

/// A term of an eliminated node's value as a linear combination of nodes that
/// aren't eliminated.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct LinearTerm {
  /// The node's index in the topological list.
  uint32_t k;

  /// The node's coefficient.
  Scalar coeff;
};

/// An edge from a node to a child it pushes to.
///
/// Without linear node elimination, a node's children are its arguments. An
/// argument that was eliminated is replaced by its children, so the partial
/// derivative of the node's expression with respect to such a child is the
/// partial derivative with respect to the argument times the product of the
/// eliminated nodes's constant partial derivatives between them.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Edge {
  /// The child's index in the topological list.
  uint32_t k;

  /// The index of the argument of the node's expression the edge goes through.
  uint32_t arg;

  /// The product of the constant partial derivatives of the eliminated nodes
  /// between the argument and the child (1 if the child is the argument).
  Scalar scale;
};

/// Which second partial derivatives of each of an AdjointTape's nodes aren't
/// structurally zero.
///
/// They're stored flat, so a node with n arguments takes n(n + 1)/2 bytes
/// instead of MAX_ARGS² for the operator with the most arguments.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class HessStructures {
 public:
  /// Constructs an empty HessStructures.
  HessStructures() = default;

  /// Finds the second partial derivative structure of each node in a tape's
  /// nodes(). They're all zero for linear nodes, which get none.
  ///
  /// @param tape The tape.
  explicit HessStructures(const AdjointTape<Scalar>& tape) {
    const auto nodes = tape.nodes();
    m_first.reserve(nodes.size() + 1);
    m_first.push_back(0);
    for (const auto& node : nodes) {
      if (node.expr->type() > ExpressionType::LINEAR) {
        const auto H = hess_structure(*node.expr);
        for (size_t j = 0; j < node.num_args; ++j) {
          for (size_t k = 0; k <= j; ++k) {
            m_nonzero.push_back(H[j][k]);
          }
        }
      }
      m_first.push_back(static_cast<uint32_t>(m_nonzero.size()));
    }
  }

  /// Returns whether each second partial derivative ∂²ϕᵢ/∂aⱼ∂aₖ with j ≥ k of
  /// a node isn't structurally zero, at index j(j + 1)/2 + k, or an empty span
  /// if the node is linear.
  ///
  /// @param pos The node's position in the tape's nodes().
  /// @return The node's second partial derivative structure.
  std::span<const uint8_t> operator[](size_t pos) const {
    return {m_nonzero.data() + m_first[pos], m_first[pos + 1] - m_first[pos]};
  }

 private:
  /// Each node's first element in m_nonzero, indexed by position in nodes(),
  /// followed by m_nonzero's size.
  gch::small_vector<uint32_t> m_first;

  /// Each node's lower triangle of second partial derivatives (row-major),
  /// one if it isn't structurally zero and zero if it is.
  gch::small_vector<uint8_t> m_nonzero;
};

/// Linear nodes eliminated from an AdjointTape's expression graph before edge
/// pushing.
///
/// A node's partial derivatives are constant if it's linear in its children
/// with constant coefficients (see constant_partials()). Such a node has no
/// second partial derivatives, so it only pushes its Hessian entries to its
/// children, each scaled by a constant. Replacing each edge from a parent to
/// the node with edges from the parent to the node's children, scaled by the
/// node's partial derivatives, gives the same Hessian without storing entries
/// at the node.
///
/// Eliminating a node with p parent edges and c children replaces p + c edges
/// with pc edges, so only nodes with p ≤ 1 or c ≤ 1 are eliminated. Nodes that
/// no nonlinear node is an ancestor of are skipped since they never have
/// Hessian entries. Elements of wrt are never eliminated.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class LinearElimination {
 public:
  /// Constructs a LinearElimination that eliminates nothing.
  LinearElimination() = default;

  /// Chooses which linear nodes to eliminate from a tape's expression graph.
  ///
  /// @param tape The tape.
  /// @param hess_structures Which second partial derivatives of each node
  ///     aren't structurally zero.
  LinearElimination(const AdjointTape<Scalar>& tape,
                    const HessStructures<Scalar>& hess_structures);

  /// Returns whether the node at the given topological list index is
  /// eliminated.
  ///
  /// @param idx The node's index in the topological list.
  /// @return Whether the node is eliminated.
  bool eliminated(size_t idx) const {
    return m_expansion_idxs[idx] != NOT_ELIMINATED;
  }

  /// Returns the edges from a node that isn't eliminated to the children it
  /// pushes to, in argument order.
  ///
  /// @param children The node's children (see AdjointTape::children()).
  /// @param edges The edges.
  void get_edges(std::span<const uint32_t> children,
                 gch::small_vector<Edge<Scalar>>& edges) const {
    constexpr auto SKIPPED_CHILD = AdjointTape<Scalar>::SKIPPED_CHILD;

    edges.clear();
    for (size_t arg = 0; arg < children.size(); ++arg) {
      const uint32_t k = children[arg];
      if (k == SKIPPED_CHILD) {
        continue;
      }

      if (eliminated(k)) {
        for (const auto& term : expansion(k)) {
          edges.push_back({term.k, static_cast<uint32_t>(arg), term.coeff});
        }
      } else {
        edges.push_back({k, static_cast<uint32_t>(arg), Scalar(1)});
      }
    }
  }

 private:
  /// Marks a node that isn't eliminated in m_expansion_idxs.
  static constexpr uint32_t NOT_ELIMINATED = UINT32_MAX;

  /// Each node's index in m_expansions if it's eliminated, or NOT_ELIMINATED
  /// if it isn't, indexed by topological list index.
  gch::small_vector<uint32_t> m_expansion_idxs;

  /// Each eliminated node's value as a linear combination of nodes that aren't
  /// eliminated, without constant terms. Empty for eliminated nodes whose only
  /// parent consumed it.
  gch::small_vector<gch::small_vector<LinearTerm<Scalar>>> m_expansions;

  /// Returns the expansion of the eliminated node at the given topological
  /// list index.
  ///
  /// @param idx The node's index in the topological list.
  /// @return The node's expansion.
  gch::small_vector<LinearTerm<Scalar>>& expansion(size_t idx) {
    return m_expansions[m_expansion_idxs[idx]];
  }

  /// Returns the expansion of the eliminated node at the given topological
  /// list index.
  ///
  /// @param idx The node's index in the topological list.
  /// @return The node's expansion.
  const gch::small_vector<LinearTerm<Scalar>>& expansion(size_t idx) const {
    return m_expansions[m_expansion_idxs[idx]];
  }
};

template <typename Scalar>
LinearElimination<Scalar>::LinearElimination(
    const AdjointTape<Scalar>& tape,
    const HessStructures<Scalar>& hess_structures) {
  constexpr auto SKIPPED_CHILD = AdjointTape<Scalar>::SKIPPED_CHILD;

  const auto nodes = tape.nodes();
  const size_t num_nodes = tape.top_list().size();

  m_expansion_idxs.assign(num_nodes, NOT_ELIMINATED);

  // Returns whether the node at the given position in nodes() creates Hessian
  // entries, which requires a second partial derivative that isn't
  // structurally zero between children it propagates to
  auto creates = [&](size_t pos) {
    const auto nonzero = hess_structures[pos];
    if (nonzero.empty()) {
      return false;
    }

    const auto children = tape.children(nodes[pos]);
    const uint8_t* h = nonzero.data();
    for (size_t arg1 = 0; arg1 < children.size(); ++arg1) {
      for (size_t arg2 = 0; arg2 <= arg1; ++arg2) {
        if (children[arg1] != SKIPPED_CHILD &&
            children[arg2] != SKIPPED_CHILD && *h) {
          return true;
        }
        ++h;
      }
    }
    return false;
  };

  // Count each node's parent edges and find which nodes can have Hessian
  // entries. Entries are created by nonlinear nodes, then pushed to their
  // children, so a node can only have entries if a node that creates them is
  // its ancestor. Parents come first in topological order.
  gch::small_vector<uint32_t> num_parents(num_nodes);
  gch::small_vector<uint8_t> may_have_entries(num_nodes);
  for (size_t pos = 0; pos < nodes.size(); ++pos) {
    const auto& node = nodes[pos];
    const bool pushes = may_have_entries[node.idx] || creates(pos);
    for (auto k : tape.children(node)) {
      if (k != SKIPPED_CHILD) {
        ++num_parents[k];
        may_have_entries[k] = may_have_entries[k] || pushes;
      }
    }
  }

  // Visit children before parents so each node's children's expansions are
  // final
  for (size_t pos = nodes.size(); pos-- > 0;) {
    const auto& node = nodes[pos];
    const uint32_t i = node.idx;

    if (!may_have_entries[i] || tape.col(i) != -1) {
      continue;
    }

    auto g = constant_partials(*node.expr);
    if (!g) {
      continue;
    }

    const auto children = tape.children(node);

    // Count the node's children, with eliminated arguments replaced by theirs
    size_t num_children = 0;
    for (auto k : children) {
      if (k != SKIPPED_CHILD) {
        num_children += eliminated(k) ? expansion(k).size() : 1;
      }
    }
    if (num_parents[i] > 1 && num_children > 1) {
      continue;
    }

    m_expansion_idxs[i] = m_expansions.size();
    auto& node_expansion = m_expansions.emplace_back();

    // An eliminated argument with one parent edge isn't expanded anywhere
    // else, so the largest one's expansion is moved instead of copied. This
    // keeps long chains of sums from copying their terms at every node.
    size_t moved_arg = SIZE_MAX;
    for (size_t arg = 0; arg < children.size(); ++arg) {
      const uint32_t k = children[arg];
      if (k != SKIPPED_CHILD && eliminated(k) && num_parents[k] == 1 &&
          (moved_arg == SIZE_MAX ||
           expansion(k).size() > expansion(children[moved_arg]).size())) {
        moved_arg = arg;
      }
    }
    if (moved_arg != SIZE_MAX) {
      node_expansion = std::move(expansion(children[moved_arg]));
      if (const Scalar coeff = (*g)[moved_arg]; coeff != Scalar(1)) {
        for (auto& term : node_expansion) {
          term.coeff *= coeff;
        }
      }
    }

    for (size_t arg = 0; arg < children.size(); ++arg) {
      const uint32_t k = children[arg];
      if (k == SKIPPED_CHILD || arg == moved_arg) {
        continue;
      }

      const Scalar coeff = (*g)[arg];
      if (eliminated(k)) {
        for (const auto& term : expansion(k)) {
          node_expansion.push_back({term.k, coeff * term.coeff});
        }
      } else {
        node_expansion.push_back({k, coeff});
      }
    }
  }
}

}  // namespace slp::detail
