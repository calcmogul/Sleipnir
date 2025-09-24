// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <cstddef>
#include <span>
#include <utility>

#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/adjoint_tape.hpp"
#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/hess_structures.hpp"
#include "sleipnir/autodiff/vertex_elimination.hpp"

namespace slp::detail {

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
/// This is vertex elimination on the linearized graph with
/// ConstantEliminationPolicy (see vertex_elimination.hpp). Nodes are decided
/// from child to parent, so a node's in-edges p are counted before any
/// elimination and its out-edges c after its children's. Nodes that no
/// nonlinear node is an ancestor of are skipped since they never have Hessian
/// entries. Elements of wrt are never eliminated.
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
          edges.push_back({term.k, static_cast<uint32_t>(arg), term.scale});
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
  gch::small_vector<gch::small_vector<ScaledChild<Scalar>>> m_expansions;

  /// Returns the expansion of the eliminated node at the given topological
  /// list index.
  ///
  /// @param idx The node's index in the topological list.
  /// @return The node's expansion.
  gch::small_vector<ScaledChild<Scalar>>& expansion(size_t idx) {
    return m_expansions[m_expansion_idxs[idx]];
  }

  /// Returns the expansion of the eliminated node at the given topological
  /// list index.
  ///
  /// @param idx The node's index in the topological list.
  /// @return The node's expansion.
  const gch::small_vector<ScaledChild<Scalar>>& expansion(size_t idx) const {
    return m_expansions[m_expansion_idxs[idx]];
  }
};

template <typename Scalar>
LinearElimination<Scalar>::LinearElimination(
    const AdjointTape<Scalar>& tape,
    const HessStructures<Scalar>& hess_structures) {
  constexpr auto SKIPPED_CHILD = AdjointTape<Scalar>::SKIPPED_CHILD;

  using Policy = ConstantEliminationPolicy<Scalar>;

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

    auto g = Policy::weights(*node.expr);
    if (!g) {
      continue;
    }

    const auto children = tape.children(node);

    // Count the node's out-edges in the reduced graph, where eliminated
    // arguments are replaced by their expansions
    uint32_t num_children = 0;
    for (auto k : children) {
      if (k != SKIPPED_CHILD) {
        num_children += eliminated(k) ? expansion(k).size() : 1;
      }
    }
    if (!Policy::accepts(num_parents[i], num_children)) {
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
          term.scale *= coeff;
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
          node_expansion.push_back({term.k, coeff * term.scale});
        }
      } else {
        node_expansion.push_back({k, coeff});
      }
    }
  }
}

}  // namespace slp::detail
