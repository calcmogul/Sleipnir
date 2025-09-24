// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <optional>
#include <span>
#include <type_traits>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/expression_type.hpp"

namespace slp::detail {

/// A sparse matrix entry whose value is read from a tape slot.
struct TapeEntry {
  /// The slot the entry's value is read from.
  uint32_t slot;

  /// The entry's row.
  int row;

  /// The entry's column.
  int col;
};

/// Maps a tape slot to an index in a sparse matrix's value array.
struct SlotMapEntry {
  /// The index of the slot the value is read from.
  uint32_t slot_idx;

  /// The index in the sparse matrix's value array.
  uint32_t value_idx;
};

/// Sets a sparse matrix's sparsity pattern to the given entries and returns
/// where in its value array each entry is stored.
///
/// @tparam Scalar Scalar type.
/// @param mat The sparse matrix.
/// @param entries The sparse matrix entries.
/// @return The slot map entry for each entry, in the same order as entries.
template <typename Scalar>
gch::small_vector<SlotMapEntry> make_slot_map(
    Eigen::SparseMatrix<Scalar>& mat, std::span<const TapeEntry> entries) {
  // Set sparsity pattern
  gch::small_vector<Eigen::Triplet<Scalar>> triplets;
  for (const auto& entry : entries) {
    triplets.emplace_back(entry.row, entry.col, Scalar(0));
  }
  mat.setFromTriplets(triplets.begin(), triplets.end());

  // Generate map from tape slot to value array position
  gch::small_vector<SlotMapEntry> slot_map;
  for (const auto& entry : entries) {
    slot_map.emplace_back(
        entry.slot, static_cast<uint32_t>(&mat.coeffRef(entry.row, entry.col) -
                                          mat.valuePtr()));
  }
  return slot_map;
}

/// Writes slot values into a sparse matrix's value array.
///
/// @tparam Scalar Scalar type.
/// @param m The sparse matrix.
/// @param slot_map Map from slot to value array index.
/// @param slots The slot values.
template <typename Scalar>
void copy_slots(Eigen::SparseMatrix<Scalar>& m,
                std::span<const SlotMapEntry> slot_map,
                std::type_identity_t<std::span<const Scalar>> slots) {
  auto values = m.valuePtr();
  for (const auto& entry : slot_map) {
    values[entry.value_idx] = slots[entry.slot_idx];
  }
}

/// Records reverse accumulation of the adjoints of several roots, then replays
/// it with the expression graph's current values.
///
/// Which adjoints are nonzero only depends on the expression graph's structure,
/// so the recording gives each (node, root) pair that can have a nonzero
/// adjoint a slot, and each adjoint contribution an operation between slots.
/// Replaying evaluates each node's partial derivatives once regardless of how
/// many roots reach it.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class AdjointTape {
 public:
  /// Marks a child that adjoint propagation skips because it's inactive or its
  /// partial derivative is structurally zero.
  static constexpr uint32_t SKIPPED_CHILD = UINT32_MAX;

  /// A node that propagates adjoints to its children.
  struct Node {
    /// The node's expression.
    Expression<Scalar>* expr;

    /// The node's index in the topological list.
    uint32_t idx;

    /// The node's first adjoint slot. The node has one adjoint slot per root
    /// that reaches it.
    uint32_t slot;

    /// The number of the node's adjoint slots.
    uint32_t num_slots;

    /// Each child's index in the topological list, or SKIPPED_CHILD if the
    /// child is inactive or its partial derivative is structurally zero.
    ArgArray<uint32_t> children;

    /// The argument positions of the children that aren't SKIPPED_CHILD.
    ArgArray<uint8_t> propagated_args;

    /// The number of argument positions in propagated_args.
    uint8_t num_propagated_args;
  };

  /// Constructs an empty AdjointTape.
  AdjointTape() = default;

  /// Constructs an AdjointTape.
  ///
  /// The graph's structure must not change while the AdjointTape is in use.
  ///
  /// @param roots The roots whose adjoints to record. Null and constant roots
  ///     are skipped.
  /// @param wrt The leaves with respect to which to record adjoints.
  AdjointTape(std::span<Expression<Scalar>* const> roots,
              std::span<Expression<Scalar>* const> wrt) {
    using enum ExpressionType;

    m_top_list = topological_sort(roots);

    // Sort dependent variables before independent ones while maintaining
    // relative order. Leaves have no children, so this is still a topological
    // sort. Edge pushing requires it.
    std::ranges::stable_partition(
        m_top_list, [](const auto& elem) { return !elem->is_leaf; });

    // Temporarily store each node's index in m_top_list in scratch
    for (size_t i = 0; i < m_top_list.size(); ++i) {
      m_top_list[i]->scratch = i;
    }

    // Map each node's index in m_top_list to its column in wrt, or -1 if it
    // isn't an element of wrt
    m_cols.assign(m_top_list.size(), -1);
    for (size_t col = 0; col < wrt.size(); ++col) {
      if (wrt[col]->scratch != -1) {
        m_cols[wrt[col]->scratch] = col;
      }
    }

    // A node is active if it's an element of wrt or depends on one. Inactive
    // nodes have zero derivatives with respect to wrt.
    gch::small_vector<uint8_t> active(m_top_list.size());
    for (int i = static_cast<int>(m_top_list.size()) - 1; i >= 0; --i) {
      const auto& node = m_top_list[i];
      active[i] = node->type() != CONSTANT && m_cols[i] != -1;
      if (!node->is_leaf) {
        node->visit_args(
            [&](auto* arg) { active[i] = active[i] || active[arg->scratch]; });
      }
    }

    // Find which roots reach each node, starting with each active root reaching
    // its own node
    gch::small_vector<gch::small_vector<uint32_t>> node_roots(
        m_top_list.size());
    gch::small_vector<ArgArray<uint32_t>> node_children(m_top_list.size());
    for (size_t root_idx = 0; root_idx < roots.size(); ++root_idx) {
      const auto& root = roots[root_idx];
      if (root != nullptr && root->type() != CONSTANT &&
          active[root->scratch]) {
        node_roots[root->scratch].push_back(root_idx);
      }
    }

    // Push each node's roots down to its active children with nonzero partial
    // derivatives. Parents come first in topological order, so by the time
    // node i is visited, its root list is complete.
    for (size_t i = 0; i < m_top_list.size(); ++i) {
      const auto& current_node = m_top_list[i];

      // Sort and deduplicate the roots. Each parent appends its roots, so a
      // root that reaches the node through more than one parent shows up more
      // than once. get_slot() also relies on the sorted order.
      auto& root_idxs = node_roots[i];
      std::ranges::sort(root_idxs);
      root_idxs.erase(std::unique(root_idxs.begin(), root_idxs.end()),
                      root_idxs.end());

      // Skip inactive or leaf nodes
      if (!active[i] || current_node->is_leaf) {
        continue;
      }

      // Cache the children for recording the adjoint operations later
      node_children[i] = propagating_children(current_node, active);

      // Roots that reach this node also reach its children
      for (auto arg : node_children[i]) {
        if (arg != SKIPPED_CHILD) {
          node_roots[arg].insert(node_roots[arg].end(), root_idxs.begin(),
                                 root_idxs.end());
        }
      }
    }

    // Allocate each node's adjoint slots contiguously in topological order
    gch::small_vector<uint32_t> slot_begin(m_top_list.size() + 1);
    for (size_t i = 0; i < m_top_list.size(); ++i) {
      slot_begin[i + 1] = slot_begin[i] + node_roots[i].size();
    }
    m_adjoints.resize(slot_begin[m_top_list.size()]);

    auto get_slot = [&](size_t i, uint32_t root_idx) -> uint32_t {
      const auto& root_idxs = node_roots[i];
      return slot_begin[i] + (std::ranges::lower_bound(root_idxs, root_idx) -
                              root_idxs.begin());
    };

    for (size_t root_idx = 0; root_idx < roots.size(); ++root_idx) {
      const auto& root = roots[root_idx];
      if (root != nullptr && root->type() != CONSTANT &&
          active[root->scratch]) {
        m_seeds.push_back(get_slot(root->scratch, root_idx));
      }
    }

    // Record adjoint operations. Each of a node's adjoint slots propagates to
    // the same children, so only the destination slots are recorded.
    //
    // a(vₖ) += ∂ϕᵢ/∂vₖ a(vᵢ)
    for (size_t i = 0; i < m_top_list.size(); ++i) {
      const auto& v_i = m_top_list[i];

      // Nodes no root reaches through nonzero partial derivatives have zero
      // adjoints, so they have nothing to propagate
      if (!active[i] || v_i->is_leaf || node_roots[i].empty()) {
        continue;
      }

      const auto& children = node_children[i];

      ArgArray<uint8_t> propagated_args;
      uint8_t num_propagated_args = 0;
      for (size_t k = 0; k < children.size(); ++k) {
        if (children[k] != SKIPPED_CHILD) {
          propagated_args[num_propagated_args++] = k;
        }
      }

      for (auto root_idx : node_roots[i]) {
        for (size_t propagated_idx = 0; propagated_idx < num_propagated_args;
             ++propagated_idx) {
          m_dests.push_back(
              get_slot(children[propagated_args[propagated_idx]], root_idx));
        }
      }

      m_nodes.push_back({v_i, static_cast<uint32_t>(i), slot_begin[i],
                         static_cast<uint32_t>(node_roots[i].size()), children,
                         propagated_args, num_propagated_args});
    }

    // Record which slots hold the adjoints of wrt
    for (size_t i = 0; i < m_top_list.size(); ++i) {
      if (m_cols[i] == -1) {
        continue;
      }
      for (size_t slot_offset = 0; slot_offset < node_roots[i].size();
           ++slot_offset) {
        m_entries.push_back({static_cast<uint32_t>(slot_begin[i] + slot_offset),
                             static_cast<int>(node_roots[i][slot_offset]),
                             m_cols[i]});
      }
    }

    for (auto& node : m_top_list) {
      node->scratch = -1;
    }
  }

  /// Returns the topologically sorted union of the roots's expression graphs,
  /// with all non-leaf nodes before all leaf nodes.
  ///
  /// @return The topologically sorted union of the roots's expression graphs.
  const ExpressionGraph<Scalar>& top_list() const { return m_top_list; }

  /// Returns the nodes that propagate adjoints, in topological order.
  ///
  /// @return The nodes that propagate adjoints, in topological order.
  std::span<const Node> nodes() const { return m_nodes; }

  /// Returns the column of the node at the given index in the topological
  /// list, or -1 if the node isn't an element of wrt.
  ///
  /// @param idx The node's index in the topological list.
  /// @return The node's column.
  int col(size_t idx) const { return m_cols[idx]; }

  /// Returns which slot holds each nonzero adjoint of wrt as a sparse matrix
  /// entry. The row is the root's index and the column is the index in wrt.
  ///
  /// @return Which slot holds each nonzero adjoint of wrt.
  std::span<const TapeEntry> entries() const { return m_entries; }

  /// Returns the adjoint slot values.
  ///
  /// @return The adjoint slot values.
  std::span<const Scalar> adjoints() const { return m_adjoints; }

  /// Updates the values of the expression graph's nodes.
  void update_values() { detail::update_values(m_top_list); }

  /// Computes the adjoints with the expression graph's current values.
  ///
  /// @param f Called with each node's index in nodes(), the node, and the
  ///     node's partial derivatives after its adjoints are propagated.
  template <typename F>
    requires std::invocable<F&, size_t, const Node&,
                            const ArgArray<std::optional<Scalar>>&>
  void replay(F&& f) {
    std::ranges::fill(m_adjoints, Scalar(0));

    // Set each root's adjoint to 1 since df/df is 1
    for (auto seed : m_seeds) {
      m_adjoints[seed] = Scalar(1);
    }

    const uint32_t* dest = m_dests.data();
    for (size_t node_idx = 0; node_idx < m_nodes.size(); ++node_idx) {
      const auto& node = m_nodes[node_idx];

      ArgArray<std::optional<Scalar>> g;
      node.expr->grad(g);

      ArgArray<Scalar> g_propagated;
      for (size_t propagated_idx = 0; propagated_idx < node.num_propagated_args;
           ++propagated_idx) {
        g_propagated[propagated_idx] = *g[node.propagated_args[propagated_idx]];
      }

      // A node's adjoint slots never alias its children's slots, so each
      // adjoint is final before it's propagated
      for (uint32_t slot_offset = 0; slot_offset < node.num_slots;
           ++slot_offset) {
        Scalar adjoint = m_adjoints[node.slot + slot_offset];
        for (size_t propagated_idx = 0;
             propagated_idx < node.num_propagated_args; ++propagated_idx) {
          m_adjoints[*dest++] += g_propagated[propagated_idx] * adjoint;
        }
      }

      f(node_idx, node, g);
    }
  }

  /// Computes the adjoints with the expression graph's current values.
  void replay() {
    replay([](size_t, const Node&, const ArgArray<std::optional<Scalar>>&) {});
  }

 private:
  /// The topologically sorted union of the roots's expression graphs, with all
  /// non-leaf nodes before all leaf nodes.
  ExpressionGraph<Scalar> m_top_list;

  /// Each node's column in wrt, indexed by topological list index, or -1 if
  /// the node isn't an element of wrt.
  gch::small_vector<int> m_cols;

  /// The nodes that propagate adjoints, in topological order.
  gch::small_vector<Node> m_nodes;

  /// The destination adjoint slot of each adjoint operation, in replay order.
  /// Each node contributes num_slots × num_propagated_args entries.
  gch::small_vector<uint32_t> m_dests;

  /// The adjoint slot of each active root's own adjoint, which replay() seeds
  /// with 1.
  gch::small_vector<uint32_t> m_seeds;

  /// Which slot holds each nonzero adjoint of wrt as a sparse matrix entry.
  gch::small_vector<TapeEntry> m_entries;

  /// The adjoint slot values.
  gch::small_vector<Scalar> m_adjoints;

  /// Returns the children the node propagates adjoints to. Each element is the
  /// child's index in the topological list, or SKIPPED_CHILD if the child is
  /// inactive or its partial derivative is structurally zero. Requires scratch
  /// to hold topological list indices.
  ///
  /// Whether a partial derivative is zero only depends on the expression type,
  /// not child values.
  static ArgArray<uint32_t> propagating_children(
      Expression<Scalar>* node, std::span<const uint8_t> active) {
    ArgArray<uint32_t> children;
    children.fill(SKIPPED_CHILD);

    ArgArray<std::optional<Scalar>> g;
    node->grad(g);

    size_t k = 0;
    node->visit_args([&](auto* arg) {
      if (g[k].has_value() && active[arg->scratch]) {
        children[k] = arg->scratch;
      }
      ++k;
    });

    return children;
  }
};

}  // namespace slp::detail
