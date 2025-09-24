// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/util/flat_map.hpp"

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
/// The tape stores the expression graph as flat arrays: each non-leaf node's
/// operator and argument indices, and each node's value. update_values()
/// evaluates each node's value and partial derivatives from the leaves's values
/// without traversing the expression graph or making virtual calls.
///
/// Which adjoints are nonzero only depends on the expression graph's structure,
/// so the recording gives each (node, root) pair that can have a nonzero
/// adjoint a slot, and each adjoint contribution an operation between slots.
/// replay() then propagates adjoints with those operations, scaled by the
/// partial derivatives from update_values(), so each node's partial derivatives
/// are evaluated once regardless of how many roots reach it.
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

    /// The index of ∂ϕᵢ/∂a₀ in partials(), where a₀ is the node's first
    /// argument. ∂ϕᵢ/∂aₖ is at index first_partial + k.
    uint32_t first_partial;

    /// The index of ∂²ϕᵢ/∂a₀∂a₀ in second_partials(). ∂²ϕᵢ/∂aⱼ∂aₖ with j ≥ k
    /// is at index first_second_partial + j(j + 1)/2 + k.
    uint32_t first_second_partial;

    /// Each child's index in the topological list, or SKIPPED_CHILD if the
    /// child is inactive or its partial derivative is structurally zero.
    ArgArray<uint32_t> children;
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
    m_num_non_leaves =
        std::ranges::find_if(m_top_list,
                             [](const auto& elem) { return elem->is_leaf; }) -
        m_top_list.begin();

    // Temporarily store each node's index in m_top_list in scratch
    for (size_t i = 0; i < m_top_list.size(); ++i) {
      m_top_list[i]->scratch = i;
    }

    // Record where each root's value is
    for (auto root : roots) {
      m_roots.emplace_back(root);
      m_root_idxs.emplace_back(root != nullptr && root->scratch != -1
                                   ? static_cast<uint32_t>(root->scratch)
                                   : NOT_IN_TAPE);
    }

    // Record each non-leaf node's operator and arguments, and where its
    // partial derivatives are stored. Each argument has one first partial
    // derivative, so those are stored at the argument's index in m_args.
    uint32_t num_second_partials = 0;
    m_ops.reserve(m_num_non_leaves);
    m_first_partials.reserve(m_num_non_leaves);
    m_first_second_partials.reserve(m_num_non_leaves);
    m_args.reserve(2 * m_num_non_leaves);
    for (size_t i = 0; i < m_num_non_leaves; ++i) {
      auto op = m_top_list[i]->op();
      m_ops.emplace_back(op);
      m_first_partials.emplace_back(m_args.size());
      m_first_second_partials.emplace_back(num_second_partials);
      m_top_list[i]->visit_args(
          [&](auto* arg) { m_args.emplace_back(arg->scratch); });
      num_second_partials += visit_op<Scalar>(
          op, []<typename T>() { return NUM_SECOND_PARTIALS<T>; });
    }
    m_values.resize(m_top_list.size());
    m_partials.resize(m_args.size());
    m_num_second_partials = num_second_partials;

    // Without wrt, there are no adjoints to record
    if (wrt.empty()) {
      for (auto& node : m_top_list) {
        node->scratch = -1;
      }
      return;
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

    // Allocate adjoint slots. The adjoints of wrt are the results, so their
    // slots are allocated first and never reused. Every other node's slots are
    // allocated when they're first written and freed once the node has
    // propagated its adjoints, so the slots in use at any time stay few and
    // cache-resident. Freed slots are reused most recently freed first.
    constexpr uint32_t UNALLOCATED = UINT32_MAX;
    gch::small_vector<uint32_t> slot_begin(m_top_list.size(), UNALLOCATED);
    uint32_t num_slots = 0;
    for (size_t i = 0; i < m_top_list.size(); ++i) {
      if (m_cols[i] != -1) {
        slot_begin[i] = num_slots;
        num_slots += node_roots[i].size();
      }
    }
    m_num_result_slots = num_slots;

    // Maps a block size to the first slots of the free blocks of that size
    flat_map<uint32_t, gch::small_vector<uint32_t>> free_blocks;
    auto allocate = [&](size_t i) {
      if (slot_begin[i] != UNALLOCATED) {
        return;
      }

      auto& blocks = free_blocks[node_roots[i].size()];
      if (!blocks.empty()) {
        slot_begin[i] = blocks.back();
        blocks.pop_back();
      } else {
        slot_begin[i] = num_slots;
        num_slots += node_roots[i].size();
      }
    };

    auto get_slot = [&](size_t i, uint32_t root_idx) -> uint32_t {
      allocate(i);
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

      uint32_t num_propagated_args = 0;
      for (size_t k = 0; k < children.size(); ++k) {
        if (children[k] != SKIPPED_CHILD) {
          m_propagated_partials.push_back(m_first_partials[i] + k);
          ++num_propagated_args;
        }
      }

      for (auto root_idx : node_roots[i]) {
        for (size_t k = 0; k < children.size(); ++k) {
          if (children[k] != SKIPPED_CHILD) {
            m_dests.push_back(get_slot(children[k], root_idx));
          }
        }
      }

      m_nodes.push_back({v_i, static_cast<uint32_t>(i), m_first_partials[i],
                         m_first_second_partials[i], children});
      m_instructions.push_back({slot_begin[i],
                                static_cast<uint32_t>(node_roots[i].size()),
                                num_propagated_args});

      // The node's adjoints are dead once they're propagated
      if (m_cols[i] == -1) {
        free_blocks[node_roots[i].size()].push_back(slot_begin[i]);
      }
    }

    m_adjoints.resize(num_slots);

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

  /// Returns a root's value from the last update_values() call.
  ///
  /// @param root_idx The root's index in the roots passed to the constructor.
  /// @return The root's value, or zero if the root is null.
  Scalar root_value(size_t root_idx) const {
    if (uint32_t idx = m_root_idxs[root_idx]; idx != NOT_IN_TAPE) {
      return m_values[idx];
    }

    // Constant roots aren't in the tape since their values never change
    const auto& root = m_roots[root_idx];
    return root != nullptr ? root->val : Scalar(0);
  }

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
  /// After replay(), only the slots of entries() hold adjoints. The other slots
  /// are reused during replay() and zeroed after their last use.
  ///
  /// @return The adjoint slot values.
  std::span<const Scalar> adjoints() const { return m_adjoints; }

  /// Returns the non-leaf nodes's partial derivatives from the last
  /// update_values() call with ORDER ≥ 1. Node::first_partial says where a
  /// node's are. Structurally zero partial derivatives are zero.
  ///
  /// @return The non-leaf nodes's partial derivatives.
  std::span<const Scalar> partials() const { return m_partials; }

  /// Returns the non-leaf nodes's second partial derivatives from the last
  /// update_values<2>() call. Node::first_second_partial says where a
  /// node's are. Structurally zero second partial derivatives are zero.
  ///
  /// @return The non-leaf nodes's second partial derivatives.
  std::span<const Scalar> second_partials() const { return m_second_partials; }

  /// Evaluates each node's value and each non-leaf node's partial derivatives
  /// up to the given order from the leaves's current values.
  ///
  /// The values are stored in the tape instead of the expression graph's
  /// nodes, which are left unmodified.
  ///
  /// @tparam ORDER The highest order of partial derivatives to evaluate: 0 for
  ///     only values, 1 for partial derivatives (needed by replay()), or 2 for
  ///     second partial derivatives too.
  template <int ORDER = 1>
    requires(ORDER >= 0 && ORDER <= 2)
  void update_values() {
    if constexpr (ORDER >= 2) {
      m_second_partials.resize(m_num_second_partials);
    }

    // Read the leaves's values
    for (size_t i = m_num_non_leaves; i < m_top_list.size(); ++i) {
      m_values[i] = m_top_list[i]->val;
    }

    // Evaluate non-leaf nodes from child to parent.
    //
    // The loop index is captured by value because GCC keeps a variable
    // captured by reference on the stack, which would chain every iteration
    // through a store and load of the index.
    for (size_t i = m_num_non_leaves; i-- > 0;) {
      visit_op<Scalar>(m_ops[i], [&, i]<typename T>() {
        const auto& expr = *m_top_list[i];
        const uint32_t first_partial = m_first_partials[i];

        ArgArray<Scalar> x{};
        for (size_t k = 0; k < NUM_ARGS<T>; ++k) {
          x[k] = m_values[m_args[first_partial + k]];
        }

        Scalar val = op_value<T>(expr, x);
        m_values[i] = val;

        if constexpr (ORDER >= 1) {
          ArgArray<Scalar> g{};
          op_grad<T>(expr, x, val, g);
          for (size_t k = 0; k < NUM_ARGS<T>; ++k) {
            m_partials[first_partial + k] = g[k];
          }
        }

        if constexpr (ORDER >= 2 && NUM_SECOND_PARTIALS<T> > 0) {
          ArgArray<ArgArray<Scalar>> H{};
          op_hess<T>(expr, x, val, H);
          Scalar* h = &m_second_partials[m_first_second_partials[i]];
          for (size_t j = 0; j < NUM_ARGS<T>; ++j) {
            for (size_t k = 0; k <= j; ++k) {
              *h++ = H[j][k];
            }
          }
        }
      });
    }
  }

  /// Computes the adjoints with the partial derivatives from the last
  /// update_values() call.
  ///
  /// @param f Called with each node's index in nodes() and the node's adjoints
  ///     (one per root that reaches it) after its adjoints are propagated.
  template <typename F>
    requires std::invocable<F&, size_t, std::span<const Scalar>>
  void replay(F&& f) {
    // Slots other than result slots are zeroed after their last use, so only
    // the result slots need to be zeroed here
    std::fill_n(m_adjoints.begin(), m_num_result_slots, Scalar(0));

    // Set each root's adjoint to 1 since df/df is 1
    for (auto seed : m_seeds) {
      m_adjoints[seed] = Scalar(1);
    }

    const uint32_t* dest = m_dests.data();
    const uint32_t* propagated_partial = m_propagated_partials.data();
    for (size_t node_idx = 0; node_idx < m_instructions.size(); ++node_idx) {
      const auto& instruction = m_instructions[node_idx];
      const uint32_t slot = instruction.slot;
      const uint32_t num_slots = instruction.num_slots;
      const uint32_t num_propagated_args = instruction.num_propagated_args;

      // A node's adjoint slots never alias its children's slots, so each
      // adjoint is final before it's propagated.
      //
      // Most nodes are reached by one root (all of them for a gradient or
      // Hessian), so that case avoids the loop over slots.
      if (num_slots == 1) {
        Scalar adjoint = m_adjoints[slot];
        for (uint32_t k = 0; k < num_propagated_args; ++k) {
          m_adjoints[dest[k]] += m_partials[propagated_partial[k]] * adjoint;
        }
        dest += num_propagated_args;
      } else {
        for (uint32_t slot_offset = 0; slot_offset < num_slots; ++slot_offset) {
          Scalar adjoint = m_adjoints[slot + slot_offset];
          for (uint32_t k = 0; k < num_propagated_args; ++k) {
            m_adjoints[dest[k]] += m_partials[propagated_partial[k]] * adjoint;
          }
          dest += num_propagated_args;
        }
      }
      propagated_partial += num_propagated_args;

      f(node_idx, std::span<const Scalar>{m_adjoints}.subspan(slot, num_slots));

      // Zero the node's dead adjoint slots so they start at zero when reused
      if (slot >= m_num_result_slots) {
        if (num_slots == 1) {
          m_adjoints[slot] = Scalar(0);
        } else {
          std::fill_n(m_adjoints.begin() + slot, num_slots, Scalar(0));
        }
      }
    }
  }

  /// Computes the adjoints with the partial derivatives from the last
  /// update_values() call.
  void replay() {
    replay([](size_t, std::span<const Scalar>) {});
  }

 private:
  /// Marks a root that isn't in the tape because it's null or constant.
  static constexpr uint32_t NOT_IN_TAPE = UINT32_MAX;

  /// A node's adjoint propagation.
  struct Instruction {
    /// The node's first adjoint slot. The node has one adjoint slot per root
    /// that reaches it.
    uint32_t slot;

    /// The number of the node's adjoint slots.
    uint32_t num_slots;

    /// The number of the node's arguments that are active and have partial
    /// derivatives that aren't structurally zero.
    uint32_t num_propagated_args;
  };

  /// The topologically sorted union of the roots's expression graphs, with all
  /// non-leaf nodes before all leaf nodes.
  ExpressionGraph<Scalar> m_top_list;

  /// The roots passed to the constructor.
  gch::small_vector<Expression<Scalar>*> m_roots;

  /// Each root's index in m_top_list, or NOT_IN_TAPE if it's null or constant.
  gch::small_vector<uint32_t> m_root_idxs;

  /// The number of non-leaf nodes at the front of m_top_list.
  size_t m_num_non_leaves = 0;

  /// Each non-leaf node's operator, indexed by topological list index.
  gch::small_vector<ExpressionOp> m_ops;

  /// Each non-leaf node's arguments's topological list indices, in
  /// topological order of the nodes and visit_args() order of the arguments.
  gch::small_vector<uint32_t> m_args;

  /// Each non-leaf node's first argument's index in m_args and m_partials,
  /// indexed by topological list index.
  gch::small_vector<uint32_t> m_first_partials;

  /// Each non-leaf node's first second partial derivative's index in
  /// m_second_partials, indexed by topological list index.
  gch::small_vector<uint32_t> m_first_second_partials;

  /// The total number of second partial derivatives.
  uint32_t m_num_second_partials = 0;

  /// Each node's value, indexed by topological list index.
  gch::small_vector<Scalar> m_values;

  /// Each non-leaf node's partial derivatives with respect to its arguments,
  /// laid out like m_args.
  gch::small_vector<Scalar> m_partials;

  /// Each non-leaf node's second partial derivatives with respect to its
  /// arguments (lower triangle, row-major). Only allocated once
  /// update_values<2>() is called.
  gch::small_vector<Scalar> m_second_partials;

  /// Each node's column in wrt, indexed by topological list index, or -1 if
  /// the node isn't an element of wrt.
  gch::small_vector<int> m_cols;

  /// The nodes that propagate adjoints, in topological order.
  gch::small_vector<Node> m_nodes;

  /// Each node's adjoint propagation, in the same order as m_nodes.
  gch::small_vector<Instruction> m_instructions;

  /// The index in m_partials of each propagated argument's partial
  /// derivative, in replay order. Each node contributes num_propagated_args
  /// entries.
  gch::small_vector<uint32_t> m_propagated_partials;

  /// The destination adjoint slot of each adjoint operation, in replay order.
  /// Each node contributes num_slots × num_propagated_args entries.
  gch::small_vector<uint32_t> m_dests;

  /// The adjoint slot of each active root's own adjoint, which replay() seeds
  /// with 1.
  gch::small_vector<uint32_t> m_seeds;

  /// Which slot holds each nonzero adjoint of wrt as a sparse matrix entry.
  gch::small_vector<TapeEntry> m_entries;

  /// The adjoint slot values.
  ///
  /// The first m_num_result_slots slots hold the adjoints of wrt. The other
  /// slots are reused by nodes whose adjoints are live at different times, and
  /// they're zero outside of replay().
  gch::small_vector<Scalar> m_adjoints;

  /// The number of slots at the front of m_adjoints that hold the adjoints of
  /// wrt.
  uint32_t m_num_result_slots = 0;

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

    auto nonzero = grad_structure(*node);

    size_t k = 0;
    node->visit_args([&](auto* arg) {
      if (nonzero[k] && active[arg->scratch]) {
        children[k] = arg->scratch;
      }
      ++k;
    });

    return children;
  }
};

}  // namespace slp::detail
