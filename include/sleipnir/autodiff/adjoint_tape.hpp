// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/expression_op.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/util/assert.hpp"
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
    /// is at index first_second_partial + j(j + 1)/2 + k. Only meaningful if
    /// the node's second partial derivatives aren't constant (see
    /// HAS_CONSTANT_SECOND_PARTIALS).
    uint32_t first_second_partial;

    /// Each child's index in the topological list, or SKIPPED_CHILD if the
    /// child is inactive or its partial derivative is structurally zero.
    ArgArray<uint32_t> children;

    /// The index of the node's adjoint propagation among the tape's
    /// instructions, or of the last one before it if the node was eliminated
    /// from the adjoint operations. replay() calls back after this
    /// instruction for nodes whose index is in its callbacks.
    uint32_t instruction;
  };

  /// Terminates the callback instruction indices passed to replay().
  static constexpr uint32_t END_CALLBACKS = UINT32_MAX;

  /// Constructs an empty AdjointTape.
  AdjointTape() = default;

  /// Constructs an AdjointTape that only evaluates values.
  ///
  /// This records less than AdjointTape(roots, {}), so it's faster to
  /// construct, but only update_values<0>() and root_value() may be called.
  ///
  /// The graph's structure must not change while the AdjointTape is in use.
  ///
  /// @param roots The roots whose values to record. Null and constant roots
  ///     are skipped.
  explicit AdjointTape(std::span<Expression<Scalar>* const> roots)
      : AdjointTape{roots, {}, true} {}

  /// Constructs an AdjointTape.
  ///
  /// The graph's structure must not change while the AdjointTape is in use.
  ///
  /// @param roots The roots whose adjoints to record. Null and constant roots
  ///     are skipped.
  /// @param wrt The leaves with respect to which to record adjoints.
  AdjointTape(std::span<Expression<Scalar>* const> roots,
              std::span<Expression<Scalar>* const> wrt)
      : AdjointTape{roots, wrt, false} {}

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
  /// This includes the nodes eliminated from the adjoint operations, which
  /// have unit partial derivatives (see HAS_UNIT_PARTIALS). Their parents
  /// propagate past them to their children with their own partial
  /// derivatives or their negations, so they have no adjoints.
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
  /// node's are. Structurally zero partial derivatives are zero. They're
  /// followed by the negations of the ones that adjoint operations through
  /// eliminated nodes need (see nodes()).
  ///
  /// @return The non-leaf nodes's partial derivatives.
  std::span<const Scalar> partials() const { return m_partials; }

  /// Returns the non-leaf nodes's second partial derivatives from the last
  /// update_values<2>() call. Node::first_second_partial says where a
  /// node's are. Structurally zero second partial derivatives are zero.
  /// Constant ones (see HAS_CONSTANT_SECOND_PARTIALS) aren't stored.
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
        for (size_t k = 0; k < T::NUM_ARGS; ++k) {
          x[k] = m_values[m_args[first_partial + k]];
        }

        Scalar val = op_value<T>(expr, x);
        m_values[i] = val;

        if constexpr (ORDER >= 1) {
          ArgArray<Scalar> g{};
          op_grad<T>(expr, x, val, g);
          for (size_t k = 0; k < T::NUM_ARGS; ++k) {
            m_partials[first_partial + k] = g[k];
          }
        }

        if constexpr (ORDER >= 2 && NUM_VARYING_SECOND_PARTIALS<T> > 0) {
          ArgArray<ArgArray<Scalar>> H{};
          op_hess<T>(expr, x, val, H);
          Scalar* h = &m_second_partials[m_first_second_partials[i]];
          for (size_t j = 0; j < T::NUM_ARGS; ++j) {
            for (size_t k = 0; k <= j; ++k) {
              *h++ = H[j][k];
            }
          }
        }
      });
    }

    if constexpr (ORDER >= 1) {
      Scalar* negated = m_partials.data() + m_args.size();
      for (auto partial : m_negated_partials) {
        *negated++ = -m_partials[partial];
      }
    }
  }

  /// Computes the adjoints with the partial derivatives from the last
  /// update_values() call.
  ///
  /// @param callbacks The indices of the instructions to call f after, in
  ///     nondecreasing order and terminated by END_CALLBACKS. An index may
  ///     repeat to call f more than once after the same instruction. See
  ///     Node::instruction.
  /// @param f Called with each callback's index in callbacks and the adjoints
  ///     (one per root that reaches it) of the node whose instruction was just
  ///     replayed.
  template <typename F>
    requires std::invocable<F&, size_t, std::span<const Scalar>>
  void replay(std::span<const uint32_t> callbacks, F&& f) {
    slp_assert(!callbacks.empty() && callbacks.back() == END_CALLBACKS);
    propagate<Scalar>(
        [this] {
          return PropagationArrays<Scalar>{m_adjoints.data(),
                                           m_partials.data()};
        },
        callbacks.data(), f);
  }

  /// Computes the adjoints with the partial derivatives from the last
  /// update_values() call.
  void replay() {
    propagate<Scalar>(
        [this] {
          return PropagationArrays<Scalar>{m_adjoints.data(),
                                           m_partials.data()};
        },
        nullptr, NoCallbacks{});
  }

  /// Returns the non-leaf nodes's partial derivatives as expressions, laid
  /// out like partials().
  ///
  /// Only the partial derivatives replay() propagates are computed. The
  /// others are null.
  ///
  /// @return The non-leaf nodes's partial derivatives as expressions.
  gch::small_vector<ExpressionPtr<Scalar>> symbolic_partials() const {
    gch::small_vector<ExpressionPtr<Scalar>> partials(m_partials.size());
    for (const auto& node : m_nodes) {
      ArgArray<ExpressionPtr<Scalar>> g;
      grad_expr(node.expr, g);
      for (size_t k = 0; k < node.children.size(); ++k) {
        if (node.children[k] != SKIPPED_CHILD) {
          // Expression multiplication doesn't accept null operands
          slp_assert(g[k] != nullptr);
          partials[node.first_partial + k] = std::move(g[k]);
        }
      }
    }
    for (size_t idx = 0; idx < m_negated_partials.size(); ++idx) {
      partials[m_args.size() + idx] = -partials[m_negated_partials[idx]];
    }
    return partials;
  }

  /// Returns the second partial derivatives of the nodes that propagate
  /// adjoints as expressions, laid out like second_partials().
  ///
  /// Structurally zero second partial derivatives are null. Constant ones
  /// aren't stored, like second_partials().
  ///
  /// @return The second partial derivatives as expressions.
  gch::small_vector<ExpressionPtr<Scalar>> symbolic_second_partials() const {
    gch::small_vector<ExpressionPtr<Scalar>> second_partials(
        m_num_second_partials);
    for (const auto& node : m_nodes) {
      visit_op<Scalar>(node.expr->op(), [&]<typename T>() {
        if constexpr (NUM_VARYING_SECOND_PARTIALS<T> > 0) {
          ArgArray<ArgArray<ExpressionPtr<Scalar>>> H;
          hess_expr(node.expr, H);
          auto h = second_partials.begin() + node.first_second_partial;
          for (size_t j = 0; j < T::NUM_ARGS; ++j) {
            for (size_t k = 0; k <= j; ++k) {
              *h++ = std::move(H[j][k]);
            }
          }
        }
      });
    }
    return second_partials;
  }

  /// Computes the adjoints as expressions, like replay() does with values.
  ///
  /// This is useful when constructing optimization problems with derivatives
  /// in them. Unlike replay(), it doesn't need update_values() and doesn't
  /// modify the tape.
  ///
  /// @param partials The partial derivatives from symbolic_partials().
  /// @param callbacks The instructions to call f after, like replay()'s.
  /// @param f Called like replay()'s.
  /// @return The adjoint slots, laid out like adjoints(). Only the slots of
  ///     entries() hold adjoints.
  template <typename F>
    requires std::invocable<F&, size_t, std::span<const ExpressionPtr<Scalar>>>
  gch::small_vector<ExpressionPtr<Scalar>> symbolic_replay(
      std::span<const ExpressionPtr<Scalar>> partials,
      std::span<const uint32_t> callbacks, F&& f) const {
    slp_assert(!callbacks.empty() && callbacks.back() == END_CALLBACKS);
    gch::small_vector<ExpressionPtr<Scalar>> adjoints(m_adjoints.size());
    propagate<ExpressionPtr<Scalar>>(
        [&] {
          return PropagationArrays<ExpressionPtr<Scalar>>{adjoints.data(),
                                                          partials.data()};
        },
        callbacks.data(), f);
    return adjoints;
  }

  /// Computes the adjoints as expressions.
  ///
  /// @return The adjoint slots, laid out like adjoints(). Only the slots of
  ///     entries() hold adjoints.
  gch::small_vector<ExpressionPtr<Scalar>> symbolic_adjoints() const {
    const auto partials = symbolic_partials();
    gch::small_vector<ExpressionPtr<Scalar>> adjoints(m_adjoints.size());
    propagate<ExpressionPtr<Scalar>>(
        [&] {
          return PropagationArrays<ExpressionPtr<Scalar>>{adjoints.data(),
                                                          partials.data()};
        },
        nullptr, NoCallbacks{});
    return adjoints;
  }

 private:
  /// Marks a root that isn't in the tape because it's null or constant.
  static constexpr uint32_t NOT_IN_TAPE = UINT32_MAX;

  /// Tells propagate() that there are no callbacks, so it doesn't check for
  /// them after each instruction.
  struct NoCallbacks {};

  /// The arrays propagate() reads and writes.
  ///
  /// @tparam T The adjoint type.
  template <typename T>
  struct PropagationArrays {
    /// The adjoint slots.
    T* adjoints;

    /// The partial derivatives, laid out like partials().
    const T* partials;
  };

  /// A node's adjoint propagation.
  struct Instruction {
    /// The node's first adjoint slot. The node has one adjoint slot per root
    /// that reaches it.
    uint32_t slot;

    /// The number of the node's adjoint slots.
    uint32_t num_slots;

    /// The number of the node's adjoint operations per slot: one per active
    /// argument with a partial derivative that isn't structurally zero, after
    /// expanding eliminated arguments into their children.
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
  /// laid out like m_args, followed by the negations listed in
  /// m_negated_partials.
  gch::small_vector<Scalar> m_partials;

  /// Each non-leaf node's second partial derivatives with respect to its
  /// arguments (lower triangle, row-major), except constant ones. Only
  /// allocated once update_values<2>() is called.
  gch::small_vector<Scalar> m_second_partials;

  /// Each node's column in wrt, indexed by topological list index, or -1 if
  /// the node isn't an element of wrt.
  gch::small_vector<int> m_cols;

  /// The nodes that propagate adjoints, in topological order.
  gch::small_vector<Node> m_nodes;

  /// Each non-eliminated node's adjoint propagation, in the same order as
  /// m_nodes.
  gch::small_vector<Instruction> m_instructions;

  /// The index in m_partials of each adjoint operation's partial derivative,
  /// in replay order. Each node contributes num_propagated_args
  /// entries.
  gch::small_vector<uint32_t> m_propagated_partials;

  /// The index in m_partials of the partial derivative that each negated
  /// partial derivative negates. update_values() stores the negations in this
  /// order after the m_args.size() partial derivatives laid out like m_args.
  gch::small_vector<uint32_t> m_negated_partials;

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

  /// Constructs an AdjointTape.
  ///
  /// @param roots The roots whose adjoints to record. Null and constant roots
  ///     are skipped.
  /// @param wrt The leaves with respect to which to record adjoints. Must be
  ///     empty if values_only is true.
  /// @param values_only Whether to skip recording what's needed for partial
  ///     derivatives.
  AdjointTape(std::span<Expression<Scalar>* const> roots,
              std::span<Expression<Scalar>* const> wrt, bool values_only) {
    using enum ExpressionType;

    m_top_list = topological_sort(roots);

    // Sort dependent variables before independent ones while maintaining
    // relative order. Leaves have no children, so this is still a topological
    // sort. Edge pushing requires it.
    m_num_non_leaves =
        std::ranges::stable_partition(
            m_top_list, [](const auto& elem) { return !elem->is_leaf; })
            .begin() -
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
    //
    // The arguments are visited through visit_op() instead of a virtual call.
    // The branch predictor learns its conditional branches from the recent
    // operator sequence, and that history also lets it predict the virtual
    // op() calls. Without it, they mispredict about three times per node.
    uint32_t num_second_partials = 0;
    m_ops.reserve(m_num_non_leaves);
    m_first_partials.reserve(m_num_non_leaves);
    if (!values_only) {
      m_first_second_partials.reserve(m_num_non_leaves);
    }
    m_args.reserve(2 * m_num_non_leaves);
    for (size_t i = 0; i < m_num_non_leaves; ++i) {
      auto op = m_top_list[i]->op();
      m_ops.emplace_back(op);
      m_first_partials.emplace_back(static_cast<uint32_t>(m_args.size()));
      visit_op<Scalar>(op, [&, i]<typename T>() {
        for (const auto& arg : static_cast<const T&>(*m_top_list[i]).args) {
          m_args.emplace_back(arg->scratch);
        }
        if (!values_only) {
          m_first_second_partials.emplace_back(num_second_partials);
          num_second_partials += NUM_VARYING_SECOND_PARTIALS<T>;
        }
      });
    }
    m_values.resize(m_top_list.size());
    if (!values_only) {
      m_partials.resize(m_args.size());
      m_num_second_partials = num_second_partials;
    }

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
      if (static_cast<size_t>(i) < m_num_non_leaves) {
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

    gch::small_vector<NodeState> states(m_top_list.size());

    // The number of adjoint operations into each node
    gch::small_vector<uint32_t> num_in(m_top_list.size());

    // Push each node's roots down to its active children with nonzero partial
    // derivatives. Parents come first in topological order, so by the time
    // node i is visited, its root list is complete.
    //
    // Nodes with unit partial derivatives (a + b, a − b, and −a) are also
    // eliminated from the adjoint operations here where that doesn't add
    // operations. A parent's contribution ∂ϕᵢ/∂v a(vᵢ) reaches each of such a
    // node's children unchanged up to sign. The parent propagates it to them
    // directly with its own partial derivative or its negation instead, which
    // saves the node's instruction and adjoint slots. Sums of many terms are
    // chains of plus nodes, so this also lets a sum's root propagate to all of
    // its terms at once.
    //
    // A node with p incoming operations and c children replaces p + c
    // operations with p c, so it's eliminated if p c ≤ p + c. Nodes are
    // decided in topological order, so p counts the operations that reach the
    // node through eliminated parents. Roots aren't eliminated since their
    // adjoints are seeded.
    for (size_t i = 0; i < m_top_list.size(); ++i) {
      // Sort and deduplicate the roots. Each parent appends its roots, so a
      // root that reaches the node through more than one parent shows up more
      // than once. get_slot() also relies on the sorted order.
      auto& root_idxs = node_roots[i];
      if (root_idxs.size() > 1) {
        std::ranges::sort(root_idxs);
        root_idxs.erase(std::unique(root_idxs.begin(), root_idxs.end()),
                        root_idxs.end());
      }

      // Skip inactive or leaf nodes
      if (!active[i] || i >= m_num_non_leaves) {
        continue;
      }

      // Cache the children for recording the adjoint operations later
      const auto& children = node_children[i] = propagating_children(i, active);

      if (root_idxs.empty()) {
        continue;
      }

      // Each edge to a child carries one operation, or the p operations into
      // the node if it's eliminated
      uint32_t num_ops_per_edge = 1;
      const uint32_t p = num_in[i];
      const uint32_t c = std::ranges::count_if(
          children, [](auto k) { return k != SKIPPED_CHILD; });
      if (p * c <= p + c && unit_partial_signs(m_ops[i]) &&
          std::ranges::none_of(root_idxs, [&](auto root_idx) {
            return roots[root_idx] == m_top_list[i];
          })) {
        states[i] = NodeState::ELIMINATED;
        num_ops_per_edge = p;
      } else {
        states[i] = NodeState::RECORDED;
      }

      // Roots that reach this node also reach its children
      for (auto arg : children) {
        if (arg != SKIPPED_CHILD) {
          node_roots[arg].insert(node_roots[arg].end(), root_idxs.begin(),
                                 root_idxs.end());
          num_in[arg] += num_ops_per_edge;
        }
      }
    }

    record_adjoint_operations(roots, active, node_roots, node_children, states);

    for (auto& node : m_top_list) {
      node->scratch = -1;
    }
  }

  /// Each node's role in the adjoint operations, which the constructor
  /// decides.
  enum class NodeState : uint8_t {
    /// Inactive, a leaf, or not reached by any root through nonzero partial
    /// derivatives, so it has nothing to propagate.
    SKIPPED,

    /// Propagates adjoints with its own instruction.
    RECORDED,

    /// Eliminated from the adjoint operations (see the constructor).
    ELIMINATED
  };

  /// Allocates adjoint slots and records the adjoint operations.
  ///
  /// This is separate from the constructor because GCC stops inlining
  /// small_vector's push_back() in the recording loop when it's part of the
  /// constructor, which made recording about 3% slower.
  ///
  /// @param roots The roots passed to the constructor.
  /// @param active Whether each node is active, indexed by topological list
  ///     index.
  /// @param node_roots Which roots reach each node (sorted), indexed by
  ///     topological list index.
  /// @param node_children Each node's propagating children (see
  ///     propagating_children()), indexed by topological list index.
  /// @param states Each node's role, indexed by topological list index.
  void record_adjoint_operations(
      std::span<Expression<Scalar>* const> roots,
      std::span<const uint8_t> active,
      std::span<const gch::small_vector<uint32_t>> node_roots,
      std::span<const ArgArray<uint32_t>> node_children,
      std::span<const NodeState> states) {
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
      if (root != nullptr && root->type() != ExpressionType::CONSTANT &&
          active[root->scratch]) {
        m_seeds.push_back(get_slot(root->scratch, root_idx));
      }
    }

    // Record adjoint operations. Each of a node's adjoint slots propagates to
    // the same children, so only the destination slots are recorded.
    //
    // a(vₖ) += ∂ϕᵢ/∂vₖ a(vᵢ)
    //
    // An argument that's an eliminated node is expanded into the eliminated
    // node's children, recursively, and each gets an operation with the
    // argument's partial derivative.
    //
    // A child reached through an odd number of −1 partial derivatives gets
    // the negation of the argument's partial derivative, which update_values()
    // stores after the others (see m_negated_partials).
    gch::small_vector<uint32_t> op_children;
    gch::small_vector<std::pair<uint32_t, bool>> expansion_stack;

    // Each partial derivative's negation's index in m_partials, allocated on
    // first use
    constexpr uint32_t NOT_NEGATED = UINT32_MAX;
    gch::small_vector<uint32_t> negated_partials;
    auto get_partial = [&](uint32_t partial, bool negated) -> uint32_t {
      if (!negated) {
        return partial;
      }
      if (negated_partials.empty()) {
        negated_partials.assign(m_args.size(), NOT_NEGATED);
      }
      auto& negated_partial = negated_partials[partial];
      if (negated_partial == NOT_NEGATED) {
        negated_partial = m_args.size() + m_negated_partials.size();
        m_negated_partials.push_back(partial);
      }
      return negated_partial;
    };

    for (size_t i = 0; i < m_top_list.size(); ++i) {
      const auto& v_i = m_top_list[i];
      const auto state = states[i];

      if (state == NodeState::SKIPPED) {
        continue;
      }

      const auto& children = node_children[i];
      const auto& root_idxs = node_roots[i];

      // An eliminated node's operations are part of its parents's, so it has
      // no instruction. The first node in topological order is a root, so
      // there's always an instruction before it.
      const uint32_t num_instructions = m_instructions.size();
      slp_assert(state == NodeState::RECORDED || num_instructions > 0);
      m_nodes.push_back({v_i, static_cast<uint32_t>(i), m_first_partials[i],
                         m_first_second_partials[i], children,
                         state == NodeState::ELIMINATED ? num_instructions - 1
                                                        : num_instructions});
      if (state == NodeState::ELIMINATED) {
        continue;
      }

      // Records an operation. With one root (e.g., for a gradient or
      // Hessian), its destination is recorded right away. Otherwise, the
      // child is kept for recording each root's destination afterward.
      auto record_op = [&](uint32_t partial, uint32_t child) {
        m_propagated_partials.push_back(partial);
        if (root_idxs.size() == 1) {
          m_dests.push_back(get_slot(child, root_idxs[0]));
        } else {
          op_children.push_back(child);
        }
      };

      // Expand the arguments into operations
      const size_t first_op = m_propagated_partials.size();
      op_children.clear();
      for (size_t k = 0; k < children.size(); ++k) {
        const uint32_t child = children[k];
        if (child == SKIPPED_CHILD) {
          continue;
        }

        const uint32_t partial = m_first_partials[i] + k;
        if (states[child] != NodeState::ELIMINATED) {
          record_op(partial, child);
          continue;
        }

        expansion_stack.emplace_back(child, false);
        while (!expansion_stack.empty()) {
          const auto [j, negated] = expansion_stack.back();
          expansion_stack.pop_back();
          if (states[j] == NodeState::ELIMINATED) {
            const auto signs = *unit_partial_signs(m_ops[j]);

            // Pushed in reverse so they're expanded in argument order
            for (size_t arg = node_children[j].size(); arg-- > 0;) {
              if (node_children[j][arg] != SKIPPED_CHILD) {
                expansion_stack.emplace_back(node_children[j][arg],
                                             negated != (signs[arg] < 0));
              }
            }
          } else {
            record_op(get_partial(partial, negated), j);
          }
        }
      }
      const auto num_ops =
          static_cast<uint32_t>(m_propagated_partials.size() - first_op);

      if (root_idxs.size() > 1) {
        for (auto root_idx : root_idxs) {
          for (auto k : op_children) {
            m_dests.push_back(get_slot(k, root_idx));
          }
        }
      }

      m_instructions.push_back({slot_begin[i],
                                static_cast<uint32_t>(node_roots[i].size()),
                                num_ops});

      // The node's adjoints are dead once they're propagated
      if (m_cols[i] == -1) {
        free_blocks[node_roots[i].size()].push_back(slot_begin[i]);
      }
    }

    m_adjoints.resize(num_slots);
    m_partials.resize(m_args.size() + m_negated_partials.size());

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
  }

  /// Propagates adjoints from the roots with the recorded operations.
  ///
  /// The arrays are returned by a callable at each use instead of passed in,
  /// so replay()'s callable reads them from members like a loop written
  /// directly in replay() would. Passing spans kept their pointers and sizes
  /// live across the whole loop, including replay()'s callback, which made GCC
  /// spill registers and Hessian evaluation 11% slower. Separate callables for
  /// the two arrays made gradient evaluation 2% slower, since GCC no longer saw
  /// that both read from the same object.
  ///
  /// @tparam T The adjoint type: Scalar for values or ExpressionPtr<Scalar>
  ///     for expressions. A default-constructed T is zero.
  /// @param arrays Returns the PropagationArrays. The result slots are
  ///     overwritten, and the other slots must be zero.
  /// @param callbacks The instructions to call f after, like replay()'s, or
  ///     null if f is NoCallbacks.
  /// @param f Called like replay()'s, or NoCallbacks.
  template <typename T, typename GetArrays, typename F>
  void propagate(GetArrays&& arrays, const uint32_t* callbacks, F&& f) const {
    // Slots other than result slots are zeroed after their last use, so only
    // the result slots need to be zeroed here
    std::fill_n(arrays().adjoints, m_num_result_slots, T{});

    // Set each root's adjoint to 1 since df/df is 1
    for (auto seed : m_seeds) {
      arrays().adjoints[seed] = lit<T>(Scalar(1));
    }

    const uint32_t* dest = m_dests.data();
    const uint32_t* propagated_partial = m_propagated_partials.data();
    const uint32_t* callback = callbacks;
    for (size_t instruction_idx = 0; instruction_idx < m_instructions.size();
         ++instruction_idx) {
      const auto& instruction = m_instructions[instruction_idx];
      const uint32_t slot = instruction.slot;
      const uint32_t num_slots = instruction.num_slots;
      const uint32_t num_propagated_args = instruction.num_propagated_args;

      // A node's adjoint slots never alias its children's slots, so each
      // adjoint is final before it's propagated.
      //
      // Most nodes are reached by one root (all of them for a gradient or
      // Hessian), so that case avoids the loop over slots.
      //
      // The adjoint is copied instead of referenced, since the compiler can't
      // tell that the stores to its children's slots don't modify it.
      if (num_slots == 1) {
        const T adjoint = arrays().adjoints[slot];
        for (uint32_t k = 0; k < num_propagated_args; ++k) {
          arrays().adjoints[dest[k]] +=
              arrays().partials[propagated_partial[k]] * adjoint;
        }
        dest += num_propagated_args;
      } else {
        for (uint32_t slot_offset = 0; slot_offset < num_slots; ++slot_offset) {
          const T adjoint = arrays().adjoints[slot + slot_offset];
          for (uint32_t k = 0; k < num_propagated_args; ++k) {
            arrays().adjoints[dest[k]] +=
                arrays().partials[propagated_partial[k]] * adjoint;
          }
          dest += num_propagated_args;
        }
      }
      propagated_partial += num_propagated_args;

      if constexpr (!std::same_as<std::remove_cvref_t<F>, NoCallbacks>) {
        for (; *callback == instruction_idx; ++callback) {
          f(static_cast<size_t>(callback - callbacks),
            std::span<const T>{arrays().adjoints + slot, num_slots});
        }
      }

      // Zero the node's dead adjoint slots so they start at zero when reused
      if (slot >= m_num_result_slots) {
        if (num_slots == 1) {
          arrays().adjoints[slot] = T{};
        } else {
          std::fill_n(arrays().adjoints + slot, num_slots, T{});
        }
      }
    }
  }

  /// Returns the signs of an operator's partial derivatives if they're all 1
  /// or −1 (see HAS_UNIT_PARTIALS).
  ///
  /// @param op The operator.
  /// @return The signs in visit_args() order, or std::nullopt if the partial
  ///     derivatives aren't all 1 or −1.
  static std::optional<ArgArray<int>> unit_partial_signs(ExpressionOp op) {
    return visit_op<Scalar>(
        op, []<typename T>() -> std::optional<ArgArray<int>> {
          if constexpr (HAS_UNIT_PARTIALS<T>) {
            ArgArray<int> signs{};
            std::ranges::copy(T::UNIT_PARTIAL_SIGNS, signs.begin());
            return signs;
          } else {
            return std::nullopt;
          }
        });
  }

  /// Returns the children the node propagates adjoints to. Each element is the
  /// child's index in the topological list, or SKIPPED_CHILD if the child is
  /// inactive or its partial derivative is structurally zero.
  ///
  /// Whether a partial derivative is zero only depends on the expression type,
  /// not child values.
  ///
  /// The arguments are read from m_args instead of visited, which avoids a
  /// function_ref call per argument and loading each argument's scratch.
  ///
  /// @param i The node's index in the topological list. Must be a non-leaf.
  /// @param active Whether each node is active, indexed by topological list
  ///     index.
  ArgArray<uint32_t> propagating_children(
      size_t i, std::span<const uint8_t> active) const {
    ArgArray<uint32_t> children;
    children.fill(SKIPPED_CHILD);

    auto nonzero = grad_structure(*m_top_list[i]);

    const uint32_t first_arg = m_first_partials[i];
    const uint32_t end_arg = i + 1 < m_num_non_leaves
                                 ? m_first_partials[i + 1]
                                 : static_cast<uint32_t>(m_args.size());
    for (uint32_t k = 0; k < end_arg - first_arg; ++k) {
      const uint32_t arg = m_args[first_arg + k];
      if (nonzero[k] && active[arg]) {
        children[k] = arg;
      }
    }

    return children;
  }
};

}  // namespace slp::detail
