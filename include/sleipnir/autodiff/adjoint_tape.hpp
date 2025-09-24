// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_eval.hpp"
#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/expression_op.hpp"
#include "sleipnir/autodiff/expression_tape.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/slot_allocator.hpp"
#include "sleipnir/autodiff/slot_map.hpp"
#include "sleipnir/autodiff/vertex_elimination.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp::detail {

/// Records reverse accumulation of the adjoints of several roots, then replays
/// it with the expression graph's current values.
///
/// The roots's expression graphs are stored in an ExpressionTape, whose
/// update_values() evaluates each node's partial derivatives.
///
/// Which adjoints are nonzero only depends on the expression graph's structure,
/// so the recording gives each (node, root) pair that can have a nonzero
/// adjoint a slot, and each adjoint contribution an operation between slots:
///
///   a(vₖ) += ∂ϕᵢ/∂vₖ a(vᵢ)
///
/// The operations of a node vᵢ are grouped into an instruction. replay() runs
/// the instructions from parent to child with the partial derivatives from
/// update_values(), so each node's partial derivatives are evaluated once
/// regardless of how many roots reach it.
///
/// Each operation propagates along an edge of the linearized graph, which is
/// first reduced by vertex elimination with UnitEliminationPolicy (see
/// vertex_elimination.hpp). Nodes whose partial derivatives are all 1 or −1
/// (e.g., a + b, a − b, and −a) are eliminated where that doesn't add edges.
/// A parent's contribution ∂ϕᵢ/∂v a(vᵢ) reaches each of such a node's
/// children unchanged up to sign. The parent propagates it to them directly
/// with its own partial derivative or its negation instead, which saves the
/// node's instruction and adjoint slots. Sums of many terms are chains of
/// plus nodes, so this also lets a sum's root propagate to all of its terms
/// at once.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class AdjointTape {
 private:
  /// The default callback of replay() and symbolic_replay(), which tells
  /// propagate() that there are no callbacks, so it doesn't check for them
  /// after each instruction.
  struct NoCallbacks {};

  /// Whether F is NoCallbacks or a callback for adjoints of type T.
  template <typename F, typename T>
  static constexpr bool IS_CALLBACK =
      std::same_as<std::remove_cvref_t<F>, NoCallbacks> ||
      std::invocable<F&, size_t, std::span<const T>>;

 public:
  /// Marks a child that adjoint propagation skips because it's inactive or its
  /// partial derivative is structurally zero.
  static constexpr uint32_t SKIPPED_CHILD = UINT32_MAX;

  /// Terminates the callback instruction indices passed to replay().
  static constexpr uint32_t END_CALLBACKS = UINT32_MAX;

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
    /// the tape was constructed with a max_order of 2 and the node's second
    /// partial derivatives aren't constant (see HAS_CONSTANT_SECOND_PARTIALS).
    uint32_t first_second_partial;

    /// The number of arguments. children() returns this many children.
    uint32_t num_args;

    /// The index of the node's instruction, or of the last one before it if
    /// the node was eliminated from the adjoint operations. replay() calls
    /// back after this instruction for nodes whose index is in its callbacks.
    uint32_t instruction;
  };

  /// Constructs an empty AdjointTape.
  AdjointTape() = default;

  /// Constructs an AdjointTape.
  ///
  /// The graph's structure must not change while the AdjointTape is in use.
  ///
  /// @param roots The roots whose adjoints to record. Null and constant roots
  ///     are skipped.
  /// @param wrt The variables with respect to which to record adjoints.
  /// @param max_order The highest order of partial derivatives
  ///     update_values() will be called with: 1 for partial derivatives or 2
  ///     for second partial derivatives too.
  AdjointTape(std::span<Expression<Scalar>* const> roots,
              std::span<Expression<Scalar>* const> wrt, int max_order = 1)
      : m_exprs{roots, max_order} {
    slp_assert(max_order >= 1);
    Recorder{*this, wrt}.record();
  }

  /// Returns the topologically sorted union of the roots's expression graphs,
  /// with all non-leaf nodes before all leaf nodes.
  ///
  /// @return The topologically sorted union of the roots's expression graphs.
  const ExpressionGraph<Scalar>& top_list() const { return m_exprs.top_list(); }

  /// Returns the nodes that propagate adjoints, in topological order.
  ///
  /// This includes the nodes eliminated from the adjoint operations, which
  /// have no adjoints.
  ///
  /// @return The nodes that propagate adjoints, in topological order.
  std::span<const Node> nodes() const { return m_nodes; }

  /// Returns a node's children in visit_args() order. Each element is the
  /// child's index in the topological list, or SKIPPED_CHILD if the child is
  /// inactive or its partial derivative is structurally zero.
  ///
  /// @param node The node, from nodes().
  /// @return The node's children.
  std::span<const uint32_t> children(const Node& node) const {
    return {m_children.data() + node.first_partial, node.num_args};
  }

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
  /// update_values() call. Node::first_partial says where a node's are.
  /// Structurally zero partial derivatives are zero. They're followed by the
  /// negations of the ones that adjoint operations through eliminated nodes
  /// need.
  ///
  /// @return The non-leaf nodes's partial derivatives.
  std::span<const Scalar> partials() const { return m_exprs.partials(); }

  /// Returns the non-leaf nodes's second partial derivatives from the last
  /// update_values<2>() call. Node::first_second_partial says where a
  /// node's are. Structurally zero second partial derivatives are zero.
  /// Constant ones (see HAS_CONSTANT_SECOND_PARTIALS) aren't stored.
  ///
  /// @return The non-leaf nodes's second partial derivatives.
  std::span<const Scalar> second_partials() const {
    return m_exprs.second_partials();
  }

  /// Evaluates each non-leaf node's partial derivatives up to the given order
  /// from the leaves's current values.
  ///
  /// @tparam ORDER The highest order of partial derivatives to evaluate: 1 for
  ///     partial derivatives (needed by replay()), or 2 for second partial
  ///     derivatives too. Must be at most the constructor's max_order.
  template <int ORDER = 1>
    requires(ORDER >= 1 && ORDER <= 2)
  void update_values() {
    m_exprs.template update_values<ORDER>();

    const Scalar* partials = m_exprs.partials().data();
    Scalar* negated = m_exprs.extra_partials().data();
    for (auto partial : m_negated_partials) {
      *negated++ = -partials[partial];
    }
  }

  /// Computes the adjoints with the partial derivatives from the last
  /// update_values() call.
  ///
  /// @param callbacks The indices of the instructions to call f after, in
  ///     nondecreasing order and terminated by END_CALLBACKS. An index may
  ///     repeat to call f more than once after the same instruction. See
  ///     Node::instruction. Empty if f isn't given.
  /// @param f Called with each callback's index in callbacks and the adjoints
  ///     (one per root that reaches it) of the node whose instruction was just
  ///     replayed.
  template <typename F = NoCallbacks>
    requires IS_CALLBACK<F, Scalar>
  void replay(std::span<const uint32_t> callbacks = {}, F&& f = {}) {
    assert_callbacks<F>(callbacks);
    propagate<Scalar>(
        [this] {
          return PropagationArrays<Scalar>{m_adjoints.data(),
                                           m_exprs.partials().data()};
        },
        callbacks.data(), f);
  }

  /// Returns the non-leaf nodes's partial derivatives as expressions, laid
  /// out like partials().
  ///
  /// Only the partial derivatives replay() propagates are computed. The
  /// others are null.
  ///
  /// @return The non-leaf nodes's partial derivatives as expressions.
  gch::small_vector<ExpressionPtr<Scalar>> symbolic_partials() const {
    gch::small_vector<ExpressionPtr<Scalar>> partials(
        m_exprs.partials().size());
    for (const auto& node : m_nodes) {
      std::array<ExpressionPtr<Scalar>, MAX_ARGS<Scalar>> g;
      grad_expr(node.expr, g);
      const auto node_children = children(node);
      for (size_t k = 0; k < node_children.size(); ++k) {
        if (node_children[k] != SKIPPED_CHILD) {
          // Expression multiplication doesn't accept null operands
          slp_assert(g[k] != nullptr);
          partials[node.first_partial + k] = std::move(g[k]);
        }
      }
    }
    for (size_t idx = 0; idx < m_negated_partials.size(); ++idx) {
      partials[m_exprs.num_partials() + idx] =
          -partials[m_negated_partials[idx]];
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
        m_exprs.second_partials().size());
    for (const auto& node : m_nodes) {
      visit_op<Scalar>(node.expr->op(), [&]<typename T>() {
        if constexpr (NUM_VARYING_SECOND_PARTIALS<T> > 0) {
          std::array<std::array<ExpressionPtr<Scalar>, MAX_ARGS<Scalar>>,
                     MAX_ARGS<Scalar>>
              H;
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
  template <typename F = NoCallbacks>
    requires IS_CALLBACK<F, ExpressionPtr<Scalar>>
  gch::small_vector<ExpressionPtr<Scalar>> symbolic_replay(
      std::span<const ExpressionPtr<Scalar>> partials,
      std::span<const uint32_t> callbacks = {}, F&& f = {}) const {
    assert_callbacks<F>(callbacks);
    gch::small_vector<ExpressionPtr<Scalar>> adjoints(m_adjoints.size());
    propagate<ExpressionPtr<Scalar>>(
        [&] {
          return PropagationArrays<ExpressionPtr<Scalar>>{adjoints.data(),
                                                          partials.data()};
        },
        callbacks.data(), f);
    return adjoints;
  }

 private:
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

  /// A node's adjoint operations.
  ///
  /// Each of the node's adjoint slots has the same number of operations with
  /// the same partial derivatives. Only their destination slots differ.
  struct Instruction {
    /// The node's first adjoint slot. The node has one adjoint slot per root
    /// that reaches it.
    uint32_t slot;

    /// The number of the node's adjoint slots.
    uint32_t num_slots;

    /// The number of the node's adjoint operations per slot: one per active
    /// argument with a partial derivative that isn't structurally zero, after
    /// expanding eliminated arguments into their children.
    uint32_t num_ops;
  };

  /// The roots's expression graphs, which evaluates the partial derivatives.
  ExpressionTape<Scalar> m_exprs;

  /// Each node's column in wrt, indexed by topological list index, or -1 if
  /// the node isn't an element of wrt.
  gch::small_vector<int> m_cols;

  /// Each non-leaf node's children (see children()), laid out like the
  /// partial derivatives.
  gch::small_vector<uint32_t> m_children;

  /// The nodes that propagate adjoints, in topological order.
  gch::small_vector<Node> m_nodes;

  /// Each non-eliminated node's adjoint operations, in the same order as
  /// m_nodes.
  gch::small_vector<Instruction> m_instructions;

  /// The index in partials() of each adjoint operation's partial derivative,
  /// in replay order. Each instruction has num_ops elements.
  gch::small_vector<uint32_t> m_op_partials;

  /// The destination adjoint slot of each adjoint operation, in replay order.
  /// Each instruction has num_slots × num_ops elements, grouped by slot.
  gch::small_vector<uint32_t> m_op_dests;

  /// The index in partials() of the partial derivative that each negated
  /// partial derivative negates. update_values() stores the negations in this
  /// order in the expression tape's extra partial derivatives.
  gch::small_vector<uint32_t> m_negated_partials;

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

  /// Asserts that callbacks are terminated by END_CALLBACKS if F is a
  /// callback, or empty if it's NoCallbacks.
  template <typename F>
  static void assert_callbacks(
      [[maybe_unused]] std::span<const uint32_t> callbacks) {
    if constexpr (std::same_as<std::remove_cvref_t<F>, NoCallbacks>) {
      slp_assert(callbacks.empty());
    } else {
      slp_assert(!callbacks.empty() && callbacks.back() == END_CALLBACKS);
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
  /// @param callbacks The instructions to call f after, like replay()'s. Not
  ///     read if f is NoCallbacks.
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

    const uint32_t* dest = m_op_dests.data();
    const uint32_t* partial = m_op_partials.data();
    const uint32_t* callback = callbacks;
    for (size_t instruction_idx = 0; instruction_idx < m_instructions.size();
         ++instruction_idx) {
      const auto& instruction = m_instructions[instruction_idx];
      const uint32_t slot = instruction.slot;
      const uint32_t num_slots = instruction.num_slots;
      const uint32_t num_ops = instruction.num_ops;

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
        for (uint32_t k = 0; k < num_ops; ++k) {
          arrays().adjoints[dest[k]] += arrays().partials[partial[k]] * adjoint;
        }
        dest += num_ops;
      } else {
        for (uint32_t slot_offset = 0; slot_offset < num_slots; ++slot_offset) {
          const T adjoint = arrays().adjoints[slot + slot_offset];
          for (uint32_t k = 0; k < num_ops; ++k) {
            arrays().adjoints[dest[k]] +=
                arrays().partials[partial[k]] * adjoint;
          }
          dest += num_ops;
        }
      }
      partial += num_ops;

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

  /// Records an AdjointTape's adjoint operations from its expression tape.
  ///
  /// record() runs these steps, each of which visits the nodes once:
  ///
  /// 1. find_cols() and find_active() find which nodes depend on wrt.
  /// 2. find_roots() finds which roots reach each node, and so how many
  ///    adjoints it has, and chooses which nodes to eliminate.
  /// 3. record_instructions() allocates the adjoint slots and records the
  ///    operations between them.
  class Recorder {
   public:
    /// Constructs a Recorder.
    ///
    /// @param tape The tape to record into, whose expression tape is already
    ///     constructed.
    /// @param wrt The variables with respect to which to record adjoints.
    Recorder(AdjointTape& tape, std::span<Expression<Scalar>* const> wrt)
        : m_tape{tape},
          m_exprs{tape.m_exprs},
          m_wrt{wrt},
          m_num_nodes{m_exprs.top_list().size()} {}

    /// Records the adjoint operations.
    void record() {
      find_cols();
      find_active();
      find_roots();
      allocate_result_slots();
      record_seeds();
      record_instructions();
      record_entries();

      m_tape.m_adjoints.resize(m_slots.size());
      m_tape.m_exprs.resize_extra_partials(m_tape.m_negated_partials.size());
    }

   private:
    /// Which nodes are eliminated from the linearized graph.
    using Policy = UnitEliminationPolicy<Scalar>;

    /// A node's role in the adjoint operations.
    enum class NodeState : uint8_t {
      /// Inactive, a leaf, or not reached by any root through nonzero partial
      /// derivatives, so it has nothing to propagate.
      SKIPPED,

      /// Propagates adjoints with its own instruction.
      RECORDED,

      /// Eliminated from the adjoint operations. Its parents propagate past
      /// it to its children.
      ELIMINATED
    };

    /// Marks a node whose adjoint slots aren't allocated yet in m_slot_begin.
    static constexpr uint32_t UNALLOCATED = UINT32_MAX;

    /// Marks a partial derivative without a negation in m_negations.
    static constexpr uint32_t NOT_NEGATED = UINT32_MAX;

    AdjointTape& m_tape;
    const ExpressionTape<Scalar>& m_exprs;
    std::span<Expression<Scalar>* const> m_wrt;

    /// The number of nodes in the topological list.
    size_t m_num_nodes;

    /// Whether each node is active, indexed by topological list index. A node
    /// is active if it's an element of wrt or depends on one. Inactive nodes
    /// have zero derivatives with respect to wrt.
    gch::small_vector<uint8_t> m_active;

    /// The indices of the roots that reach each node through nonzero partial
    /// derivatives, indexed by topological list index. A node has one adjoint
    /// slot per root, in this order. Sorted once find_roots() has visited the
    /// node.
    gch::small_vector<gch::small_vector<uint32_t>> m_node_roots;

    /// Each node's role, indexed by topological list index.
    gch::small_vector<NodeState> m_states;

    /// Each node's first adjoint slot, indexed by topological list index, or
    /// UNALLOCATED.
    gch::small_vector<uint32_t> m_slot_begin;

    SlotAllocator m_slots;

    /// Each partial derivative's negation's index in partials(), or
    /// NOT_NEGATED. Allocated when the first negation is needed.
    gch::small_vector<uint32_t> m_negations;

    /// The children of the current instruction's operations, if more than one
    /// root reaches its node.
    gch::small_vector<uint32_t> m_op_children;

    /// The nodes left to visit while expanding an eliminated argument, each
    /// with the sign of the edge composed so far.
    gch::small_vector<ScaledChild<int>> m_expansion_stack;

    /// Finds each node's column in wrt.
    void find_cols() {
      const auto& top_list = m_exprs.top_list();

      // The nodes don't know their indices in the topological list, so each
      // element of wrt's column is temporarily stored in its scratch, which
      // is -1 for every other node
      for (size_t col = 0; col < m_wrt.size(); ++col) {
        m_wrt[col]->scratch = col;
      }
      auto& cols = m_tape.m_cols;
      cols.reserve(m_num_nodes);
      for (const auto& node : top_list) {
        cols.emplace_back(node->scratch);
      }
      for (auto& elem : m_wrt) {
        elem->scratch = -1;
      }
    }

    /// Finds which nodes are active.
    void find_active() {
      const auto& top_list = m_exprs.top_list();
      const size_t num_non_leaves = m_exprs.num_non_leaves();

      // Children come last in topological order, so by the time node i is
      // visited, its arguments are final
      m_active.resize(m_num_nodes);
      for (size_t i = m_num_nodes; i-- > 0;) {
        bool active = m_tape.m_cols[i] != -1 &&
                      top_list[i]->type() != ExpressionType::CONSTANT;
        if (i < num_non_leaves) {
          for (auto arg : m_exprs.args(i)) {
            active = active || m_active[arg];
          }
        }
        m_active[i] = active;
      }
    }

    /// Returns whether a root has an adjoint to seed.
    ///
    /// @param root The root's index in the roots passed to the constructor.
    bool is_seeded(size_t root) const {
      const uint32_t idx = m_exprs.root_idx(root);
      return idx != ExpressionTape<Scalar>::NOT_IN_TAPE && m_active[idx];
    }

    /// Finds which roots reach each node, each node's children, and each
    /// node's role.
    ///
    /// Nodes are eliminated from the linearized graph according to the
    /// elimination policy. They're decided from parent to child, so a node's
    /// in-edges p are counted after its parents's eliminations, which is the
    /// number of operations into it, and its out-edges c before its
    /// children's. Roots aren't eliminated since their adjoints are seeded,
    /// and elements of wrt aren't since their adjoints are results.
    void find_roots() {
      const auto& top_list = m_exprs.top_list();
      const size_t num_non_leaves = m_exprs.num_non_leaves();

      m_tape.m_children.assign(m_exprs.num_partials(), SKIPPED_CHILD);
      m_states.resize(m_num_nodes);

      // Each seeded root reaches its own node
      m_node_roots.resize(m_num_nodes);
      for (size_t root = 0; root < m_exprs.num_roots(); ++root) {
        if (is_seeded(root)) {
          m_node_roots[m_exprs.root_idx(root)].push_back(root);
        }
      }

      // The number of adjoint operations into each node
      gch::small_vector<uint32_t> num_in(m_num_nodes);

      // Push each node's roots down to its children. Parents come first in
      // topological order, so by the time node i is visited, its root list is
      // complete.
      for (size_t i = 0; i < m_num_nodes; ++i) {
        // Sort and deduplicate the roots. Each parent appends its roots, so a
        // root that reaches the node through more than one parent shows up
        // more than once.
        auto& roots = m_node_roots[i];
        if (roots.size() > 1) {
          std::ranges::sort(roots);
          roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
        }

        // Skip inactive or leaf nodes
        if (!m_active[i] || i >= num_non_leaves) {
          continue;
        }

        // The children are also needed by nodes no root reaches, since
        // children() is defined for them
        const auto children = find_children(i);

        if (roots.empty()) {
          continue;
        }

        // Each edge to a child carries one operation, or the p operations
        // into the node if it's eliminated
        uint32_t num_ops_per_edge = 1;
        const uint32_t p = num_in[i];
        const uint32_t c = std::ranges::count_if(
            children, [](auto k) { return k != SKIPPED_CHILD; });
        if (Policy::accepts(p, c) && m_tape.m_cols[i] == -1 &&
            Policy::weights(*top_list[i]) &&
            std::ranges::none_of(roots, [&](auto root) {
              return m_exprs.root_idx(root) == i;
            })) {
          m_states[i] = NodeState::ELIMINATED;
          num_ops_per_edge = p;
        } else {
          m_states[i] = NodeState::RECORDED;
        }

        // Roots that reach this node also reach its children
        for (auto k : children) {
          if (k != SKIPPED_CHILD) {
            m_node_roots[k].insert(m_node_roots[k].end(), roots.begin(),
                                   roots.end());
            num_in[k] += num_ops_per_edge;
          }
        }
      }
    }

    /// Returns a non-leaf node's children in the tape's m_children.
    ///
    /// @param i The node's index in the topological list. Must be a non-leaf.
    std::span<uint32_t> children(size_t i) {
      return {m_tape.m_children.data() + m_exprs.first_partial(i),
              m_exprs.args(i).size()};
    }

    /// Finds the children a node propagates adjoints to and stores them in
    /// the tape's m_children, whose elements start as SKIPPED_CHILD.
    ///
    /// Whether a partial derivative is structurally zero only depends on the
    /// operator, not the argument values.
    ///
    /// @param i The node's index in the topological list. Must be a non-leaf.
    /// @return The node's children (see AdjointTape::children()).
    std::span<const uint32_t> find_children(size_t i) {
      const auto nonzero = grad_structure(*m_exprs.top_list()[i]);

      const auto node_children = children(i);
      const uint32_t* args = m_exprs.args(i).data();
      for (size_t k = 0; k < node_children.size(); ++k) {
        const uint32_t arg = args[k];
        if (nonzero[k] && m_active[arg]) {
          node_children[k] = arg;
        }
      }

      return node_children;
    }

    /// Allocates the adjoint slots of wrt. They hold the results, so they're
    /// allocated first and never reused.
    void allocate_result_slots() {
      m_slot_begin.assign(m_num_nodes, UNALLOCATED);
      for (size_t i = 0; i < m_num_nodes; ++i) {
        if (m_tape.m_cols[i] != -1) {
          m_slot_begin[i] = m_slots.allocate_new(m_node_roots[i].size());
        }
      }
      m_tape.m_num_result_slots = m_slots.size();
    }

    /// Returns the slot of a node's adjoint for a root, allocating the node's
    /// slots if they aren't yet.
    ///
    /// Every node other than an element of wrt gets its slots when they're
    /// first written, and they're freed once the node has propagated its
    /// adjoints.
    ///
    /// @param i The node's index in the topological list.
    /// @param root The index of a root that reaches the node.
    uint32_t get_slot(size_t i, uint32_t root) {
      const auto& roots = m_node_roots[i];
      if (m_slot_begin[i] == UNALLOCATED) {
        m_slot_begin[i] = m_slots.allocate(roots.size());
      }
      return m_slot_begin[i] +
             (std::ranges::lower_bound(roots, root) - roots.begin());
    }

    /// Records the slot of each seeded root's own adjoint.
    void record_seeds() {
      for (size_t root = 0; root < m_exprs.num_roots(); ++root) {
        if (is_seeded(root)) {
          m_tape.m_seeds.push_back(get_slot(m_exprs.root_idx(root), root));
        }
      }
    }

    /// Returns the index in partials() of a partial derivative's negation,
    /// adding the negation if it isn't there.
    ///
    /// @param partial The partial derivative's index in partials().
    uint32_t get_negation(uint32_t partial) {
      if (m_negations.empty()) {
        m_negations.assign(m_exprs.num_partials(), NOT_NEGATED);
      }
      auto& negation = m_negations[partial];
      if (negation == NOT_NEGATED) {
        negation = m_exprs.num_partials() + m_tape.m_negated_partials.size();
        m_tape.m_negated_partials.push_back(partial);
      }
      return negation;
    }

    /// Records the tape's nodes and each recorded node's instruction.
    void record_instructions() {
      const auto& top_list = m_exprs.top_list();
      const bool has_second_partials = m_exprs.max_order() >= 2;

      for (size_t i = 0; i < m_num_nodes; ++i) {
        const auto state = m_states[i];

        if (state == NodeState::SKIPPED) {
          continue;
        }

        const auto node_children = children(i);
        const auto& roots = m_node_roots[i];

        // An eliminated node's operations are part of its parents's, so it
        // has no instruction. The first node in topological order is a root,
        // so there's always an instruction before it.
        const uint32_t num_instructions = m_tape.m_instructions.size();
        slp_assert(state == NodeState::RECORDED || num_instructions > 0);
        m_tape.m_nodes.push_back(
            {top_list[i], static_cast<uint32_t>(i), m_exprs.first_partial(i),
             has_second_partials ? m_exprs.first_second_partial(i) : 0,
             static_cast<uint32_t>(node_children.size()),
             state == NodeState::ELIMINATED ? num_instructions - 1
                                            : num_instructions});
        if (state == NodeState::ELIMINATED) {
          continue;
        }

        // Records an operation. With one root (e.g., for a gradient or
        // Hessian), its destination is recorded right away. Otherwise, the
        // child is kept for recording each root's destination afterward.
        auto record_op = [&](uint32_t partial, uint32_t child) {
          m_tape.m_op_partials.push_back(partial);
          if (roots.size() == 1) {
            m_tape.m_op_dests.push_back(get_slot(child, roots[0]));
          } else {
            m_op_children.push_back(child);
          }
        };

        // Record an operation per edge of the reduced graph. An edge to an
        // eliminated child is composed with the child's out-edges,
        // recursively, so each node they reach gets an operation with the
        // partial derivative with respect to the eliminated child, or its
        // negation if the composed edge's scale is −1.
        const size_t first_op = m_tape.m_op_partials.size();
        m_op_children.clear();
        for (size_t k = 0; k < node_children.size(); ++k) {
          const uint32_t child = node_children[k];
          if (child == SKIPPED_CHILD) {
            continue;
          }

          const uint32_t partial = m_exprs.first_partial(i) + k;
          if (m_states[child] != NodeState::ELIMINATED) {
            record_op(partial, child);
            continue;
          }

          m_expansion_stack.push_back({child, 1});
          while (!m_expansion_stack.empty()) {
            const auto [j, scale] = m_expansion_stack.back();
            m_expansion_stack.pop_back();
            if (m_states[j] == NodeState::ELIMINATED) {
              const auto signs = *Policy::weights(*top_list[j]);

              // Pushed in reverse so they're expanded in argument order
              const auto j_children = children(j);
              for (size_t arg = j_children.size(); arg-- > 0;) {
                if (j_children[arg] != SKIPPED_CHILD) {
                  m_expansion_stack.push_back(
                      {j_children[arg], scale * signs[arg]});
                }
              }
            } else {
              record_op(scale < 0 ? get_negation(partial) : partial, j);
            }
          }
        }
        const auto num_ops =
            static_cast<uint32_t>(m_tape.m_op_partials.size() - first_op);

        // Each of the node's adjoint slots propagates to the same children
        if (roots.size() > 1) {
          for (auto root : roots) {
            for (auto child : m_op_children) {
              m_tape.m_op_dests.push_back(get_slot(child, root));
            }
          }
        }

        m_tape.m_instructions.push_back(
            {m_slot_begin[i], static_cast<uint32_t>(roots.size()), num_ops});

        // The node's adjoints are dead once they're propagated
        if (m_tape.m_cols[i] == -1) {
          m_slots.free(m_slot_begin[i], roots.size());
        }
      }
    }

    /// Records which slots hold the adjoints of wrt.
    void record_entries() {
      for (size_t i = 0; i < m_num_nodes; ++i) {
        const int col = m_tape.m_cols[i];
        if (col == -1) {
          continue;
        }

        const auto& roots = m_node_roots[i];
        for (size_t slot_offset = 0; slot_offset < roots.size();
             ++slot_offset) {
          m_tape.m_entries.push_back(
              {static_cast<uint32_t>(m_slot_begin[i] + slot_offset),
               static_cast<int>(roots[slot_offset]), col});
        }
      }
    }
  };
};

}  // namespace slp::detail
