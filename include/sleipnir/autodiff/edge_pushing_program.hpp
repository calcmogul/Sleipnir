// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/adjoint_tape.hpp"
#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_op.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/linear_elimination.hpp"
#include "sleipnir/autodiff/slot_allocator.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/flat_map.hpp"
#include "sleipnir/util/force_inline.hpp"

namespace slp::detail {

/// The edge pushing operations that compute the Hessian of an AdjointTape's
/// root.
///
/// Which Hessian entries each node pushes to only depends on the expression
/// graph's structure, so the pushes are recorded once on construction, then
/// replayed with the tape's current values on each call to compute().
///
/// Before recording, linear nodes with constant partial derivatives (e.g.,
/// a + b, a − b, −a, and 2a) are eliminated where that doesn't add edges (see
/// LinearElimination). Their parents push directly to their children with the
/// constant partial derivatives folded into the operations's coefficients, so
/// the eliminated nodes never store Hessian entries.
///
/// @tparam Scalar Scalar type.
/// @tparam UpLo Which part of the Hessian to compute (Lower or Lower | Upper).
template <typename Scalar, int UpLo>
  requires(UpLo == Eigen::Lower) || (UpLo == (Eigen::Lower | Eigen::Upper))
class EdgePushingProgram {
 public:
  /// Constructs an empty EdgePushingProgram.
  EdgePushingProgram() = default;

  /// Records the edge pushing operations for a tape's root.
  ///
  /// The tape must have one root, which must be its first node.
  ///
  /// @param tape The tape.
  /// @param num_wrt The number of variables the Hessian is with respect to.
  EdgePushingProgram(const AdjointTape<Scalar>& tape, int num_wrt)
      : m_H{num_wrt, num_wrt} {
    Recorder{*this, tape}.record();
  }

  /// Replays the tape's adjoint operations and the recorded edge pushing
  /// operations with the tape's current values, then writes the result to
  /// value().
  ///
  /// The tape's values must be up to date.
  ///
  /// @param tape The tape the program was recorded from.
  void compute(AdjointTape<Scalar>& tape) {
    // The results accumulate onto their initial values. The temporaries's
    // first writes assign.
    std::ranges::copy(m_initial_results, m_slots.begin());

    auto g = tape.partials();
    auto H = tape.second_partials();
    tape.replay(
        m_callbacks, [&](size_t ops_idx, std::span<const Scalar> adjoints) {
          replay_node(ops_idx, adjoints[0], [&] {
            return ReplayArrays<Scalar>{m_slots.data(), m_weights.data(),
                                        m_coeffs.data(), g.data(), H.data()};
          });
        });

    std::copy_n(m_slots.begin(), m_H.nonZeros(), m_H.valuePtr());
    if constexpr (UpLo != Eigen::Lower) {
      copy_slots(m_H, m_mirrors, m_slots);
    }
  }

  /// Computes the Hessian as expressions, like compute() does with values.
  ///
  /// This is useful when constructing optimization problems with derivatives
  /// in them. Unlike compute(), it doesn't need the tape's values to be up to
  /// date.
  ///
  /// @param tape The tape the program was recorded from.
  /// @return The Hessian's nonzeros as expressions, laid out like value()'s
  ///     value array.
  gch::small_vector<ExpressionPtr<Scalar>> symbolic_values(
      const AdjointTape<Scalar>& tape) const {
    using T = ExpressionPtr<Scalar>;

    gch::small_vector<T> slots(m_slots.size());
    for (size_t i = 0; i < m_initial_results.size(); ++i) {
      if (m_initial_results[i] != Scalar(0)) {
        slots[i] = constant_ptr(m_initial_results[i]);
      }
    }

    gch::small_vector<T> weights(m_weights.size());
    gch::small_vector<T> coeffs;
    for (const auto& coeff : m_coeffs) {
      coeffs.emplace_back(constant_ptr(coeff));
    }

    const auto g = tape.symbolic_partials();
    const auto H = tape.symbolic_second_partials();
    tape.symbolic_replay(
        g, m_callbacks, [&](size_t ops_idx, std::span<const T> adjoints) {
          replay_node(ops_idx, adjoints[0], [&] {
            return ReplayArrays<T>{slots.data(), weights.data(), coeffs.data(),
                                   g.data(), H.data()};
          });
        });

    slots.resize(m_H.nonZeros());
    for (const auto& mirror : m_mirrors) {
      slots[mirror.value_idx] = slots[mirror.slot_idx];
    }
    return slots;
  }

  /// Returns the Hessian computed by the last call to compute().
  ///
  /// @return The Hessian.
  const Eigen::SparseMatrix<Scalar>& value() const { return m_H; }

 private:
  /// A term of a node's weight for a child, w(vₖ) = ∂ϕᵢ/∂vₖ.
  ///
  /// Edges from node vᵢ to the same child vₖ are merged, so w(vₖ) is the sum
  /// of coeff ∂ϕᵢ/∂a over the arguments a the edges go through. Each node's
  /// terms start with each child's first term in child order, which assign
  /// the weights, followed by the rest, which add to them. Most children have
  /// one term, so most weights are assigned once instead of zeroed and
  /// accumulated.
  struct WeightTerm {
    /// Index in the tape's partial derivatives of ∂ϕᵢ/∂a.
    uint32_t partial;

    /// Index in m_coeffs of the edge's scale.
    uint32_t coeff;

    /// The child's index among the node's children.
    uint32_t child;
  };

  /// Adds w(vⱼ) h(vᵢ, vⱼ) to h(vⱼ, vⱼ) a second time, where vⱼ is a child of
  /// node vᵢ.
  ///
  /// Pushing h(vᵢ, vⱼ) adds 2 w(vₖ) h(vᵢ, vⱼ) to h(vⱼ, vₖ) if j = k, since
  /// h(vⱼ, vₖ) and h(vₖ, vⱼ) are the same symmetric entry. The push records
  /// the first addition like any other, and this records the second.
  struct PushFixup {
    /// Slot of the entry h(vⱼ, vⱼ) to add to.
    uint32_t dest;

    /// Slot of the entry h(vᵢ, vⱼ) being pushed.
    uint32_t src;

    /// vⱼ's index among the node's children.
    uint32_t child;
  };

  /// Adds coeff ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ) to h(vⱼ, vₖ), where vⱼ and vₖ are children of
  /// node vᵢ.
  ///
  /// Creating adds the second-order contribution of node vᵢ's own expression,
  /// weighted by its adjoint, to its arguments's Hessian entries. These are
  /// only recorded for nonlinear expressions and stored in m_create_ops.
  struct CreateOp {
    /// Slot of the entry h(vⱼ, vₖ) to add to.
    uint32_t dest;

    /// Index in the tape's second partial derivatives of the second partial
    /// derivative with respect to the arguments the edges to vⱼ and vₖ go
    /// through, or CONSTANT_SECOND_PARTIAL if it's constant.
    uint32_t second_partial;

    /// Index in m_coeffs of the contribution's multiplicity (1 or 2) times the
    /// edges's scales. The multiplicity is 2 if the edges differ but j = k
    /// (e.g., x * x), since the (j, k) and (k, j) terms land on the same
    /// entry. A constant second partial derivative is folded into it.
    uint32_t coeff;
  };

  /// Marks a create operation whose second partial derivative is constant and
  /// folded into its coefficient.
  ///
  /// The tape doesn't store constant second partial derivatives (see
  /// HAS_CONSTANT_SECOND_PARTIALS), so the operation multiplies by m_coeffs[0]
  /// = 1 instead. Selecting which array to load from is cheaper than looping
  /// over a separate array of constant create operations, since that loop's
  /// overhead is paid by every node.
  static constexpr uint32_t CONSTANT_SECOND_PARTIAL = UINT32_MAX;

  /// Marks a node without a diagonal entry h(vᵢ, vᵢ) to push.
  static constexpr uint32_t NO_DIAG = UINT32_MAX;

  /// Marks a destination slot whose operation assigns instead of adds, since
  /// it's the first write to a temporary entry.
  static constexpr uint32_t ASSIGN = uint32_t{1} << 31;

  /// A tape node's operations.
  ///
  /// Only nodes with operations have a NodeOps, so compute() doesn't visit the
  /// others. Each one's operations are stored contiguously in m_weight_terms,
  /// m_push_srcs, m_push_dests, m_push_fixups, and m_create_ops, in tape
  /// order. Like the row pointers of a CSR matrix, each begin member is the
  /// index of the node's first operation in the corresponding array, and the
  /// node's operations end at the next NodeOps's begins. m_node_ops has a
  /// sentinel after the last node.
  ///
  /// Node vᵢ pushes its Hessian entries to its children with their weights
  /// w(vₖ), which are computed once per node. The pushes only record their
  /// destination slots, in this order:
  ///
  /// 1. If h(vᵢ, vᵢ) exists, h(vₖ₁, vₖ₂) += w(vₖ₁) w(vₖ₂) h(vᵢ, vᵢ) for each
  ///    pair of children with k1 ≥ k2 in child order (row-major lower
  ///    triangle).
  /// 2. For each source h(vᵢ, vⱼ) in m_push_srcs, h(vⱼ, vₖ) += w(vₖ) h(vᵢ, vⱼ)
  ///    for each child vₖ in child order.
  ///
  /// Push and create destinations marked with ASSIGN are their entry's first
  /// write, so they assign instead of add.
  ///
  /// Grouping the operations by node lets compute() replay them while the
  /// node's adjoint is available.
  struct NodeOps {
    /// Begin index in m_weight_terms.
    uint32_t weight_term;

    /// Begin index in m_push_srcs.
    uint32_t push_src;

    /// Begin index in m_push_dests.
    uint32_t push_dest;

    /// Begin index in m_push_fixups.
    uint32_t push_fixup;

    /// Begin index in m_create_ops.
    uint32_t create;

    /// The number of children the node pushes to, or zero if it has no
    /// Hessian entries to push.
    uint32_t num_children;

    /// Slot of the entry h(vᵢ, vᵢ), or NO_DIAG if it doesn't exist.
    uint32_t diag_src;
  };

  /// The arrays replay_node() reads and writes.
  ///
  /// @tparam T The value type.
  template <typename T>
  struct ReplayArrays {
    /// The Hessian entries's slots.
    T* slots;

    /// Scratch space for the node's weights.
    T* weights;

    /// The coefficients, laid out like m_coeffs.
    const T* coeffs;

    /// The tape's partial derivatives.
    const T* partials;

    /// The tape's second partial derivatives.
    const T* second_partials;
  };

  /// Replays a node's edge pushing operations.
  ///
  /// The arrays are returned by a callable, like AdjointTape::propagate()'s,
  /// so compute() reads them from members once per node like a loop written
  /// directly in compute() would.
  ///
  /// @tparam T The value type: Scalar for values or ExpressionPtr<Scalar> for
  ///     expressions. A default-constructed T is zero.
  /// @param ops_idx The node's index in m_node_ops.
  /// @param adjoint The node's adjoint. Only read if the node has create
  ///     operations.
  /// @param get_arrays Returns the ReplayArrays.
  template <typename T, typename GetArrays>
  SLEIPNIR_FORCE_INLINE void replay_node(size_t ops_idx, const T& adjoint,
                                         GetArrays&& get_arrays) const {
    const auto arrays = get_arrays();
    T* slots = arrays.slots;
    const T* coeffs = arrays.coeffs;
    const T* g = arrays.partials;
    const T* H = arrays.second_partials;

    // Adds a value to a destination's slot, or assigns it if the destination
    // is marked with ASSIGN. The old value is selected away rather than
    // branched around, since which writes assign doesn't follow a pattern the
    // branch predictor learns.
    auto write = [slots](uint32_t dest, const T& value) {
      const uint32_t slot = dest & ~ASSIGN;
      const T old = slots[slot];
      slots[slot] = ((dest & ASSIGN) ? T{} : old) + value;
    };

    const auto& ops = m_node_ops[ops_idx];
    const auto& next = m_node_ops[ops_idx + 1];

    if (const uint32_t num_children = ops.num_children; num_children > 0) {
      // w(vₖ) = ∂ϕᵢ/∂vₖ
      T* w = arrays.weights;
      const WeightTerm* terms = m_weight_terms.data() + ops.weight_term;
      for (uint32_t c = 0; c < num_children; ++c) {
        w[c] = coeffs[terms[c].coeff] * g[terms[c].partial];
      }
      for (uint32_t term_idx = ops.weight_term + num_children;
           term_idx < next.weight_term; ++term_idx) {
        const auto& term = m_weight_terms[term_idx];
        w[term.child] += coeffs[term.coeff] * g[term.partial];
      }

      const uint32_t* dest = m_push_dests.data() + ops.push_dest;

      // h(vₖ₁, vₖ₂) += w(vₖ₁) w(vₖ₂) h(vᵢ, vᵢ)
      if (ops.diag_src != NO_DIAG) {
        const T src = slots[ops.diag_src];
        for (uint32_t c1 = 0; c1 < num_children; ++c1) {
          const T w1_src = w[c1] * src;
          for (uint32_t c2 = 0; c2 <= c1; ++c2) {
            write(*dest++, w[c2] * w1_src);
          }
        }
      }

      // h(vⱼ, vₖ) += w(vₖ) h(vᵢ, vⱼ)
      for (uint32_t src_idx = ops.push_src; src_idx < next.push_src;
           ++src_idx) {
        const T src = slots[m_push_srcs[src_idx]];
        for (uint32_t c = 0; c < num_children; ++c) {
          write(*dest++, w[c] * src);
        }
      }

      // h(vⱼ, vⱼ) += w(vⱼ) h(vᵢ, vⱼ) a second time
      for (uint32_t fixup_idx = ops.push_fixup; fixup_idx < next.push_fixup;
           ++fixup_idx) {
        const auto& fixup = m_push_fixups[fixup_idx];
        slots[fixup.dest] += w[fixup.child] * slots[fixup.src];
      }
    }

    // The Hessian has one root, so each node has one adjoint
    for (uint32_t op_idx = ops.create; op_idx < next.create; ++op_idx) {
      const auto& op = m_create_ops[op_idx];
      const T& second_partial = op.second_partial == CONSTANT_SECOND_PARTIAL
                                    ? coeffs[0]
                                    : H[op.second_partial];
      write(op.dest, coeffs[op.coeff] * second_partial * adjoint);
    }
  }

  /// The operations of each tape node that has any, followed by a sentinel
  gch::small_vector<NodeOps> m_node_ops;

  /// The tape instruction after which each element of m_node_ops is replayed,
  /// terminated by AdjointTape::END_CALLBACKS.
  ///
  /// A node's create operations need its adjoint, so they're replayed right
  /// after its own instruction. Nodes the tape eliminated from its adjoint
  /// operations are linear, so they only push, and are replayed after the
  /// last instruction before them. Either way, nodes are replayed in tape
  /// order, which is all pushing needs.
  gch::small_vector<uint32_t> m_callbacks;

  gch::small_vector<WeightTerm> m_weight_terms;

  /// The slots of the off-diagonal entries h(vᵢ, vⱼ) each node pushes.
  gch::small_vector<uint32_t> m_push_srcs;

  /// The destination slots of each node's pushes. See NodeOps for their order.
  gch::small_vector<uint32_t> m_push_dests;

  gch::small_vector<PushFixup> m_push_fixups;
  gch::small_vector<CreateOp> m_create_ops;

  /// Scratch space for a node's weights during compute(), sized for the node
  /// with the most children.
  gch::small_vector<Scalar> m_weights;

  /// The distinct coefficients the operations multiply by. The first two are
  /// always 1 and 2.
  gch::small_vector<Scalar> m_coeffs;

  /// Value of each structurally nonzero entry h(vⱼ, vₖ) of the expression
  /// graph's Hessian.
  ///
  /// * Entries between elements of wrt are the results. They come first, and
  ///   each one's slot is its position in m_H's value array (in the lower
  ///   triangle for a full Hessian), so they're copied to m_H with one
  ///   contiguous copy.
  /// * The other entries are temporaries, which follow the results. They're
  ///   dead once the node they're stored at has pushed them, so their slots
  ///   are reused. Each one's first write assigns, so its slot's stale value
  ///   doesn't need to be zeroed.
  gch::small_vector<Scalar> m_slots;

  /// The results's values before compute() replays the operations: zero plus
  /// any constant contributions.
  gch::small_vector<Scalar> m_initial_results;

  /// For a full Hessian, which upper triangle positions in m_H's value array
  /// are copied from which lower triangle slots after compute() replays the
  /// operations. Empty for a lower triangular Hessian.
  gch::small_vector<SlotMapEntry> m_mirrors;

  Eigen::SparseMatrix<Scalar> m_H;

  /// Runs edge pushing structurally and records the operations it performs.
  ///
  /// Implements edge pushing as described by figure 4 on p. 14 of [1], on the
  /// expression graph left after linear node elimination.
  ///
  /// [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
  ///     efficient Hessian computation via automatic differentiation", 2016.
  ///     https://sci-hub.st/10.1007/s12532-016-0100-3
  class Recorder {
   public:
    /// Constructs a Recorder.
    ///
    /// @param program The program to record into.
    /// @param tape The tape to record the program from.
    Recorder(EdgePushingProgram& program, const AdjointTape<Scalar>& tape)
        : m_program{program},
          m_tape{tape},
          m_h(tape.top_list().size()),
          m_const_adjoints(tape.top_list().size(), Scalar(0)),
          m_child_idxs(tape.top_list().size(), UNMERGED) {}

    /// Records the program.
    void record() {
      const auto nodes = m_tape.nodes();

      m_hess_structures = HessStructures<Scalar>{m_tape};

      m_elimination = LinearElimination<Scalar>{m_tape, m_hess_structures};

      // The root's adjoint is 1, and each node propagates its own to its
      // children when it's visited below
      if (!nodes.empty()) {
        m_const_adjoints[nodes[0].idx] = Scalar(1);
      }

      m_program.m_coeffs = {Scalar(1), Scalar(2)};

      // The tape's nodes are the dependent variables vᵢ in topological order.
      // Each loop iteration records the pushing and creating steps for vᵢ.
      // The adjoint step a(vₖ) += ∂ϕᵢ/∂vₖ a(vᵢ) is recorded by the
      // AdjointTape.
      //
      // Nodes without operations are dropped, so compute() doesn't visit them.
      for (size_t pos = 0; pos < nodes.size(); ++pos) {
        record_node(pos);
        if (has_ops(m_program.m_node_ops.back())) {
          // Only nodes with their own instructions have adjoints. The tape
          // only eliminates nodes with unit partial derivatives, which have
          // no second partial derivatives to create with.
          slp_assert(m_program.m_create_ops.size() ==
                         m_program.m_node_ops.back().create ||
                     pos == 0 ||
                     nodes[pos].instruction != nodes[pos - 1].instruction);
          m_program.m_callbacks.push_back(nodes[pos].instruction);
        } else {
          m_program.m_node_ops.pop_back();
        }
      }
      m_program.m_node_ops.emplace_back(begin_node_ops());
      m_program.m_callbacks.push_back(AdjointTape<Scalar>::END_CALLBACKS);

      assign_slots();
    }

   private:
    /// Marks a node that isn't one of the current node's children in
    /// m_child_idxs.
    static constexpr uint32_t UNMERGED = UINT32_MAX;

    EdgePushingProgram& m_program;
    const AdjointTape<Scalar>& m_tape;

    /// Which second partial derivatives of each node aren't structurally
    /// zero.
    HessStructures<Scalar> m_hess_structures;

    LinearElimination<Scalar> m_elimination;

    /// Maps h(vⱼ, vₖ) with j ≤ k to its slot. Stored at index j.
    ///
    /// Entries are allocated slots when they're first written. An entry
    /// stored at index j is only written before node vⱼ pushes it (pushes and
    /// creates only write to entries of node vᵢ's children, which come after
    /// vᵢ), so unless it's a result, its slot is freed once node vⱼ's
    /// operations are recorded.
    gch::small_vector<flat_map<uint32_t, uint32_t>> m_h;
    SlotAllocator m_slots;

    /// Maps each coefficient other than 1 and 2 to its index in m_coeffs.
    flat_map<Scalar, uint32_t> m_coeff_idxs;

    /// Each node's adjoint if it's the same on every evaluation, or
    /// std::nullopt if it isn't, indexed by topological list index. Adjoints
    /// start at zero, the sum of no contributions.
    gch::small_vector<std::optional<Scalar>> m_const_adjoints;

    /// Constant contributions to results, as (slot, value) pairs.
    gch::small_vector<std::pair<uint32_t, Scalar>> m_baked;

    /// The current node's edges.
    gch::small_vector<Edge<Scalar>> m_edges;

    /// The current node's children, with its edges merged by child.
    gch::small_vector<uint32_t> m_children;

    /// Each of the current node's children's weight terms, as (partial,
    /// coefficient) pairs.
    gch::small_vector<gch::small_vector<std::pair<uint32_t, Scalar>>>
        m_child_terms;

    /// Each node's index among the current node's children, indexed by
    /// topological list index, or UNMERGED if it isn't one.
    gch::small_vector<uint32_t> m_child_idxs;

    /// The current node's create contributions, keyed by (j, k, arg1, arg2).
    flat_map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, Scalar>
        m_creates;

    /// Returns whether h(vⱼ, vₖ) is a result.
    bool is_result(uint32_t j, uint32_t k) const {
      return m_tape.col(j) != -1 && m_tape.col(k) != -1;
    }

    /// Returns h(vⱼ, vₖ)'s slot and whether it was just allocated.
    ///
    /// Results never reuse a temporary's slot, since slots are renumbered
    /// afterward and a result's slot becomes its position in m_H's value
    /// array.
    std::pair<uint32_t, bool> get_slot(uint32_t j, uint32_t k) {
      auto [it, inserted] = m_h[std::min(j, k)].try_emplace(std::max(j, k), 0);
      if (inserted) {
        it->second =
            is_result(j, k) ? m_slots.allocate_new() : m_slots.allocate();
      }
      return {it->second, inserted};
    }

    /// Returns the destination of an operation that writes h(vⱼ, vₖ).
    ///
    /// It's marked with ASSIGN if the operation is the entry's first write and
    /// the entry is a temporary, since temporaries's slots hold stale values
    /// from the last entry that used them. Results are initialized by
    /// compute() instead.
    ///
    /// Operations are recorded in the order compute() replays them, so the
    /// first recorded write is the first replayed write.
    uint32_t get_dest(uint32_t j, uint32_t k) {
      auto [slot, inserted] = get_slot(j, k);
      return inserted && !is_result(j, k) ? slot | ASSIGN : slot;
    }

    /// Returns a coefficient's index in m_coeffs, adding it if it isn't there.
    ///
    /// Most coefficients are 1 or 2, so they're always at indices 0 and 1 and
    /// skip the lookup.
    uint32_t get_coeff(Scalar coeff) {
      if (coeff == Scalar(1)) {
        return 0;
      } else if (coeff == Scalar(2)) {
        return 1;
      }

      auto& coeffs = m_program.m_coeffs;
      auto [it, inserted] =
          m_coeff_idxs.try_emplace(coeff, static_cast<uint32_t>(coeffs.size()));
      if (inserted) {
        coeffs.emplace_back(coeff);
      }
      return it->second;
    }

    /// Returns whether a node recorded any operations.
    ///
    /// Pushes are only recorded for nodes with weights, so the weight terms
    /// and create operations are the only arrays that need to be checked.
    ///
    /// @param ops The node's operations, which are the last ones recorded.
    bool has_ops(const NodeOps& ops) const {
      return ops.weight_term != m_program.m_weight_terms.size() ||
             ops.create != m_program.m_create_ops.size();
    }

    /// Returns a NodeOps that begins at the current end of each operation
    /// array.
    NodeOps begin_node_ops() const {
      return NodeOps{static_cast<uint32_t>(m_program.m_weight_terms.size()),
                     static_cast<uint32_t>(m_program.m_push_srcs.size()),
                     static_cast<uint32_t>(m_program.m_push_dests.size()),
                     static_cast<uint32_t>(m_program.m_push_fixups.size()),
                     static_cast<uint32_t>(m_program.m_create_ops.size()),
                     0,
                     NO_DIAG};
    }

    // record_node() and the steps it calls are force inlined into record()'s
    // loop. GCC doesn't inline them on its own, and Hessian setup took about 1%
    // more cycles without it.

    /// Records the pushing and creating steps for a node.
    ///
    /// @param pos The node's position in the tape's nodes().
    SLEIPNIR_FORCE_INLINE void record_node(size_t pos) {
      using enum ExpressionType;

      const auto& node = m_tape.nodes()[pos];
      const uint32_t i = node.idx;
      auto& ops = m_program.m_node_ops.emplace_back(begin_node_ops());

      // Parents come first in topological order, so the node's adjoint is
      // final
      propagate_constant_adjoint(node);

      // Eliminated nodes have no Hessian entries since their parents push past
      // them, and no second partial derivatives to create entries with
      if (m_elimination.eliminated(i)) {
        slp_assert(m_h[i].empty());
        return;
      }

      m_elimination.get_edges(m_tape.children(node), m_edges);

      // Pushed entries are always stored at nodes after vᵢ, so m_h[i] isn't
      // modified while it's iterated over
      if (!m_h[i].empty()) {
        merge_children(node);
        record_weights(ops);
        record_pushes(i, ops);
      }

      // The tape only contains nodes that a(vᵢ) can be nonzero for
      if (node.expr->type() > LINEAR) {
        record_creates(node, m_hess_structures[pos]);
      }

      // Entries stored at vᵢ are dead once vᵢ has pushed them, unless they're
      // results
      for (const auto& [j, slot] : m_h[i]) {
        if (!is_result(i, j)) {
          m_slots.free(slot);
        }
      }
    }

    /// Propagates a node's adjoint to its children if it's the same on every
    /// evaluation.
    ///
    /// A node's adjoint is constant if each of its parents has a constant
    /// adjoint and constant partial derivatives (see constant_partials()). For
    /// example, the nodes of a sum of squares just below the sum have constant
    /// adjoints. A child's adjoint becomes std::nullopt once any parent's
    /// contribution isn't constant.
    ///
    /// @param node The node, whose parents have all propagated to it.
    SLEIPNIR_FORCE_INLINE void propagate_constant_adjoint(
        const typename AdjointTape<Scalar>::Node& node) {
      constexpr auto SKIPPED_CHILD = AdjointTape<Scalar>::SKIPPED_CHILD;

      auto& adjoints = m_const_adjoints;

      const uint32_t i = node.idx;
      const auto g = adjoints[i] ? constant_partials(*node.expr) : std::nullopt;

      const auto children = m_tape.children(node);
      for (size_t arg = 0; arg < children.size(); ++arg) {
        const uint32_t k = children[arg];
        if (k == SKIPPED_CHILD || !adjoints[k]) {
          continue;
        }

        if (g) {
          *adjoints[k] += (*g)[arg] * *adjoints[i];
        } else {
          adjoints[k].reset();
        }
      }
    }

    /// Merges a node's edges by child into m_children and m_child_terms.
    ///
    /// ∂ϕᵢ/∂vₖ is the weight w(vₖ): the sum over the edges to vₖ of each
    /// edge's scale times the partial derivative with respect to the argument
    /// it goes through. Edges to the same child through the same argument are
    /// merged by summing their scales. Children whose terms all cancel (e.g.,
    /// x − x) are dropped, so every pushed entry gets a nonzero contribution.
    ///
    /// @param node The node.
    SLEIPNIR_FORCE_INLINE void merge_children(
        const typename AdjointTape<Scalar>::Node& node) {
      m_children.clear();
      m_child_terms.clear();
      for (const auto& edge : m_edges) {
        auto& child_idx = m_child_idxs[edge.k];
        if (child_idx == UNMERGED) {
          child_idx = m_children.size();
          m_children.push_back(edge.k);
          m_child_terms.emplace_back();
        }
        const uint32_t partial = node.first_partial + edge.arg;
        auto& terms = m_child_terms[child_idx];
        auto term = std::ranges::find(terms, partial,
                                      &std::pair<uint32_t, Scalar>::first);
        if (term != terms.end()) {
          term->second += edge.scale;
        } else {
          terms.emplace_back(partial, edge.scale);
        }
      }
      for (auto k : m_children) {
        m_child_idxs[k] = UNMERGED;
      }
      for (auto& terms : m_child_terms) {
        terms.erase(std::remove_if(terms.begin(), terms.end(),
                                   [](const auto& term) {
                                     return term.second == Scalar(0);
                                   }),
                    terms.end());
      }
      size_t num_children = 0;
      for (size_t c = 0; c < m_children.size(); ++c) {
        if (!m_child_terms[c].empty()) {
          m_children[num_children] = m_children[c];
          m_child_terms[num_children] = std::move(m_child_terms[c]);
          ++num_children;
        }
      }
      m_children.resize(num_children);
      m_child_terms.resize(num_children);
    }

    /// Records the weights's terms for the current node's children.
    ///
    /// Each child's first term is recorded in child order, and the rest after
    /// them.
    ///
    /// @param ops The node's operations.
    SLEIPNIR_FORCE_INLINE void record_weights(NodeOps& ops) {
      auto& weight_terms = m_program.m_weight_terms;
      for (const auto& terms : m_child_terms) {
        weight_terms.push_back({terms[0].first, get_coeff(terms[0].second), 0});
      }
      for (size_t c = 0; c < m_children.size(); ++c) {
        for (size_t t = 1; t < m_child_terms[c].size(); ++t) {
          weight_terms.push_back({m_child_terms[c][t].first,
                                  get_coeff(m_child_terms[c][t].second),
                                  static_cast<uint32_t>(c)});
        }
      }
      ops.num_children = m_children.size();
      if (m_program.m_weights.size() < m_children.size()) {
        m_program.m_weights.resize(m_children.size());
      }
    }

    /// Records the pushing step for node vᵢ.
    ///
    /// for all vᵢ, vⱼ such that h(vᵢ, vⱼ) ≠ 0
    ///   for all vₖ such that ∂ϕᵢ/∂vₖ ≠ 0
    ///     if i ≠ j
    ///       for all unordered pairs (vⱼ, vₖ) such that vⱼ < vᵢ or vₖ < vᵢ
    ///         if j = k
    ///           h(vⱼ, vₖ) += 2 ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
    ///         else
    ///           h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
    ///     else
    ///       for all unordered pairs (vₖ₁, vₖ₂) such that vₖ₁ < vᵢ or vₖ₂ < vᵢ
    ///         if k1 = k2
    ///           h(vₖ₁, vₖ₂) += 2 ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vⱼ)
    ///         else
    ///           h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vⱼ)
    ///
    /// The node's edges are merged by child, so each pair of children is
    /// distinct and the diagonal push's multiplicity is always 1.
    ///
    /// @param i The node's index in the topological list.
    /// @param ops The node's operations.
    SLEIPNIR_FORCE_INLINE void record_pushes(uint32_t i, NodeOps& ops) {
      auto& push_dests = m_program.m_push_dests;

      // m_h[i]'s first key is i if h(vᵢ, vᵢ) exists, so the diagonal push is
      // recorded first
      for (const auto& [j, src] : m_h[i]) {
        if (i == j) {
          // h(vₖ₁, vₖ₂) += w(vₖ₁) w(vₖ₂) h(vᵢ, vᵢ)
          ops.diag_src = src;
          for (size_t c1 = 0; c1 < m_children.size(); ++c1) {
            for (size_t c2 = 0; c2 <= c1; ++c2) {
              push_dests.push_back(get_dest(m_children[c1], m_children[c2]));
            }
          }
        } else {
          // h(vⱼ, vₖ) += w(vₖ) h(vᵢ, vⱼ), plus another w(vⱼ) h(vᵢ, vⱼ) if vⱼ
          // is a child
          m_program.m_push_srcs.push_back(src);
          for (size_t c = 0; c < m_children.size(); ++c) {
            const uint32_t dest = get_dest(j, m_children[c]);
            push_dests.push_back(dest);
            if (m_children[c] == j) {
              m_program.m_push_fixups.push_back(
                  {dest & ~ASSIGN, src, static_cast<uint32_t>(c)});
            }
          }
        }
      }
    }

    /// Records the creating step for a nonlinear node vᵢ.
    ///
    /// if a(vᵢ) ≠ 0
    ///   for all unordered pairs (vⱼ, vₖ) such that ∂²ϕᵢ/∂vⱼ∂vₖ ≠ 0
    ///     if j = k
    ///       h(vⱼ, vₖ) += 2 ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
    ///     else
    ///       h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
    ///
    /// Eliminated nodes are linear, so ∂²ϕᵢ/∂vⱼ∂vₖ is the edges's scales times
    /// the second partial derivative with respect to the arguments they go
    /// through. Contributions to the same entry through the same second
    /// partial derivative are merged, and ones that cancel are dropped.
    ///
    /// If ∂²ϕᵢ/∂vⱼ∂vₖ is constant (e.g., for products), it's folded into the
    /// contribution's coefficient. If a(vᵢ) is constant too (e.g., the
    /// products in a quadratic cost), a contribution to a result is constant,
    /// so it's added to the results's initial values instead of recorded.
    ///
    /// @param node The node.
    /// @param nonzero Which of the node's second partial derivatives aren't
    ///     structurally zero (see HessStructures).
    SLEIPNIR_FORCE_INLINE void record_creates(
        const typename AdjointTape<Scalar>::Node& node,
        std::span<const uint8_t> nonzero) {
      // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
      //
      // Edges are in argument order, so each argument's edges are a range of
      // m_edges. Only pairs of edges through arguments with a second partial
      // derivative that isn't structurally zero are visited, since a node's
      // edges can expand to many children (e.g., the terms of a sum).
      ArgArray<uint32_t> arg_begin{};
      ArgArray<uint32_t> arg_end{};
      for (uint32_t e = m_edges.size(); e-- > 0;) {
        arg_begin[m_edges[e].arg] = e;
      }
      for (uint32_t e = 0; e < m_edges.size(); ++e) {
        arg_end[m_edges[e].arg] = e + 1;
      }

      m_creates.clear();
      const uint8_t* h = nonzero.data();
      for (uint32_t arg1 = 0; arg1 < node.num_args; ++arg1) {
        for (uint32_t arg2 = 0; arg2 <= arg1; ++arg2) {
          if (!*h++) {
            continue;
          }

          // Edge pairs within one argument are unordered, so e2 ≤ e1
          for (uint32_t e1 = arg_begin[arg1]; e1 < arg_end[arg1]; ++e1) {
            const uint32_t e2_end = arg1 == arg2 ? e1 + 1 : arg_end[arg2];
            for (uint32_t e2 = arg_begin[arg2]; e2 < e2_end; ++e2) {
              const auto& edge1 = m_edges[e1];
              const auto& edge2 = m_edges[e2];
              m_creates[{std::min(edge1.k, edge2.k), std::max(edge1.k, edge2.k),
                         arg1, arg2}] +=
                  Scalar(e1 != e2 && edge1.k == edge2.k ? 2 : 1) * edge1.scale *
                  edge2.scale;
            }
          }
        }
      }

      const auto& const_adjoint = m_const_adjoints[node.idx];
      const auto H = constant_second_partials(*node.expr);
      for (const auto& [key, coeff] : m_creates) {
        const auto [j, k, arg1, arg2] = key;
        if (coeff == Scalar(0)) {
          continue;
        }

        if (H) {
          const Scalar folded_coeff = coeff * (*H)[arg1][arg2];
          if (folded_coeff == Scalar(0)) {
            continue;
          }

          if (const_adjoint && is_result(j, k)) {
            m_baked.emplace_back(get_slot(j, k).first,
                                 folded_coeff * *const_adjoint);
          } else {
            m_program.m_create_ops.push_back({get_dest(j, k),
                                              CONSTANT_SECOND_PARTIAL,
                                              get_coeff(folded_coeff)});
          }
        } else {
          m_program.m_create_ops.push_back(
              {get_dest(j, k),
               node.first_second_partial + arg1 * (arg1 + 1) / 2 + arg2,
               get_coeff(coeff)});
        }
      }
    }

    /// Returns the second partial derivatives of a node's expression with
    /// respect to its arguments if they're constant, or std::nullopt if they
    /// aren't.
    ///
    /// They're constant for products, where ∂²(ab)/∂a∂b = 1. See
    /// HAS_CONSTANT_SECOND_PARTIALS.
    ///
    /// @param expr The node's expression.
    /// @return The second partial derivatives in visit_args() order (lower
    ///     triangle only), or std::nullopt.
    static std::optional<ArgArray<ArgArray<Scalar>>> constant_second_partials(
        const Expression<Scalar>& expr) {
      return visit_op<Scalar>(
          expr.op(),
          [&]<typename T>() -> std::optional<ArgArray<ArgArray<Scalar>>> {
            if constexpr (HAS_CONSTANT_SECOND_PARTIALS<T>) {
              // The second partial derivatives don't depend on the
              // arguments's values
              ArgArray<ArgArray<Scalar>> H{};
              op_hess<T>(expr, ArgArray<Scalar>{}, Scalar(0), H);
              return H;
            } else {
              return std::nullopt;
            }
          });
    }

    /// Maps the results to m_H's nonzeros and renumbers the slots so each
    /// result's slot is its position in m_H's value array, and the
    /// temporaries follow the results.
    ///
    /// A full Hessian's upper triangle entries are copied from the lower
    /// triangle slots instead.
    void assign_slots() {
      auto& H = m_program.m_H;

      // Map Hessian entries between elements of wrt to m_H's nonzeros
      gch::small_vector<TapeEntry> entries;
      for (size_t j = 0; j < m_h.size(); ++j) {
        int row = m_tape.col(j);
        if (row == -1) {
          continue;
        }

        for (const auto& [k, slot] : m_h[j]) {
          int col = m_tape.col(k);

          // If indices don't refer to element in wrt, skip this value
          if (col == -1) {
            continue;
          }

          // In lower triangle, row index ≥ column index. For a full Hessian,
          // the lower triangle entry comes first so it's the one that gets the
          // slot.
          entries.push_back({slot, std::max(row, col), std::min(row, col)});
          if constexpr (UpLo != Eigen::Lower) {
            if (row != col) {
              entries.push_back({slot, std::min(row, col), std::max(row, col)});
            }
          }
        }
      }

      auto slot_map = make_slot_map(H, entries);

      constexpr uint32_t UNMAPPED = UINT32_MAX;
      gch::small_vector<uint32_t> new_slots(m_slots.size(), UNMAPPED);
      for (const auto& entry : slot_map) {
        auto& new_slot = new_slots[entry.slot_idx];
        if (new_slot == UNMAPPED) {
          new_slot = entry.value_idx;
        } else {
          m_program.m_mirrors.push_back({new_slot, entry.value_idx});
        }
      }
      uint32_t num_new_slots = H.nonZeros();
      for (auto& new_slot : new_slots) {
        if (new_slot == UNMAPPED) {
          new_slot = num_new_slots++;
        }
      }
      slp_assert(num_new_slots <= ASSIGN);
      m_program.m_slots.resize(num_new_slots);

      m_program.m_initial_results.assign(H.nonZeros(), Scalar(0));
      for (const auto& [slot, value] : m_baked) {
        m_program.m_initial_results[new_slots[slot]] += value;
      }

      // Renumbers a destination, keeping its ASSIGN mark
      auto new_dest = [&](uint32_t dest) {
        return new_slots[dest & ~ASSIGN] | (dest & ASSIGN);
      };

      for (auto& ops : m_program.m_node_ops) {
        if (ops.diag_src != NO_DIAG) {
          ops.diag_src = new_slots[ops.diag_src];
        }
      }
      for (auto& slot : m_program.m_push_srcs) {
        slot = new_slots[slot];
      }
      for (auto& dest : m_program.m_push_dests) {
        dest = new_dest(dest);
      }
      for (auto& fixup : m_program.m_push_fixups) {
        fixup.dest = new_slots[fixup.dest];
        fixup.src = new_slots[fixup.src];
      }
      for (auto& op : m_program.m_create_ops) {
        op.dest = new_dest(op.dest);
      }
    }
  };
};

}  // namespace slp::detail
