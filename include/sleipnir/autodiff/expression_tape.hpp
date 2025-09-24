// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_eval.hpp"
#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/expression_op.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp::detail {

/// The union of several roots's expression graphs stored as flat arrays: each
/// non-leaf node's operator and argument indices, and each node's value.
///
/// update_values() evaluates each node's value and partial derivatives from
/// the leaves's values without traversing the expression graph or making
/// virtual calls. Nodes shared between roots are evaluated once.
///
/// Nodes are identified by their index in top_list(). Each non-leaf node's
/// arguments are stored contiguously, and its partial derivative with respect
/// to an argument is stored at the same index as that argument, so
/// first_partial() locates both.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class ExpressionTape {
 public:
  /// Marks a root that isn't in the tape because it's null or constant.
  static constexpr uint32_t NOT_IN_TAPE = UINT32_MAX;

  /// Constructs an empty ExpressionTape.
  ExpressionTape() = default;

  /// Constructs an ExpressionTape.
  ///
  /// The graph's structure must not change while the ExpressionTape is in use.
  ///
  /// @param roots The roots whose expression graphs to record. Null and
  ///     constant roots are skipped.
  /// @param max_order The highest order of partial derivatives
  ///     update_values() will be called with. Storage for higher orders isn't
  ///     allocated.
  ExpressionTape(std::span<Expression<Scalar>* const> roots, int max_order)
      : m_max_order{max_order}, m_roots{roots.begin(), roots.end()} {
    slp_assert(max_order >= 0 && max_order <= 2);

    m_top_list = topological_sort(roots);

    // Sort dependent variables before independent ones while maintaining
    // relative order. Leaves have no children, so this is still a topological
    // sort.
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
    m_root_idxs.reserve(roots.size());
    for (auto root : roots) {
      m_root_idxs.emplace_back(root != nullptr && root->scratch != -1
                                   ? static_cast<uint32_t>(root->scratch)
                                   : NOT_IN_TAPE);
    }

    // Record each non-leaf node's operator and arguments, and where its
    // partial derivatives are stored.
    //
    // The arguments are visited through visit_op() instead of a virtual call.
    // The branch predictor learns its conditional branches from the recent
    // operator sequence, and that history also lets it predict the virtual
    // op() calls. Without it, they mispredict about three times per node.
    uint32_t num_second_partials = 0;
    m_ops.reserve(m_num_non_leaves);
    m_first_partials.reserve(m_num_non_leaves + 1);
    if (max_order >= 2) {
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
        if (max_order >= 2) {
          m_first_second_partials.emplace_back(num_second_partials);
          num_second_partials += NUM_VARYING_SECOND_PARTIALS<T>;
        }
      });
    }
    m_first_partials.emplace_back(static_cast<uint32_t>(m_args.size()));

    for (auto& node : m_top_list) {
      node->scratch = -1;
    }

    m_values.resize(m_top_list.size());
    if (max_order >= 1) {
      m_partials.resize(m_args.size());
    }
    m_second_partials.resize(num_second_partials);
  }

  /// Returns the highest order of partial derivatives update_values() can be
  /// called with.
  ///
  /// @return The highest order of partial derivatives.
  int max_order() const { return m_max_order; }

  /// Returns the topologically sorted union of the roots's expression graphs,
  /// with all non-leaf nodes before all leaf nodes.
  ///
  /// @return The topologically sorted union of the roots's expression graphs.
  const ExpressionGraph<Scalar>& top_list() const { return m_top_list; }

  /// Returns the number of non-leaf nodes at the front of top_list().
  ///
  /// @return The number of non-leaf nodes.
  size_t num_non_leaves() const { return m_num_non_leaves; }

  /// Returns the number of roots passed to the constructor.
  ///
  /// @return The number of roots.
  size_t num_roots() const { return m_roots.size(); }

  /// Returns a root's index in top_list().
  ///
  /// @param root The root's index in the roots passed to the constructor.
  /// @return The root's index in top_list(), or NOT_IN_TAPE if the root is
  ///     null or constant.
  uint32_t root_idx(size_t root) const { return m_root_idxs[root]; }

  /// Returns a root's value from the last update_values() call.
  ///
  /// @param root The root's index in the roots passed to the constructor.
  /// @return The root's value, or zero if the root is null.
  Scalar root_value(size_t root) const {
    if (uint32_t idx = m_root_idxs[root]; idx != NOT_IN_TAPE) {
      return m_values[idx];
    }

    // Constant roots aren't in the tape since their values never change
    const auto& expr = m_roots[root];
    return expr != nullptr ? expr->val : Scalar(0);
  }

  /// Returns a non-leaf node's arguments's indices in top_list(), in
  /// visit_args() order.
  ///
  /// @param idx The node's index in top_list(). Must be a non-leaf.
  /// @return The node's arguments.
  std::span<const uint32_t> args(size_t idx) const {
    return {m_args.data() + m_first_partials[idx],
            m_first_partials[idx + 1] - m_first_partials[idx]};
  }

  /// Returns the index of ∂ϕᵢ/∂a₀ in partials(), where a₀ is a non-leaf
  /// node's first argument. ∂ϕᵢ/∂aₖ is at index first_partial(idx) + k.
  ///
  /// @param idx The node's index in top_list(). Must be a non-leaf.
  /// @return The index of the node's first partial derivative.
  uint32_t first_partial(size_t idx) const { return m_first_partials[idx]; }

  /// Returns the index of ∂²ϕᵢ/∂a₀∂a₀ in second_partials(). ∂²ϕᵢ/∂aⱼ∂aₖ with
  /// j ≥ k is at index first_second_partial(idx) + j(j + 1)/2 + k.
  ///
  /// Only meaningful if max_order() is 2 and the node's operator has varying
  /// second partial derivatives (see NUM_VARYING_SECOND_PARTIALS).
  ///
  /// @param idx The node's index in top_list(). Must be a non-leaf.
  /// @return The index of the node's first second partial derivative.
  uint32_t first_second_partial(size_t idx) const {
    return m_first_second_partials[idx];
  }

  /// Returns the non-leaf nodes's partial derivatives from the last
  /// update_values() call with ORDER ≥ 1, followed by extra_partials().
  /// Structurally zero partial derivatives are zero.
  ///
  /// @return The non-leaf nodes's partial derivatives.
  std::span<const Scalar> partials() const { return m_partials; }

  /// Returns the number of partial derivatives at the front of partials(),
  /// which is one per argument of each non-leaf node.
  ///
  /// @return The number of partial derivatives.
  size_t num_partials() const { return m_args.size(); }

  /// Appends extra elements to partials(), replacing the ones from the last
  /// call.
  ///
  /// These are for values a consumer computes from the partial derivatives
  /// and reads through the same array as them (e.g., their negations).
  /// update_values() doesn't write them.
  ///
  /// @param count The number of extra elements.
  void resize_extra_partials(size_t count) {
    slp_assert(m_max_order >= 1);
    m_partials.resize(m_args.size() + count);
  }

  /// Returns the extra elements of partials(), which start at index
  /// num_partials().
  ///
  /// @return The extra elements of partials().
  std::span<Scalar> extra_partials() {
    return {m_partials.data() + m_args.size(),
            m_partials.size() - m_args.size()};
  }

  /// Returns the non-leaf nodes's second partial derivatives from the last
  /// update_values<2>() call. Structurally zero second partial derivatives
  /// are zero. Constant ones (see HAS_CONSTANT_SECOND_PARTIALS) aren't stored.
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
  ///     only values, 1 for partial derivatives, or 2 for second partial
  ///     derivatives too. Must be at most max_order().
  template <int ORDER>
    requires(ORDER >= 0 && ORDER <= 2)
  void update_values() {
    slp_assert(ORDER <= m_max_order);

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

        std::array<Scalar, T::NUM_ARGS> x;
        for (size_t k = 0; k < T::NUM_ARGS; ++k) {
          x[k] = m_values[m_args[first_partial + k]];
        }

        Scalar val = op_value<T>(expr, x);
        m_values[i] = val;

        if constexpr (ORDER >= 1) {
          std::array<Scalar, T::NUM_ARGS> g{};
          op_grad<T>(expr, x, val, g);
          for (size_t k = 0; k < T::NUM_ARGS; ++k) {
            m_partials[first_partial + k] = g[k];
          }
        }

        if constexpr (ORDER >= 2 && NUM_VARYING_SECOND_PARTIALS<T> > 0) {
          std::array<std::array<Scalar, T::NUM_ARGS>, T::NUM_ARGS> H{};
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
  }

 private:
  /// The highest order of partial derivatives update_values() can be called
  /// with.
  int m_max_order = 0;

  /// The topologically sorted union of the roots's expression graphs, with all
  /// non-leaf nodes before all leaf nodes.
  ExpressionGraph<Scalar> m_top_list;

  /// The number of non-leaf nodes at the front of m_top_list.
  size_t m_num_non_leaves = 0;

  /// The roots passed to the constructor.
  gch::small_vector<Expression<Scalar>*> m_roots;

  /// Each root's index in m_top_list, or NOT_IN_TAPE if it's null or constant.
  gch::small_vector<uint32_t> m_root_idxs;

  /// Each non-leaf node's operator, indexed by topological list index.
  gch::small_vector<uint8_t> m_ops;

  /// Each non-leaf node's arguments's topological list indices, in
  /// topological order of the nodes and visit_args() order of the arguments.
  gch::small_vector<uint32_t> m_args;

  /// Each non-leaf node's first argument's index in m_args, which is also its
  /// first partial derivative's index in m_partials. Indexed by topological
  /// list index and followed by m_args's size.
  gch::small_vector<uint32_t> m_first_partials;

  /// Each non-leaf node's first second partial derivative's index in
  /// m_second_partials, indexed by topological list index. Empty unless
  /// m_max_order is 2.
  gch::small_vector<uint32_t> m_first_second_partials;

  /// Each node's value, indexed by topological list index.
  gch::small_vector<Scalar> m_values;

  /// Each non-leaf node's partial derivatives with respect to its arguments,
  /// laid out like m_args, followed by the extra elements (see
  /// resize_extra_partials()). Empty if m_max_order is 0.
  gch::small_vector<Scalar> m_partials;

  /// Each non-leaf node's second partial derivatives with respect to its
  /// arguments (lower triangle, row-major), except constant ones. Empty unless
  /// m_max_order is 2.
  gch::small_vector<Scalar> m_second_partials;
};

}  // namespace slp::detail
