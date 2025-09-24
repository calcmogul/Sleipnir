// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <optional>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/adjoint_tape.hpp"
#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/flat_map.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/// This class calculates the Hessian of a variable with respect to a vector of
/// variables.
///
/// The Hessian is computed with edge pushing. Which Hessian entries each node
/// pushes to only depends on the expression graph's structure, so the pushes
/// are recorded once on construction, then replayed with new values on each
/// call to value().
///
/// @tparam Scalar Scalar type.
/// @tparam UpLo Which part of the Hessian to compute (Lower or Lower | Upper).
///     Default is Lower | Upper.
template <typename Scalar, int UpLo>
  requires(UpLo == Eigen::Lower) || (UpLo == (Eigen::Lower | Eigen::Upper))
class Hessian {
 public:
  /// Constructs a Hessian object.
  ///
  /// @param variable Variable of which to compute the Hessian.
  /// @param wrt Variable with respect to which to compute the Hessian.
  Hessian(Variable<Scalar> variable, Variable<Scalar> wrt)
      : Hessian{std::move(variable), VariableMatrix<Scalar>{std::move(wrt)}} {}

  /// Constructs a Hessian object.
  ///
  /// @param variable Variable of which to compute the Hessian.
  /// @param wrt Vector of variables with respect to which to compute the
  ///     Hessian.
  Hessian(Variable<Scalar> variable, SleipnirMatrixLike<Scalar> auto wrt)
      : m_variable{std::move(variable)}, m_wrt{std::move(wrt)} {
    slp_assert(m_wrt.cols() == 1);

    gch::small_vector<detail::Expression<Scalar>*> roots{m_variable.expr.get()};
    gch::small_vector<detail::Expression<Scalar>*> wrt_exprs;
    for (auto& elem : m_wrt) {
      wrt_exprs.emplace_back(elem.expr.get());
    }
    m_tape = detail::AdjointTape<Scalar>{roots, wrt_exprs};

    record_pushes();

    if (m_variable.type() <= ExpressionType::QUADRATIC) {
      m_tape.update_values();
      compute();
    }
  }

  /// Returns the Hessian as a VariableMatrix.
  ///
  /// This is useful when constructing optimization problems with derivatives in
  /// them.
  ///
  /// @return The Hessian as a VariableMatrix.
  VariableMatrix<Scalar> get() const {
    VariableMatrix<Scalar> result{detail::empty, m_wrt.rows(), m_wrt.rows()};

    auto H = hessian_tree();

    for (int row = 0; row < m_wrt.rows(); ++row) {
      if constexpr (UpLo == Eigen::Lower) {
        for (int col = 0; col <= row; ++col) {
          if (H[row, col].expr != nullptr) {
            result[row, col] = std::move(H[row, col]);
          } else {
            result[row, col] = Variable{Scalar(0)};
          }
        }
      } else {
        for (int col = 0; col < m_wrt.rows(); ++col) {
          if (H[row, col].expr != nullptr) {
            result[row, col] = std::move(H[row, col]);
          } else {
            result[row, col] = Variable{Scalar(0)};
          }
        }
      }
    }

    return result;
  }

  /// Evaluates the Hessian at wrt's value.
  ///
  /// @return The Hessian at wrt's value.
  const Eigen::SparseMatrix<Scalar>& value() {
    if (m_variable.type() > ExpressionType::QUADRATIC) {
      m_tape.update_values();
      compute();
    }

    return m_H;
  }

 private:
  /// Adds coeff ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ) to h(vⱼ, vₖ) (pushing an off-diagonal entry),
  /// or coeff ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ) to h(vₖ₁, vₖ₂) (pushing the diagonal
  /// entry), where vₖ, vₖ₁, and vₖ₂ are arguments of node vᵢ's expression.
  ///
  /// Pushing moves node vᵢ's Hessian entries onto its arguments, since vᵢ is
  /// eliminated from the expression graph once its contributions are pushed.
  /// Off-diagonal pushes are stored in m_push_ops and diagonal pushes in
  /// m_push_diag_ops.
  struct PushOp {
    /// Slot in m_slots of the entry to add to: h(vⱼ, vₖ) for an off-diagonal
    /// push, or h(vₖ₁, vₖ₂) for a diagonal push.
    uint32_t dest;

    /// Slot in m_slots of the entry h(vᵢ, vⱼ) being pushed. For a diagonal
    /// push, j = i.
    uint32_t src;

    /// Argument position of vₖ (off-diagonal push) or vₖ₁ (diagonal push) in
    /// node vᵢ's expression.
    uint8_t arg1;

    /// Argument position of vₖ₂ in node vᵢ's expression. Unused for
    /// off-diagonal pushes.
    uint8_t arg2;

    /// Multiplicity of the contribution (1 or 2).
    ///
    /// For an off-diagonal push, it's 2 if j = k because h(vⱼ, vₖ) and
    /// h(vₖ, vⱼ) are the same symmetric entry. For a diagonal push, it's 2 if
    /// arg1 ≠ arg2 but k1 = k2 (e.g., x * x), since the (arg1, arg2) and
    /// (arg2, arg1) terms land on the same entry.
    uint8_t coeff;
  };

  /// Adds coeff ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ) to h(vⱼ, vₖ), where vⱼ and vₖ are arguments
  /// of node vᵢ's expression.
  ///
  /// Creating adds the second-order contribution of node vᵢ's own expression,
  /// weighted by its adjoint, to its arguments's Hessian entries. These are
  /// only recorded for nonlinear expressions and stored in m_create_ops.
  struct CreateOp {
    /// Slot in m_slots of the entry h(vⱼ, vₖ) to add to.
    uint32_t dest;

    /// Argument position of vⱼ in node vᵢ's expression.
    uint8_t arg1;

    /// Argument position of vₖ in node vᵢ's expression.
    uint8_t arg2;

    /// Multiplicity of the contribution (1 or 2). It's 2 if arg1 ≠ arg2 but
    /// j = k (e.g., x * x), since the (arg1, arg2) and (arg2, arg1) terms land
    /// on the same entry.
    uint8_t coeff;
  };

  /// The ends of a tape node's operations.
  ///
  /// Each tape node's operations are stored contiguously in m_push_ops,
  /// m_push_diag_ops, and m_create_ops, in tape order. Like the row pointers of
  /// a CSR matrix, each member is the exclusive end index of the node's
  /// operations in the corresponding array, and the node's operations begin at
  /// the previous node's ends (or zero for the first node). For example, node
  /// n's pushes are m_push_ops[m_op_ends[n - 1].push, m_op_ends[n].push).
  ///
  /// Grouping the operations by node lets compute() replay them while the
  /// node's partial derivatives and adjoint are available.
  struct OpEnds {
    /// End index in m_push_ops.
    uint32_t push;

    /// End index in m_push_diag_ops.
    uint32_t push_diag;

    /// End index in m_create_ops.
    uint32_t create;
  };

  Variable<Scalar> m_variable;
  VariableMatrix<Scalar> m_wrt;

  detail::AdjointTape<Scalar> m_tape;

  /// The ends of each tape node's operations
  gch::small_vector<OpEnds> m_op_ends;

  gch::small_vector<PushOp> m_push_ops;
  gch::small_vector<PushOp> m_push_diag_ops;
  gch::small_vector<CreateOp> m_create_ops;

  /// Value of each structurally nonzero entry h(vⱼ, vₖ) of the expression
  /// graph's Hessian
  gch::small_vector<Scalar> m_slots;

  gch::small_vector<detail::SlotMapEntry> m_slot_map;

  Eigen::SparseMatrix<Scalar> m_H{m_wrt.rows(), m_wrt.rows()};

  /// Runs edge pushing structurally and records the operations it performs.
  ///
  /// Implements edge pushing as described by figure 4 on p. 14 of [1].
  ///
  /// [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
  ///     efficient Hessian computation via automatic differentiation", 2016.
  ///     https://sci-hub.st/10.1007/s12532-016-0100-3
  void record_pushes() {
    using enum ExpressionType;
    constexpr auto SKIPPED_CHILD = detail::AdjointTape<Scalar>::SKIPPED_CHILD;

    // Maps h(vⱼ, vₖ) with j ≤ k to its slot. Stored at index j.
    gch::small_vector<flat_map<uint32_t, uint32_t>> h(m_tape.top_list().size());
    uint32_t num_slots = 0;
    auto get_slot = [&](uint32_t j, uint32_t k) -> uint32_t {
      auto [it, inserted] =
          h[std::min(j, k)].try_emplace(std::max(j, k), num_slots);
      if (inserted) {
        ++num_slots;
      }
      return it->second;
    };

    // The tape's nodes are the dependent variables vᵢ in topological order,
    // with structurally zero partial derivatives ∂ϕᵢ/∂vₖ marked as
    // SKIPPED_CHILD. Each loop iteration records the pushing and creating steps
    // for vᵢ. The adjoint step a(vₖ) += ∂ϕᵢ/∂vₖ a(vᵢ) is recorded by the
    // AdjointTape.
    for (const auto& node : m_tape.nodes()) {
      const uint32_t i = node.idx;
      const auto& children = node.children;

      // Pushing
      //
      // for all vᵢ, vⱼ such that h(vᵢ, vⱼ) ≠ 0
      //   for all vₖ such that ∂ϕᵢ/∂vₖ ≠ 0
      //     if i ≠ j
      //       for all unordered pairs (vⱼ, vₖ) such that vⱼ < vᵢ or vₖ < vᵢ
      //         if j = k
      //           h(vⱼ, vₖ) += 2 ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
      //         else
      //           h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
      //     else
      //       for all unordered pairs (vₖ₁, vₖ₂) such that vₖ₁ < vᵢ or vₖ₂ < vᵢ
      //         if k1 = k2
      //           h(vₖ₁, vₖ₂) += 2 ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vⱼ)
      //         else
      //           h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vⱼ)
      //
      // Pushed entries are always stored at nodes after vᵢ, so h[i] isn't
      // modified while it's iterated over.
      for (const auto& [j, src] : h[i]) {
        if (i != j) {
          // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
          for (size_t arg = 0; arg < children.size(); ++arg) {
            const uint32_t k = children[arg];
            if (k != SKIPPED_CHILD) {
              m_push_ops.push_back({get_slot(j, k), src,
                                    static_cast<uint8_t>(arg), 0,
                                    static_cast<uint8_t>(j == k ? 2 : 1)});
            }
          }
        } else {
          // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
          for (size_t arg1 = 0; arg1 < children.size(); ++arg1) {
            for (size_t arg2 = 0; arg2 <= arg1; ++arg2) {
              const uint32_t k1 = children[arg1];
              const uint32_t k2 = children[arg2];
              if (k1 != SKIPPED_CHILD && k2 != SKIPPED_CHILD) {
                m_push_diag_ops.push_back(
                    {get_slot(k1, k2), src, static_cast<uint8_t>(arg1),
                     static_cast<uint8_t>(arg2),
                     static_cast<uint8_t>(arg1 != arg2 && k1 == k2 ? 2 : 1)});
              }
            }
          }
        }
      }

      // Creating
      //
      // if a(vᵢ) ≠ 0
      //   for all unordered pairs (vⱼ, vₖ) such that ∂²ϕᵢ/∂vⱼ∂vₖ ≠ 0
      //     if j = k
      //       h(vⱼ, vₖ) += 2 ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
      //     else
      //       h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
      //
      // The tape only contains nodes that a(vᵢ) can be nonzero for.
      if (node.expr->type() > LINEAR) {
        detail::ArgArray<detail::ArgArray<std::optional<Scalar>>> H;
        node.expr->hess(H);

        // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
        //
        // Second derivatives are structurally zero if either first derivative
        // is.
        for (size_t arg1 = 0; arg1 < children.size(); ++arg1) {
          for (size_t arg2 = 0; arg2 <= arg1; ++arg2) {
            const uint32_t j = children[arg1];
            const uint32_t k = children[arg2];
            if (j != SKIPPED_CHILD && k != SKIPPED_CHILD &&
                H[arg1][arg2].has_value()) {
              m_create_ops.push_back(
                  {get_slot(j, k), static_cast<uint8_t>(arg1),
                   static_cast<uint8_t>(arg2),
                   static_cast<uint8_t>(arg1 != arg2 && j == k ? 2 : 1)});
            }
          }
        }
      }

      m_op_ends.push_back({static_cast<uint32_t>(m_push_ops.size()),
                           static_cast<uint32_t>(m_push_diag_ops.size()),
                           static_cast<uint32_t>(m_create_ops.size())});
    }

    m_slots.resize(num_slots);

    // Map Hessian entries between elements of wrt to m_H's nonzeros
    gch::small_vector<detail::TapeEntry> entries;
    for (size_t j = 0; j < h.size(); ++j) {
      int row = m_tape.col(j);
      if (row == -1) {
        continue;
      }

      for (const auto& [k, slot] : h[j]) {
        int col = m_tape.col(k);

        // If indices don't refer to element in wrt, skip this value
        if (col == -1) {
          continue;
        }

        if constexpr (UpLo == Eigen::Lower) {
          // In lower triangle, row index ≥ column index
          entries.push_back({slot, std::max(row, col), std::min(row, col)});
        } else {
          entries.push_back({slot, row, col});
          if (row != col) {
            entries.push_back({slot, col, row});
          }
        }
      }
    }

    m_slot_map = detail::make_slot_map(m_H, entries);
  }

  /// Replays the recorded adjoint and edge pushing operations with the
  /// expression graph's current values, then writes the result to m_H.
  void compute() {
    using S = Scalar;

    std::ranges::fill(m_slots, S(0));

    auto adjoints = m_tape.adjoints();
    m_tape.replay([&](size_t node_idx, const auto& node, const auto& g) {
      const auto& ends = m_op_ends[node_idx];
      OpEnds begins = node_idx == 0 ? OpEnds{0, 0, 0} : m_op_ends[node_idx - 1];

      for (uint32_t op_idx = begins.push; op_idx < ends.push; ++op_idx) {
        const auto& op = m_push_ops[op_idx];
        m_slots[op.dest] += S(op.coeff) * *g[op.arg1] * m_slots[op.src];
      }

      for (uint32_t op_idx = begins.push_diag; op_idx < ends.push_diag;
           ++op_idx) {
        const auto& op = m_push_diag_ops[op_idx];
        m_slots[op.dest] +=
            S(op.coeff) * *g[op.arg1] * *g[op.arg2] * m_slots[op.src];
      }

      if (begins.create < ends.create) {
        S a = adjoints[node.slot];

        detail::ArgArray<detail::ArgArray<std::optional<S>>> H;
        node.expr->hess(H);
        for (uint32_t op_idx = begins.create; op_idx < ends.create; ++op_idx) {
          const auto& op = m_create_ops[op_idx];
          m_slots[op.dest] += S(op.coeff) * *H[op.arg1][op.arg2] * a;
        }
      }
    });

    detail::copy_slots(m_H, m_slot_map, m_slots);
  }

  /// Returns the variable's Hessian tree.
  ///
  /// This function lazily allocates variables, so elements of the returned
  /// VariableMatrix will be empty if the corresponding element of wrt had no
  /// adjoint. Ensure Variable::expr isn't nullptr before calling member
  /// functions.
  ///
  /// @return The variable's Hessian tree.
  VariableMatrix<Scalar> hessian_tree() const {
    using enum ExpressionType;

    slp_assert(m_wrt.cols() == 1);

    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    // Implements edge pushing as described by figure 4 on p. 14 of [1].
    //
    // [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
    //     efficient Hessian computation via automatic differentiation", 2016.
    //     https://sci-hub.st/10.1007/s12532-016-0100-3

    auto top_list = detail::topological_sort(m_variable.expr);

    if (top_list.empty()) {
      return VariableMatrix<Scalar>{detail::empty, m_wrt.rows(), m_wrt.rows()};
    }

    // Sort dependent variables before independent ones while maintaining
    // relative order (precondition of edge pushing)
    size_t top_list_end = std::distance(
        top_list.begin(),
        std::stable_partition(top_list.begin(), top_list.end(),
                              [](const auto& elem) { return !elem->is_leaf; }));

    // Initialize column each expression's adjoint occupies in the Hessian
    for (size_t col = 0; col < m_wrt.size(); ++col) {
      m_wrt[col].expr->scratch = col;
    }

    gch::small_vector<int> col_list;
    for (auto& node : top_list) {
      col_list.emplace_back(node->scratch);
    }

    // Reset col to -1
    for (auto& node : m_wrt) {
      node.expr->scratch = -1;
    }

    // Maps h(vⱼ, vₖ) with j ≤ k to its expression. Stored at index j.
    gch::small_vector<flat_map<size_t, detail::ExpressionPtr<Scalar>>>
        hessian_expr(top_list.size());

    // Append value to Hessian mapping
    auto push_edge = [&](size_t j, size_t k,
                         detail::ExpressionPtr<Scalar> value) {
      // Sort parent index before child index
      hessian_expr[std::min(j, k)][std::max(j, k)] += std::move(value);
    };

    auto ptr_1 = detail::constant_ptr(Scalar(1));
    auto ptr_2 = detail::constant_ptr(Scalar(2));

    // Temporarily store each node's index in top_list in scratch
    for (size_t i = 0; i < top_list.size(); ++i) {
      top_list[i]->scratch = i;
    }

    // Adjoint of each node in top_list
    gch::small_vector<detail::ExpressionPtr<Scalar>> adjoints(top_list.size());

    // Set root node's adjoint to 1 since df/df is 1
    adjoints[0] = ptr_1;

    for (size_t i = 0; i < top_list_end; ++i) {
      const auto& v_i = top_list[i];

      // Get node arguments
      detail::ArgArray<detail::Expression<Scalar>*> args;
      size_t num_args = 0;
      v_i->visit_args([&](auto* arg) { args[num_args++] = arg; });

      // Compute node gradients. Constant arguments are skipped since their
      // derivatives are never used.
      detail::ArgArray<detail::ExpressionPtr<Scalar>> g;
      v_i->grad_expr(g);
      for (size_t arg = 0; arg < num_args; ++arg) {
        if (args[arg]->type() == CONSTANT) {
          g[arg] = nullptr;
        }
      }

      // Adjoints (a null adjoint is zero, so it has nothing to contribute)
      if (adjoints[i] != nullptr) {
        for (size_t arg = 0; arg < num_args; ++arg) {
          if (g[arg] != nullptr) {
            adjoints[args[arg]->scratch] += g[arg] * adjoints[i];
          }
        }
      }

      // Pushing
      //
      // for all vᵢ, vⱼ such that h(vᵢ, vⱼ) ≠ 0
      //   for all vₖ such that ∂ϕᵢ/∂vₖ ≠ 0
      //     if i ≠ j
      //       for all unordered pairs (vⱼ, vₖ) such that vⱼ < vᵢ or vₖ < vᵢ
      //         if j = k
      //           h(vⱼ, vₖ) += 2 ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
      //         else
      //           h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
      //     else
      //       for all unordered pairs (vₖ₁, vₖ₂) such that vₖ₁ < vᵢ or vₖ₂ < vᵢ
      //         if k1 = k2
      //           h(vₖ₁, vₖ₂) += 2 ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vⱼ)
      //         else
      //           h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vⱼ)
      for (const auto& [j, h_i_j] : hessian_expr[i]) {
        if (i != j) {
          // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
          for (size_t arg = 0; arg < num_args; ++arg) {
            if (g[arg] != nullptr) {
              size_t k = args[arg]->scratch;
              push_edge(j, k, (j == k ? ptr_2 : ptr_1) * g[arg] * h_i_j);
            }
          }
        } else {
          // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
          for (size_t arg1 = 0; arg1 < num_args; ++arg1) {
            for (size_t arg2 = 0; arg2 <= arg1; ++arg2) {
              if (g[arg1] != nullptr && g[arg2] != nullptr) {
                size_t k1 = args[arg1]->scratch;
                size_t k2 = args[arg2]->scratch;
                push_edge(k1, k2,
                          (arg1 != arg2 && k1 == k2 ? ptr_2 : ptr_1) * g[arg1] *
                              g[arg2] * h_i_j);
              }
            }
          }
        }
      }

      // Creating
      //
      // if a(vᵢ) ≠ 0
      //   for all unordered pairs (vⱼ, vₖ) such that ∂²ϕᵢ/∂vⱼ∂vₖ ≠ 0
      //     if j = k
      //       h(vⱼ, vₖ) += 2 ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
      //     else
      //       h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
      if (adjoints[i] != nullptr && v_i->type() > LINEAR) {
        detail::ArgArray<detail::ArgArray<detail::ExpressionPtr<Scalar>>> H;
        v_i->hess_expr(H);

        // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
        //
        // Second derivatives are structurally zero if either first derivative
        // is.
        for (size_t arg1 = 0; arg1 < num_args; ++arg1) {
          for (size_t arg2 = 0; arg2 <= arg1; ++arg2) {
            if (g[arg1] != nullptr && g[arg2] != nullptr &&
                H[arg1][arg2] != nullptr) {
              size_t j = args[arg1]->scratch;
              size_t k = args[arg2]->scratch;
              push_edge(j, k,
                        (arg1 != arg2 && j == k ? ptr_2 : ptr_1) *
                            H[arg1][arg2] * adjoints[i]);
            }
          }
        }
      }
    }

    // Move Hessian tree to return value
    VariableMatrix<Scalar> H{detail::empty, m_wrt.rows(), m_wrt.rows()};
    for (int row = 0; row < static_cast<int>(m_wrt.rows()); ++row) {
      int row_idx = m_wrt[row].expr->scratch;

      // If wrt element isn't in the graph, its Hessian row is zero
      if (row_idx == -1) {
        continue;
      }

      for (const auto& node : hessian_expr[row_idx]) {
        const auto& col_idx = node.first;
        Variable value{node.second};

        int col = col_list[col_idx];

        // If indices don't refer to element in wrt, skip this value
        if (col == -1) {
          continue;
        }

        if constexpr (UpLo == Eigen::Lower) {
          // In lower triangle, row index ≥ column index
          if (row > col) {
            H[row, col] = value;
          } else {
            H[col, row] = value;
          }
        } else {
          H[row, col] = value;
          if (row != col) {
            H[col, row] = value;
          }
        }
      }
    }

    for (auto& node : top_list) {
      node->scratch = -1;
    }

    return H;
  }
};

// @cond Suppress Doxygen
extern template class EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
Hessian<double, Eigen::Lower | Eigen::Upper>;
// @endcond

}  // namespace slp
