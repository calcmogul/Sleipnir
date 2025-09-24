// Copyright (c) Sleipnir contributors

#pragma once

#include <array>
#include <map>
#include <ranges>
#include <set>
#include <utility>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp::detail {

/**
 * This class is an adaptor type that performs value updates of an expression's
 * adjoint graph.
 */
class AdjointExpressionGraph {
 public:
  /**
   * Generates the adjoint graph for the given expression.
   *
   * @param root The root node of the expression.
   */
  explicit AdjointExpressionGraph(const Variable& root)
      : m_top_list{topological_sort(root.expr)} {
    for (size_t i = 0; i < m_top_list.size(); ++i) {
      const auto& node = m_top_list[i];
      node->idx = i;
      m_col_list.emplace_back(node->col);
    }
  }

  /**
   * Update the values of all nodes in this adjoint graph based on the values of
   * their dependent nodes.
   */
  void update_values() { detail::update_values(m_top_list); }

  /**
   * Returns the variable's gradient tree.
   *
   * This function lazily allocates variables, so elements of the returned
   * VariableMatrix will be empty if the corresponding element of wrt had no
   * adjoint. Ensure Variable::expr isn't nullptr before calling member
   * functions.
   *
   * @param wrt Variables with respect to which to compute the gradient.
   * @return The variable's gradient tree.
   */
  VariableMatrix generate_gradient_tree(const VariableMatrix& wrt) const {
    slp_assert(wrt.cols() == 1);

    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    if (m_top_list.empty()) {
      return VariableMatrix{VariableMatrix::empty, wrt.rows(), 1};
    }

    // Set root node's adjoint to 1 since df/df is 1
    m_top_list[0]->adjoint_expr = make_expression_ptr<ConstExpression>(1.0);

    // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
    // multiplied by dy/dx. If there are multiple "paths" from the root node to
    // variable; the variable's adjoint is the sum of each path's adjoint
    // contribution.
    for (auto& node : m_top_list) {
      auto& lhs = node->args[0];
      auto& rhs = node->args[1];

      if (lhs != nullptr) {
        lhs->adjoint_expr += node->grad_expr_l(lhs, rhs, node->adjoint_expr);
        if (rhs != nullptr) {
          rhs->adjoint_expr += node->grad_expr_r(lhs, rhs, node->adjoint_expr);
        }
      }
    }

    // Move gradient tree to return value
    VariableMatrix grad{VariableMatrix::empty, wrt.rows(), 1};
    for (int row = 0; row < grad.rows(); ++row) {
      grad[row] = Variable{std::move(wrt[row].expr->adjoint_expr)};
    }

    // Unlink adjoints to avoid circular references between them and their
    // parent expressions. This ensures all expressions are returned to the free
    // list.
    for (auto& node : m_top_list) {
      node->adjoint_expr = nullptr;
    }

    return grad;
  }

  /**
   * Updates the adjoints in the expression graph (computes the gradient) then
   * appends the adjoints of wrt to the sparse matrix triplets.
   *
   * @param triplets The sparse matrix triplets.
   * @param row The row of wrt.
   * @param wrt Vector of variables with respect to which to compute the
   *   Jacobian.
   */
  void append_gradient_triplets(
      gch::small_vector<Eigen::Triplet<double>>& triplets, int row,
      const VariableMatrix& wrt) const {
    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    // If wrt has fewer nodes than graph, zero wrt's adjoints
    if (static_cast<size_t>(wrt.rows()) < m_top_list.size()) {
      for (const auto& elem : wrt) {
        elem.expr->adjoint = 0.0;
      }
    }

    if (m_top_list.empty()) {
      return;
    }

    // Set root node's adjoint to 1 since df/df is 1
    m_top_list[0]->adjoint = 1.0;

    // Zero the rest of the adjoints
    for (auto& node : m_top_list | std::views::drop(1)) {
      node->adjoint = 0.0;
    }

    // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
    // multiplied by dy/dx. If there are multiple "paths" from the root node to
    // variable; the variable's adjoint is the sum of each path's adjoint
    // contribution.
    for (const auto& node : m_top_list) {
      auto& lhs = node->args[0];
      auto& rhs = node->args[1];

      if (lhs != nullptr) {
        if (rhs != nullptr) {
          lhs->adjoint += node->grad_l(lhs->val, rhs->val, node->adjoint);
          rhs->adjoint += node->grad_r(lhs->val, rhs->val, node->adjoint);
        } else {
          lhs->adjoint += node->grad_l(lhs->val, 0.0, node->adjoint);
        }
      }
    }

    // If wrt has fewer nodes than graph, iterate over wrt
    if (static_cast<size_t>(wrt.rows()) < m_top_list.size()) {
      for (int col = 0; col < wrt.rows(); ++col) {
        const auto& node = wrt[col].expr;

        // Append adjoints of wrt to sparse matrix triplets
        if (node->adjoint != 0.0) {
          triplets.emplace_back(row, col, node->adjoint);
        }
      }
    } else {
      for (const auto& [col, node] : std::views::zip(m_col_list, m_top_list)) {
        // Append adjoints of wrt to sparse matrix triplets
        if (col != -1 && node->adjoint != 0.0) {
          triplets.emplace_back(row, col, node->adjoint);
        }
      }
    }
  }

  /**
   * Updates the adjoints in the expression graph (computes the Hessian) then
   * appends the adjoints of wrt to the sparse matrix triplets.
   *
   * @tparam UpLo Which part of the Hessian to compute (Lower or Lower | Upper).
   * @param triplets The sparse matrix triplets.
   * @param wrt Vector of variables with respect to which to compute the
   *   Jacobian.
   */
  template <int UpLo>
    requires(UpLo == Eigen::Lower) || (UpLo == (Eigen::Lower | Eigen::Upper))
  void append_hessian_triplets(
      gch::small_vector<Eigen::Triplet<double>>& triplets,
      const VariableMatrix& wrt) const {
    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    // Implements figure 4 on p. 406 of [1].
    //
    // [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
    //     efficient Hessian computation via automatic differentiation", 2016.
    //     https://sci-hub.st/10.1007/s12532-016-0100-3

    if (static_cast<size_t>(wrt.rows()) < m_top_list.size()) {
      for (const auto& elem : wrt) {
        elem.expr->adjoint = 0.0;
      }
    }

    if (m_top_list.empty()) {
      return;
    }

    // Map from m_top_list indices to Hessian value
    std::map<std::pair<size_t, size_t>, double> h;
    auto h_sym = [&h](size_t j, size_t k) -> double& {
      return j > k ? h[std::pair{j, k}] : h[std::pair{k, j}];
    };

    std::set<size_t> S;
    S.insert(0);

    triplets.emplace_back(0, 0, 0.0);
#if 0
    // Get index of Expression* in m_top_list
    auto get_index = [&](Expression* expr) {
      return std::distance(
          m_top_list.begin(),
          std::find(m_top_list.begin(), m_top_list.end(), expr));
    };
#endif

    m_top_list[0]->adjoint = 1.0;

#if 1
    auto print_tape = [&] {
      for (const auto& k : S | std::views::reverse) {
        slp::println("  a({}) = {}", k, m_top_list[k]->adjoint);
      }
      for (const auto& [indices, value] : h) {
        slp::println("  h({}, {}) = {}", indices.first, indices.second, value);
      }
    };

    slp::println("init");
    print_tape();
#endif

    h[std::pair{0, 0}] = 0.0;

    for (size_t i = 0; i < m_top_list.size(); ++i) {
      const auto& v_i = m_top_list[i];
      const auto& v_lhs = v_i->args[0];
      const auto& v_rhs = v_i->args[1];

      // If a node has no children, keep it in the live variable set instead of
      // replacing it with its children
      if (v_lhs == nullptr && v_rhs == nullptr) {
        continue;
      }

#if 1
      slp::println("\ni = {}:", i);
      if (v_lhs != nullptr) {
        slp::println("  v_lhs = {}", v_lhs->idx);
        if (v_rhs != nullptr) {
          slp::println("  v_rhs = {}", v_rhs->idx);
        }
      }
#endif

      double w = v_i->adjoint;

      // r: S → ℝ as r(v) = h(vᵢ, v)
      auto r = h;

      // Maintain the live variable set
      {
        // Remove vᵢ from S
        S.erase(i);

        // a(vᵢ) = 0
        v_i->adjoint = 0.0;

        // h(vᵢ, S) = 0
        for (const auto& k : S) {
          h[std::pair{i, k}] = 0.0;
        }

        // h(S, vᵢ) = 0
        for (const auto& k : S) {
          h[std::pair{k, i}] = 0.0;
        }

        // for all vⱼ < vᵢ
        if (v_lhs != nullptr) {
          size_t j = v_lhs->idx;

          // if vⱼ ∉ S
          if (!S.contains(j)) {
            // S = S ∪ vⱼ
            S.insert(j);

            // a(vⱼ) = 0
            v_lhs->adjoint = 0.0;

            // h(vⱼ, S) = 0
            for (const auto& k : S) {
              h[std::pair{j, k}] = 0.0;
            }
          }

          if (v_rhs != nullptr) {
            size_t j = v_rhs->idx;

            // if vⱼ ∉ S
            if (!S.contains(j)) {
              // S = S ∪ vⱼ
              S.insert(j);

              // a(vⱼ) = 0
              v_rhs->adjoint = 0.0;

              // h(vⱼ, S) = 0
              for (const auto& k : S) {
                h[std::pair{j, k}] = 0.0;
              }
            }
          }
        }
      }

      // Update the adjoints a(S)
      //
      // if w ≠ 0
      //   for all vⱼ < vᵢ
      //     a(vⱼ) += ∂ϕᵢ/∂vⱼ w
      if (v_lhs != nullptr && w != 0.0) {
        if (v_rhs != nullptr) {
          v_lhs->adjoint += v_i->grad_l(v_lhs->val, v_rhs->val, w);
          v_rhs->adjoint += v_i->grad_r(v_lhs->val, v_rhs->val, w);
        } else {
          v_lhs->adjoint += v_i->grad_l(v_lhs->val, 0.0, w);
        }
      }

      // Update the Hessian h(S, S)
      //
      // Pushing
      {
        slp::println("  pushing loop");
        // for all vⱼ ≠ vᵢ such that r(vⱼ) ≠ 0
        for (const auto& [indices, value] : r) {
          if (!((indices.first == i) ^ (indices.second == i)) || value == 0.0) {
            continue;
          }

          size_t j = indices.first == i ? indices.second : indices.first;

          // for all vₖ such that ∂ϕᵢ/∂vₖ ≠ 0
          if (v_lhs != nullptr) {
            size_t k = v_lhs->idx;

            slp::println("      push l");
            if (j == k) {
              // h(vⱼ, vₖ) += 2 ∂ϕᵢ/∂vₖ r(vⱼ)
              h[std::pair{j, k}] +=
                  2.0 *
                  v_i->grad_l(v_lhs->val, v_rhs ? v_rhs->val : 0.0, value);
            } else {
              // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ r(vⱼ)
              h_sym(j, k) +=
                  v_i->grad_l(v_lhs->val, v_rhs ? v_rhs->val : 0.0, value);
            }

            if (v_rhs != nullptr) {
              size_t k = v_rhs->idx;

              slp::println("      push r");
              if (j == k) {
                // h(vⱼ, vₖ) += 2 ∂ϕᵢ/∂vₖ r(vⱼ)
                h[std::pair{j, k}] +=
                    2.0 * v_i->grad_r(v_lhs->val, v_rhs->val, value);
              } else {
                // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ r(vⱼ)
                h_sym(j, k) += v_i->grad_r(v_lhs->val, v_rhs->val, value);
              }
            }
          }
        }

        // if r(vᵢ) ≠ 0
        if (double r_v_i = h[std::pair{i, i}]; r_v_i != 0.0) {
          if (v_lhs != nullptr) {
            size_t lhs = v_lhs->idx;
            size_t rhs = v_rhs->idx;

            // for all unordered pairs (vⱼ, vₖ) such that ∂ϕᵢ/∂vⱼ ∂ϕᵢ/∂vₖ ≠ 0
            //   h(vⱼ, vₖ) += ∂ϕᵢ/∂vⱼ ∂ϕᵢ/∂vₖ r(vᵢ)
            double g_l = v_i->grad_l(v_lhs->val, v_rhs ? v_rhs->val : 0.0, 1.0);
            double g_r = v_i->grad_r(v_lhs->val, v_rhs ? v_rhs->val : 0.0, 1.0);
            if (g_l * g_l != 0.0) {
              slp::println("  push 3 ll");
              h[std::pair{lhs, lhs}] += g_l * g_l * r_v_i;
            }
            if (g_l * g_r != 0.0) {
              slp::println("  push 3 lr");
              h_sym(lhs, rhs) += g_l * g_r * r_v_i;
            }
            if (g_r * g_r != 0.0) {
              slp::println("  push 3 rr");
              h[std::pair{rhs, rhs}] += g_r * g_r * r_v_i;
            }
          }
        }
      }

      // Creating
      //
      // if w ≠ 0
      if (w != 0.0 && v_lhs != nullptr) {
        size_t lhs = v_lhs->idx;

        // for all unordered pairs (vⱼ, vₖ) such that ∂²ϕᵢ/∂vⱼ∂vₖ ≠ 0
        //   h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ w
        if (double h_ll =
                v_i->hess_ll(v_lhs->val, v_rhs ? v_rhs->val : 0.0, 1.0);
            h_ll != 0.0) {
          slp::println("  create ll");
          h[std::pair{lhs, lhs}] += h_ll * w;
        }
        if (v_rhs != nullptr) {
          size_t rhs = v_rhs->idx;

          if (double h_lr =
                  v_i->hess_lr(v_lhs->val, v_rhs ? v_rhs->val : 0.0, 1.0);
              h_lr != 0.0) {
            slp::println("  create lr");
            if (lhs == rhs) {
              h[std::pair{lhs, rhs}] += 2.0 * h_lr * w;
            } else {
              h_sym(lhs, rhs) += h_lr * w;
            }
          }
          if (double h_rr =
                  v_i->hess_rr(v_lhs->val, v_rhs ? v_rhs->val : 0.0, 1.0);
              h_rr != 0.0) {
            slp::println("  create rr");
            h[std::pair{rhs, rhs}] += h_rr * w;
          }
        }
      }

#if 1
      slp::println("");
      print_tape();
#endif
    }

    // Iterate over h and append triplets whose indices are in live variable set
    // and denote variables in wrt
    for (const auto& [indices, value] : h) {
      if (!S.contains(indices.first) || !S.contains(indices.second)) {
        continue;
      }

      int row = m_col_list[indices.first];
      int col = m_col_list[indices.second];

      // If indices don't refer to element in wrt, skip this value
      if (row == -1 || col == -1) {
        continue;
      }

      if constexpr (UpLo == Eigen::Lower) {
        triplets.emplace_back(row, col, value);
      } else {
        triplets.emplace_back(row, col, value);
        if (row != col) {
          triplets.emplace_back(col, row, value);
        }
      }
    }
  }

 private:
  // Topological sort of graph from parent to child
  gch::small_vector<Expression*> m_top_list;

  // List that maps nodes to their respective column
  gch::small_vector<int> m_col_list;
};

}  // namespace slp::detail
