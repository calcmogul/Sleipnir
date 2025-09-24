// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <ranges>
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
    // Sort dependent variables before independent ones while maintaining
    // relative order (precondition of Edge Pushing)
    std::stable_partition(
        m_top_list.begin(), m_top_list.end(),
        [](const auto& elem) { return elem->args[0] != nullptr; });

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
    m_top_list[0]->adjoint_expr = constant_ptr(1.0);

    // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
    // multiplied by dy/dx. If there are multiple "paths" from the root node to
    // variable; the variable's adjoint is the sum of each path's adjoint
    // contribution.
    for (auto& node : m_top_list) {
      auto& lhs = node->args[0];
      auto& rhs = node->args[1];

      if (lhs != nullptr) {
        if (rhs != nullptr) {
          // Binary operator
          lhs->adjoint_expr += node->grad_expr_l(lhs, rhs, node->adjoint_expr);
          rhs->adjoint_expr += node->grad_expr_r(lhs, rhs, node->adjoint_expr);
        } else {
          // Unary operator
          lhs->adjoint_expr += node->grad_expr_l(lhs, rhs, node->adjoint_expr);
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
   * Returns the variable's Hessian tree.
   *
   * This function lazily allocates variables, so elements of the returned
   * VariableMatrix will be empty if the corresponding element of wrt had no
   * adjoint. Ensure Variable::expr isn't nullptr before calling member
   * functions.
   *
   * @tparam UpLo Which part of the Hessian to compute (Lower or Lower | Upper).
   * @param wrt Variables with respect to which to compute the Hessian.
   * @return The variable's Hessian tree.
   */
  template <int UpLo>
    requires(UpLo == Eigen::Lower) || (UpLo == (Eigen::Lower | Eigen::Upper))
  VariableMatrix generate_hessian_tree(const VariableMatrix& wrt) const {
    slp_assert(wrt.cols() == 1);

    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    // Implements Edge Pushing as described by figure 4 on p. 406 of [1].
    //
    // [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
    //     efficient Hessian computation via automatic differentiation", 2016.
    //     https://sci-hub.st/10.1007/s12532-016-0100-3

    if (m_top_list.empty()) {
      return VariableMatrix{VariableMatrix::empty, wrt.rows(), 1};
    }

    // Hessian mapping from expression graph index pair to value
    auto h = [this](size_t j, size_t k) -> ExpressionPtr& {
      // Sort parent index before child index
      if (j < k) {
        return m_top_list[j]->hessian_expr[k];
      } else {
        return m_top_list[k]->hessian_expr[j];
      }
    };

    auto ptr_1 = constant_ptr(1.0);
    auto ptr_2 = constant_ptr(2.0);

    // Set root node's adjoint to 1 since df/df is 1
    m_top_list[0]->adjoint_expr = ptr_1;

    for (size_t i = 0; i < m_top_list.size(); ++i) {
      const auto& v_i = m_top_list[i];
      const auto& v_lhs = v_i->args[0];
      const auto& v_rhs = v_i->args[1];

      // If a node has no children, we've encountered the independent variables
      // and there's no more edges to push
      if (v_lhs == nullptr) {
        break;
      }

      // Adjoints
      if (v_rhs != nullptr) {
        // Binary operator
        v_lhs->adjoint_expr +=
            v_i->grad_expr_l(v_lhs, v_rhs, v_i->adjoint_expr);
        v_rhs->adjoint_expr +=
            v_i->grad_expr_r(v_lhs, v_rhs, v_i->adjoint_expr);
      } else {
        // Unary operator
        v_lhs->adjoint_expr +=
            v_i->grad_expr_l(v_lhs, v_rhs, v_i->adjoint_expr);
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
      for (const auto& [j, h_i_j] : v_i->hessian_expr) {
        if (v_rhs != nullptr) {
          // Binary operator
          size_t k_l = v_lhs->idx;
          size_t k_r = v_rhs->idx;

          auto g_l = v_i->grad_expr_l(v_lhs, v_rhs, ptr_1);
          auto g_r = v_i->grad_expr_r(v_lhs, v_rhs, ptr_1);

          if (i != j) {
            // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
            h(j, k_l) += (j == k_l ? ptr_2 : ptr_1) * g_l * h_i_j;
            h(j, k_r) += (j == k_r ? ptr_2 : ptr_1) * g_r * h_i_j;
          } else {
            // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
            h(k_l, k_l) += g_l * g_l * h_i_j;
            h(k_l, k_r) += (k_l == k_r ? ptr_2 : ptr_1) * g_l * g_r * h_i_j;
            h(k_r, k_r) += g_r * g_r * h_i_j;
          }
        } else {
          // Unary operator
          size_t k_l = v_lhs->idx;

          auto g_l = v_i->grad_expr_l(v_lhs, v_rhs, ptr_1);

          if (i != j) {
            // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
            h(j, k_l) += (j == k_l ? ptr_2 : ptr_1) * g_l * h_i_j;
          } else {
            // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
            h(k_l, k_l) += g_l * g_l * h_i_j;
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
      if (v_i->adjoint_expr != nullptr) {
        if (v_rhs != nullptr) {
          // Binary operator
          size_t k_l = v_lhs->idx;
          size_t k_r = v_rhs->idx;

          if (auto h_ll = v_i->hess_expr_ll(v_lhs, v_rhs, ptr_1);
              h_ll != nullptr) {
            // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
            h(k_l, k_l) += h_ll * v_i->adjoint_expr;
          }

          if (auto h_lr = v_i->hess_expr_lr(v_lhs, v_rhs, ptr_1);
              h_lr != nullptr) {
            // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
            h(k_l, k_r) +=
                (k_l == k_r ? ptr_2 : ptr_1) * h_lr * v_i->adjoint_expr;
          }

          if (auto h_rr = v_i->hess_expr_rr(v_lhs, v_rhs, ptr_1);
              h_rr != nullptr) {
            // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
            h(k_r, k_r) += h_rr * v_i->adjoint_expr;
          }
        } else {
          // Unary operator
          size_t k_l = v_lhs->idx;

          if (auto h_ll = v_i->hess_expr_ll(v_lhs, v_rhs, ptr_1);
              h_ll != nullptr) {
            // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
            h(k_l, k_l) += h_ll * v_i->adjoint_expr;
          }
        }
      }
    }

    // Move Hessian tree to return value
    VariableMatrix H{VariableMatrix::empty, wrt.rows(), wrt.rows()};
    for (int row = 0; row < static_cast<int>(wrt.rows()); ++row) {
      for (const auto& elem : wrt[row].expr->hessian_expr) {
        const auto& col_idx = elem.first;
        Variable value{elem.second};

        int col = m_col_list[col_idx];

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

    // Unlink adjoints to avoid circular references between them and their
    // parent expressions. This ensures all expressions are returned to the free
    // list.
    for (auto& node : m_top_list) {
      node->adjoint_expr = nullptr;
      node->hessian_expr.clear();
    }

    return H;
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
    slp_assert(wrt.cols() == 1);

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
          // Binary operator
          lhs->adjoint += node->grad_l(lhs->val, rhs->val, node->adjoint);
          rhs->adjoint += node->grad_r(lhs->val, rhs->val, node->adjoint);
        } else {
          // Unary operator
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
   *   Hessian.
   */
  template <int UpLo>
    requires(UpLo == Eigen::Lower) || (UpLo == (Eigen::Lower | Eigen::Upper))
  void append_hessian_triplets(
      gch::small_vector<Eigen::Triplet<double>>& triplets,
      const VariableMatrix& wrt) const {
    slp_assert(wrt.cols() == 1);

    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    // Implements Edge Pushing as described by figure 4 on p. 406 of [1].
    //
    // [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
    //     efficient Hessian computation via automatic differentiation", 2016.
    //     https://sci-hub.st/10.1007/s12532-016-0100-3

    if (m_top_list.empty()) {
      return;
    }

    // Hessian mapping from expression graph index pair to value
    auto h = [this](size_t j, size_t k) -> double& {
      // Sort parent index before child index
      if (j < k) {
        return m_top_list[j]->hessian[k];
      } else {
        return m_top_list[k]->hessian[j];
      }
    };

    // Set root node's adjoint to 1 since df/df is 1
    m_top_list[0]->adjoint = 1.0;

    // Zero the rest of the adjoints
    for (auto& node : m_top_list | std::views::drop(1)) {
      node->adjoint = 0.0;
    }

    // Clear all Hessian mappings
    for (auto& elem : m_top_list) {
      elem->hessian.clear();
    }

    for (size_t i = 0; i < m_top_list.size(); ++i) {
      const auto& v_i = m_top_list[i];
      const auto& v_lhs = v_i->args[0];
      const auto& v_rhs = v_i->args[1];

      // If a node has no children, we've encountered the independent variables
      // and there's no more edges to push
      if (v_lhs == nullptr) {
        break;
      }

      // Adjoints
      if (v_rhs != nullptr) {
        // Binary operator
        v_lhs->adjoint += v_i->grad_l(v_lhs->val, v_rhs->val, v_i->adjoint);
        v_rhs->adjoint += v_i->grad_r(v_lhs->val, v_rhs->val, v_i->adjoint);
      } else {
        // Unary operator
        v_lhs->adjoint += v_i->grad_l(v_lhs->val, 0.0, v_i->adjoint);
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
      for (const auto& [j, h_i_j] : v_i->hessian) {
        if (v_rhs != nullptr) {
          // Binary operator
          size_t k_l = v_lhs->idx;
          size_t k_r = v_rhs->idx;

          double g_l = v_i->grad_l(v_lhs->val, v_rhs->val, 1.0);
          double g_r = v_i->grad_r(v_lhs->val, v_rhs->val, 1.0);

          if (i != j) {
            // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
            h(j, k_l) += (j == k_l ? 2.0 : 1.0) * g_l * h_i_j;
            h(j, k_r) += (j == k_r ? 2.0 : 1.0) * g_r * h_i_j;
          } else {
            // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
            h(k_l, k_l) += g_l * g_l * h_i_j;
            h(k_l, k_r) += (k_l == k_r ? 2.0 : 1.0) * g_l * g_r * h_i_j;
            h(k_r, k_r) += g_r * g_r * h_i_j;
          }
        } else {
          // Unary operator
          size_t k_l = v_lhs->idx;

          double g_l = v_i->grad_l(v_lhs->val, 0.0, 1.0);

          if (i != j) {
            // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
            h(j, k_l) += (j == k_l ? 2.0 : 1.0) * g_l * h_i_j;
          } else {
            // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
            h(k_l, k_l) += g_l * g_l * h_i_j;
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
      if (v_i->adjoint != 0.0) {
        if (v_rhs != nullptr) {
          // Binary operator
          size_t k_l = v_lhs->idx;
          size_t k_r = v_rhs->idx;

          if (double h_ll = v_i->hess_ll(v_lhs->val, v_rhs->val, 1.0);
              h_ll != 0.0) {
            // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
            h(k_l, k_l) += h_ll * v_i->adjoint;
          }

          if (double h_lr = v_i->hess_lr(v_lhs->val, v_rhs->val, 1.0);
              h_lr != 0.0) {
            // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
            h(k_l, k_r) += (k_l == k_r ? 2.0 : 1.0) * h_lr * v_i->adjoint;
          }

          if (double h_rr = v_i->hess_rr(v_lhs->val, v_rhs->val, 1.0);
              h_rr != 0.0) {
            // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
            h(k_r, k_r) += h_rr * v_i->adjoint;
          }
        } else {
          // Unary operator
          size_t k_l = v_lhs->idx;

          if (double h_ll = v_i->hess_ll(v_lhs->val, 0.0, 1.0); h_ll != 0.0) {
            // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
            h(k_l, k_l) += h_ll * v_i->adjoint;
          }
        }
      }
    }

    // Append Hessian triplets
    for (int row = 0; row < static_cast<int>(wrt.rows()); ++row) {
      for (const auto& [col_idx, value] : wrt[row].expr->hessian) {
        int col = m_col_list[col_idx];

        // If indices don't refer to element in wrt, skip this value
        if (col == -1) {
          continue;
        }

        if constexpr (UpLo == Eigen::Lower) {
          // In lower triangle, row index ≥ column index
          if (row > col) {
            triplets.emplace_back(row, col, value);
          } else {
            triplets.emplace_back(col, row, value);
          }
        } else {
          triplets.emplace_back(row, col, value);
          if (row != col) {
            triplets.emplace_back(col, row, value);
          }
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
