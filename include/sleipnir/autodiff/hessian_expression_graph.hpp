// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <iterator>
#include <ranges>
#include <utility>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/print_diagnostics.hpp"
#include "sleipnir/util/profiler.hpp"

namespace slp::detail {

/// This class is an adapter type that performs value updates of an expression
/// graph, generates a Hessian tree, or appends Hessian triplets for creating a
/// sparse Hessian matrix.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class HessianExpressionGraph {
 public:
  /// Generates the Hessian graph for the given expression.
  ///
  /// @param root The root node of the expression.
  explicit HessianExpressionGraph(const Variable<Scalar>& root)
      : m_top_list{topological_sort(root.expr)} {
    // Sort dependent variables before independent ones while maintaining
    // relative order (precondition of edge pushing)
    m_top_list_end = std::distance(
        m_top_list.begin(),
        std::stable_partition(
            m_top_list.begin(), m_top_list.end(),
            [](const auto& elem) { return elem->args[0] != nullptr; }));

    // TODO: Prune expression graph so checks for linear/quadratic later aren't
    // needed. Frontload repeated work in general.

    for (size_t i = 0; i < m_top_list.size(); ++i) {
      const auto& node = m_top_list[i];
      node->idx = i;
      m_col_list.emplace_back(node->col);
    }
  }

  /// Update the values of all nodes in this graph based on the values of their
  /// dependent nodes.
  void update_values() { detail::update_values(m_top_list); }

  /// Returns the variable's Hessian tree.
  ///
  /// This function lazily allocates variables, so elements of the returned
  /// VariableMatrix will be empty if the corresponding element of wrt had no
  /// adjoint. Ensure Variable::expr isn't nullptr before calling member
  /// functions.
  ///
  /// @tparam UpLo Which part of the Hessian to compute (Lower or Lower |
  ///     Upper).
  /// @param wrt Variables with respect to which to compute the Hessian.
  /// @return The variable's Hessian tree.
  template <int UpLo>
    requires(UpLo == Eigen::Lower) || (UpLo == (Eigen::Lower | Eigen::Upper))
  VariableMatrix<Scalar> generate_tree(
      const VariableMatrix<Scalar>& wrt) const {
    using enum ExpressionType;

    slp_assert(wrt.cols() == 1);

    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    // Implements edge pushing as described by figure 4 on p. 406 of [1].
    //
    // [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
    //     efficient Hessian computation via automatic differentiation", 2016.
    //     https://sci-hub.st/10.1007/s12532-016-0100-3

    if (m_top_list.empty()) {
      return VariableMatrix<Scalar>{detail::empty, wrt.rows(), 1};
    }

    // Append value to Hessian mapping
    auto push_edge = [this](size_t j, size_t k, ExpressionPtr<Scalar> value) {
      // Sort parent index before child index
      auto& h = m_top_list[std::min(j, k)]->hessian_expr;
      if (!h.empty() && h.back().first == std::max(j, k)) {
        h.back().second += std::move(value);
      } else {
        h.emplace_back(std::max(j, k), std::move(value));
      }
    };

    auto ptr_1 = constant_ptr(Scalar(1));
    auto ptr_2 = constant_ptr(Scalar(2));

    // Set root node's adjoint to 1 since df/df is 1
    m_top_list[0]->adjoint_expr = ptr_1;

    for (size_t i = 0; i < m_top_list_end; ++i) {
      const auto& v_i = m_top_list[i];
      const auto& v_l = v_i->args[0];
      const auto& v_r = v_i->args[1];

      // Compute node gradients
      ExpressionPtr<Scalar> g_l = v_i->grad_expr_l(v_l, v_r, ptr_1);
      ExpressionPtr<Scalar> g_r;
      if (v_r != nullptr) {
        g_r = v_i->grad_expr_r(v_l, v_r, ptr_1);
      }

      // Adjoints
      if (v_r != nullptr) {
        // Binary operator
        v_l->adjoint_expr += g_l * v_i->adjoint_expr;
        v_r->adjoint_expr += g_r * v_i->adjoint_expr;
      } else {
        // Unary operator
        v_l->adjoint_expr += g_l * v_i->adjoint_expr;
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
        if (v_r != nullptr) {
          // Binary operator
          size_t k_l = v_l->idx;
          size_t k_r = v_r->idx;

          if (i != j) {
            // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
            if (v_l->type() > CONSTANT) {
              push_edge(j, k_l, (j == k_l ? ptr_2 : ptr_1) * g_l * h_i_j);
            }
            if (v_r->type() > CONSTANT) {
              push_edge(j, k_r, (j == k_r ? ptr_2 : ptr_1) * g_r * h_i_j);
            }
          } else {
            // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
            if (v_l->type() > CONSTANT) {
              push_edge(k_l, k_l, g_l * g_l * h_i_j);
            }
            if (v_l->type() > CONSTANT && v_r->type() > CONSTANT) {
              push_edge(k_l, k_r,
                        (k_l == k_r ? ptr_2 : ptr_1) * g_l * g_r * h_i_j);
            }
            if (v_r->type() > CONSTANT) {
              push_edge(k_r, k_r, g_r * g_r * h_i_j);
            }
          }
        } else {
          // Unary operator
          size_t k_l = v_l->idx;

          if (i != j) {
            // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
            if (v_l->type() > CONSTANT) {
              push_edge(j, k_l, (j == k_l ? ptr_2 : ptr_1) * g_l * h_i_j);
            }
          } else {
            // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
            if (v_l->type() > CONSTANT) {
              push_edge(k_l, k_l, g_l * g_l * h_i_j);
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
      if (v_i->adjoint_expr && v_i->type() > LINEAR) {
        if (v_r != nullptr) {
          // Binary operator
          size_t k_l = v_l->idx;
          size_t k_r = v_r->idx;

          // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
          if (v_l->type() > CONSTANT) {
            if (auto h_ll = v_i->hess_expr_ll(v_l, v_r, v_i->adjoint_expr)) {
              push_edge(k_l, k_l, h_ll);
            }
          }

          // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
          if (v_l->type() > CONSTANT && v_r->type() > CONSTANT) {
            if (auto h_lr = v_i->hess_expr_lr(v_l, v_r, v_i->adjoint_expr)) {
              push_edge(k_l, k_r, (k_l == k_r ? ptr_2 : ptr_1) * h_lr);
            }
          }

          // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
          if (v_r->type() > CONSTANT) {
            if (auto h_rr = v_i->hess_expr_rr(v_l, v_r, v_i->adjoint_expr)) {
              push_edge(k_r, k_r, h_rr);
            }
          }
        } else {
          // Unary operator
          size_t k_l = v_l->idx;

          // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
          if (v_l->type() > CONSTANT) {
            if (auto h_ll = v_i->hess_expr_ll(v_l, v_r, v_i->adjoint_expr)) {
              push_edge(k_l, k_l, h_ll);
            }
          }
        }
      }
    }

    // Move Hessian tree to return value
    VariableMatrix<Scalar> H{detail::empty, wrt.rows(), wrt.rows()};
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
            H[row, col] += value;
          } else {
            H[col, row] += value;
          }
        } else {
          H[row, col] += value;
          if (row != col) {
            H[col, row] += value;
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

  /// Updates the adjoints in the expression graph (computes the Hessian) then
  /// appends the adjoints of wrt to the sparse matrix triplets.
  ///
  /// @tparam UpLo Which part of the Hessian to compute (Lower or Lower |
  ///     Upper).
  /// @param triplets The sparse matrix triplets.
  /// @param wrt Vector of variables with respect to which to compute the
  ///     Hessian.
  template <int UpLo>
    requires(UpLo == Eigen::Lower) || (UpLo == (Eigen::Lower | Eigen::Upper))
  void append_triplets(gch::small_vector<Eigen::Triplet<Scalar>>& triplets,
                       const VariableMatrix<Scalar>& wrt) const {
    using S = Scalar;
    using enum ExpressionType;

    slp_assert(wrt.cols() == 1);

    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    // Implements edge pushing as described by figure 4 on p. 406 of [1].
    //
    // [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
    //     efficient Hessian computation via automatic differentiation", 2016.
    //     https://sci-hub.st/10.1007/s12532-016-0100-3

    if (m_top_list.empty()) {
      return;
    }

#define DEBUG

#ifdef DEBUG
    gch::small_vector<SolveProfiler> profilers;
    profilers.emplace_back("∇²ₓₓL");
    profilers.emplace_back("  ↳ setup");
    profilers.emplace_back("  ↳ iteration");
    profilers.emplace_back("    ↳ adjoints");
    profilers.emplace_back("    ↳ pushing");
    profilers.emplace_back("    ↳ creating");
    profilers.emplace_back("  ↳ matrix build");

    auto& H_prof = profilers[0];
    auto& setup_prof = profilers[1];
    auto& iter_prof = profilers[2];
    auto& adjoints_prof = profilers[3];
    auto& pushing_prof = profilers[4];
    auto& creating_prof = profilers[5];
    auto& matrix_build_prof = profilers[6];

    H_prof.start();
    setup_prof.start();
#endif

    // Append value to Hessian mapping
    auto push_edge = [this](size_t j, size_t k, const Scalar& value) {
      // Sort parent index before child index
      auto& h = m_top_list[std::min(j, k)]->hessian;
      if (!h.empty() && h.back().first == std::max(j, k)) {
        h.back().second += value;
      } else {
        h.emplace_back(std::max(j, k), value);
      }
    };

    // Set root node's adjoint to 1 since df/df is 1
    m_top_list[0]->adjoint = S(1);

    // Zero the rest of the adjoints
    for (auto& node : m_top_list | std::views::drop(1)) {
      node->adjoint = S(0);
    }

    // Clear all Hessian mappings
    for (auto& elem : m_top_list) {
      elem->hessian.clear();
    }

#ifdef DEBUG
    setup_prof.stop();
#endif

    for (size_t i = 0; i < m_top_list_end; ++i) {
#ifdef DEBUG
      ScopedProfiler iter_profiler{iter_prof};
#endif

      const auto& v_i = m_top_list[i];
      const auto& v_l = v_i->args[0];
      const auto& v_r = v_i->args[1];

      // Compute node gradients
      S g_l;
      S g_r;
      if (v_r != nullptr) {
        g_l = v_i->grad_l(v_l->val, v_r->val, S(1));
        g_r = v_i->grad_r(v_l->val, v_r->val, S(1));
      } else {
        g_l = v_i->grad_l(v_l->val, S(0), S(1));
        g_r = S(0);
      }

#ifdef DEBUG
      ScopedProfiler adjoints_profiler{adjoints_prof};
#endif

      // Adjoints
      if (v_r != nullptr) {
        // Binary operator
        v_l->adjoint += g_l * v_i->adjoint;
        v_r->adjoint += g_r * v_i->adjoint;
      } else {
        // Unary operator
        v_l->adjoint += g_l * v_i->adjoint;
      }

#ifdef DEBUG
      adjoints_profiler.stop();
      ScopedProfiler pushing_profiler{pushing_prof};
#endif

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
        if (v_r != nullptr) {
          // Binary operator
          size_t k_l = v_l->idx;
          size_t k_r = v_r->idx;

          if (i != j) {
            // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
            if (v_l->type() > CONSTANT) {
              push_edge(j, k_l, S(j == k_l ? 2 : 1) * g_l * h_i_j);
            }
            if (v_r->type() > CONSTANT) {
              push_edge(j, k_r, S(j == k_r ? 2 : 1) * g_r * h_i_j);
            }
          } else {
            // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
            if (v_l->type() > CONSTANT) {
              push_edge(k_l, k_l, g_l * g_l * h_i_j);
            }
            if (v_l->type() > CONSTANT && v_r->type() > CONSTANT) {
              push_edge(k_l, k_r, S(k_l == k_r ? 2 : 1) * g_l * g_r * h_i_j);
            }
            if (v_r->type() > CONSTANT) {
              push_edge(k_r, k_r, g_r * g_r * h_i_j);
            }
          }
        } else {
          // Unary operator
          size_t k_l = v_l->idx;

          if (i != j) {
            // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
            if (v_l->type() > CONSTANT) {
              push_edge(j, k_l, S(j == k_l ? 2 : 1) * g_l * h_i_j);
            }
          } else {
            // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
            if (v_l->type() > CONSTANT) {
              push_edge(k_l, k_l, g_l * g_l * h_i_j);
            }
          }
        }
      }

#ifdef DEBUG
      pushing_profiler.stop();
      ScopedProfiler creating_profiler{creating_prof};
#endif

      // Creating
      //
      // if a(vᵢ) ≠ 0
      //   for all unordered pairs (vⱼ, vₖ) such that ∂²ϕᵢ/∂vⱼ∂vₖ ≠ 0
      //     if j = k
      //       h(vⱼ, vₖ) += 2 ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
      //     else
      //       h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
      if (v_i->adjoint != S(0) && v_i->type() > LINEAR) {
        if (v_r != nullptr) {
          // Binary operator
          size_t k_l = v_l->idx;
          size_t k_r = v_r->idx;

          // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
          if (v_l->type() > CONSTANT) {
            if (S h_ll = v_i->hess_ll(v_l->val, v_r->val, v_i->adjoint);
                h_ll != S(0)) {
              push_edge(k_l, k_l, h_ll);
            }
          }

          // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
          if (v_l->type() > CONSTANT && v_r->type() > CONSTANT) {
            if (S h_lr = v_i->hess_lr(v_l->val, v_r->val, v_i->adjoint);
                h_lr != S(0)) {
              push_edge(k_l, k_r, S(k_l == k_r ? 2 : 1) * h_lr);
            }
          }

          // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
          if (v_r->type() > CONSTANT) {
            if (S h_rr = v_i->hess_rr(v_l->val, v_r->val, v_i->adjoint);
                h_rr != S(0)) {
              push_edge(k_r, k_r, h_rr);
            }
          }
        } else {
          // Unary operator
          size_t k_l = v_l->idx;

          // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
          if (v_l->type() > CONSTANT) {
            if (S h_ll = v_i->hess_ll(v_l->val, S(0), v_i->adjoint);
                h_ll != S(0)) {
              push_edge(k_l, k_l, h_ll);
            }
          }
        }
      }
    }

#ifdef DEBUG
    matrix_build_prof.start();
#endif

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

#ifdef DEBUG
    matrix_build_prof.stop();
    H_prof.stop();

    auto H_duration = to_ms(profilers[0].total_duration());

    slp::println("┏{:━^23}┯{:━^18}┯{:━^10}┯{:━^9}┯{:━^5}┓", "", "", "", "", "");
    slp::println("┃{:^23}│{:^18}│{:^10}│{:^9}│{:^5}┃",
                 std::format("{} trace", profilers[0].name()), "percent",
                 "total (ms)", "each (ms)", "runs");
    slp::println("┡{:━^23}┷{:━^18}┷{:━^10}┷{:━^9}┷{:━^5}┩", "", "", "", "", "");

    for (auto& profiler : profilers) {
      double norm = H_duration == 0.0
                        ? (&profiler == &profilers[0] ? 1.0 : 0.0)
                        : to_ms(profiler.total_duration()) / H_duration;
      slp::println("│{:<23} {:>6.2f}%▕{}▏ {:>10.3f} {:>9.3f} {:>5}│",
                   profiler.name(), norm * 100.0, histogram<9>(norm),
                   to_ms(profiler.total_duration()),
                   to_ms(profiler.average_duration()), profiler.num_solves());
    }

    slp::println("└{:─^69}┘", "");
#endif
  }

 private:
  /// Topological sort of graph from parent to child
  gch::small_vector<Expression<Scalar>*> m_top_list;

  /// List that maps nodes to their respective column
  gch::small_vector<int> m_col_list;

  /// Index after dependent variables.
  size_t m_top_list_end = 0;
};

}  // namespace slp::detail
