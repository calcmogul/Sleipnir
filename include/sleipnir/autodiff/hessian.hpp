// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <optional>
#include <ranges>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/print_diagnostics.hpp"
#include "sleipnir/util/profiler.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/// This class calculates the Hessian of a variable with respect to a vector of
/// variables.
///
/// The gradient tree is cached so subsequent Hessian calculations are faster,
/// and the Hessian is only recomputed if the variable expression is nonlinear.
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

    m_top_list = detail::topological_sort(m_variable.expr);

    // Sort dependent variables before independent ones while maintaining
    // relative order (precondition of edge pushing)
    m_top_list_end = std::distance(
        m_top_list.begin(),
        std::stable_partition(m_top_list.begin(), m_top_list.end(),
                              [](const auto& elem) { return !elem->is_leaf; }));

    // TODO: Prune expression graph so checks for linear/quadratic later aren't
    // needed. Frontload repeated work in general.

    // Initialize column each expression's adjoint occupies in the Hessian
    for (size_t col = 0; col < m_wrt.size(); ++col) {
      m_wrt[col].expr->scratch = col;
    }

    for (auto& node : m_top_list) {
      m_col_list.emplace_back(node->scratch);
    }

    // Reset col to -1
    for (auto& node : m_wrt) {
      node.expr->scratch = -1;
    }

    if (m_variable.type() <= ExpressionType::QUADRATIC) {
      detail::update_values(m_top_list);

      gch::small_vector<Eigen::Triplet<Scalar>> triplets;
      append_triplets(triplets);

      m_H.setFromTriplets(triplets.begin(), triplets.end());
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
      detail::update_values(m_top_list);

      gch::small_vector<Eigen::Triplet<Scalar>> triplets;
      append_triplets(triplets);

      m_H.setFromTriplets(triplets.begin(), triplets.end());
    }

    return m_H;
  }

 private:
  Variable<Scalar> m_variable;
  VariableMatrix<Scalar> m_wrt;

  /// Topological sort of graph from parent to child
  gch::small_vector<detail::Expression<Scalar>*> m_top_list;

  /// Index after dependent variables
  size_t m_top_list_end = 0;

  /// List that maps nodes to their respective column
  gch::small_vector<int> m_col_list;

  Eigen::SparseMatrix<Scalar> m_H{m_wrt.rows(), m_wrt.rows()};

  /// Returns the variable's Hessian tree.
  ///
  /// This function lazily allocates variables, so elements of the returned
  /// VariableMatrix will be empty if the corresponding element of wrt had no
  /// adjoint. Ensure Variable::expr isn't nullptr before calling member
  /// functions.
  ///
  /// @param wrt Variables with respect to which to compute the Hessian.
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

    if (m_top_list.empty()) {
      return VariableMatrix<Scalar>{detail::empty, m_wrt.rows(), m_wrt.rows()};
    }

    // Append value to Hessian mapping
    auto push_edge = [this](size_t j, size_t k,
                            detail::ExpressionPtr<Scalar> value) {
      // Sort parent index before child index
      m_top_list[std::min(j, k)]->hessian_expr[std::max(j, k)] +=
          std::move(value);
    };

    auto ptr_1 = detail::constant_ptr(Scalar(1));
    auto ptr_2 = detail::constant_ptr(Scalar(2));

    // Set each node's index in m_top_list. We do this on every call because
    // other Hessian instances can overwrite it.
    for (size_t i = 0; i < m_top_list.size(); ++i) {
      const auto& node = m_top_list[i];
      node->idx = i;
    }

    // Set root node's adjoint to 1 since df/df is 1
    m_top_list[0]->adj_expr = ptr_1;

    for (size_t i = 0; i < m_top_list_end; ++i) {
      const auto& v_i = m_top_list[i];

      // Get node arguments
      detail::ArgArray<detail::Expression<Scalar>*> args;
      size_t num_args = 0;
      v_i->visit_args([&](auto* arg) { args[num_args++] = arg; });

      // Compute node gradients. Constant arguments are skipped since their
      // derivatives are never used.
      detail::ArgArray<detail::ExpressionPtr<Scalar>> g;
      v_i->grad_expr(g);
      for (size_t k = 0; k < num_args; ++k) {
        if (args[k]->type() == CONSTANT) {
          g[k] = nullptr;
        }
      }

      // Adjoints (a null adjoint is zero, so it has nothing to contribute)
      if (v_i->adj_expr != nullptr) {
        for (size_t k = 0; k < num_args; ++k) {
          if (g[k] != nullptr) {
            args[k]->adj_expr += g[k] * v_i->adj_expr;
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
      for (const auto& [j, h_i_j] : v_i->hessian_expr) {
        if (i != j) {
          // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
          for (size_t k = 0; k < num_args; ++k) {
            if (g[k] != nullptr) {
              size_t idx_k = args[k]->idx;
              push_edge(j, idx_k, (j == idx_k ? ptr_2 : ptr_1) * g[k] * h_i_j);
            }
          }
        } else {
          // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
          for (size_t k1 = 0; k1 < num_args; ++k1) {
            for (size_t k2 = 0; k2 <= k1; ++k2) {
              if (g[k1] != nullptr && g[k2] != nullptr) {
                size_t idx_k1 = args[k1]->idx;
                size_t idx_k2 = args[k2]->idx;
                push_edge(idx_k1, idx_k2,
                          (k1 != k2 && idx_k1 == idx_k2 ? ptr_2 : ptr_1) *
                              g[k1] * g[k2] * h_i_j);
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
      if (v_i->adj_expr && v_i->type() > LINEAR) {
        detail::ArgArray<detail::ArgArray<detail::ExpressionPtr<Scalar>>> H;
        v_i->hess_expr(H);

        // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
        //
        // Second derivatives are structurally zero if either first derivative
        // is.
        for (size_t k1 = 0; k1 < num_args; ++k1) {
          for (size_t k2 = 0; k2 <= k1; ++k2) {
            if (g[k1] != nullptr && g[k2] != nullptr && H[k1][k2] != nullptr) {
              size_t idx_k1 = args[k1]->idx;
              size_t idx_k2 = args[k2]->idx;
              push_edge(idx_k1, idx_k2,
                        (k1 != k2 && idx_k1 == idx_k2 ? ptr_2 : ptr_1) *
                            H[k1][k2] * v_i->adj_expr);
            }
          }
        }
      }
    }

    // Move Hessian tree to return value
    VariableMatrix<Scalar> H{detail::empty, m_wrt.rows(), m_wrt.rows()};
    for (int row = 0; row < static_cast<int>(m_wrt.rows()); ++row) {
      for (const auto& node : m_wrt[row].expr->hessian_expr) {
        const auto& col_idx = node.first;
        Variable value{node.second};

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
      node->adj_expr = nullptr;
      node->hessian_expr.clear();
    }

    return H;
  }

  /// Updates the adjoints in the expression graph (computes the Hessian) then
  /// appends the adjoints of wrt to the sparse matrix triplets.
  ///
  /// @param triplets The sparse matrix triplets.
  /// @param wrt Vector of variables with respect to which to compute the
  ///     Hessian.
  void append_triplets(
      gch::small_vector<Eigen::Triplet<Scalar>>& triplets) const {
    using S = Scalar;
    using enum ExpressionType;

    slp_assert(m_wrt.cols() == 1);

    // Read docs/algorithms.md#Reverse_accumulation_automatic_differentiation
    // for background on reverse accumulation automatic differentiation.

    // Implements edge pushing as described by figure 4 on p. 14 of [1].
    //
    // [1] Wang, M., et al. "Capitalizing on live variables: new algorithms for
    //     efficient Hessian computation via automatic differentiation", 2016.
    //     https://sci-hub.st/10.1007/s12532-016-0100-3

    if (m_top_list.empty()) {
      return;
    }

    // #define DEBUG

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
      m_top_list[std::min(j, k)]->hessian[std::max(j, k)] += value;
    };

    // Set each node's index in m_top_list. We do this on every call because
    // other Hessian instances can overwrite it.
    for (size_t i = 0; i < m_top_list.size(); ++i) {
      const auto& node = m_top_list[i];
      node->idx = i;
    }

    // Set root node's adjoint to 1 since df/df is 1
    m_top_list[0]->adj = S(1);

    // Zero the rest of the adjoints
    for (auto& node : m_top_list | std::views::drop(1)) {
      node->adj = S(0);
    }

    // Clear all Hessian mappings
    for (auto& node : m_top_list) {
      node->hessian.clear();
    }

#ifdef DEBUG
    setup_prof.stop();
#endif

    for (size_t i = 0; i < m_top_list_end; ++i) {
#ifdef DEBUG
      ScopedProfiler iter_profiler{iter_prof};
#endif

      const auto& v_i = m_top_list[i];

      // Get node arguments
      detail::ArgArray<detail::Expression<Scalar>*> args;
      size_t num_args = 0;
      v_i->visit_args([&](auto* arg) { args[num_args++] = arg; });

      // Compute node gradients. Constant arguments are skipped since their
      // derivatives are never used.
      detail::ArgArray<std::optional<S>> g;
      v_i->grad(g);
      for (size_t k = 0; k < num_args; ++k) {
        if (args[k]->type() == CONSTANT) {
          g[k].reset();
        }
      }

#ifdef DEBUG
      ScopedProfiler adjoints_profiler{adjoints_prof};
#endif

      // Adjoints
      for (size_t k = 0; k < num_args; ++k) {
        if (g[k]) {
          args[k]->adj += *g[k] * v_i->adj;
        }
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
        if (i != j) {
          // h(vⱼ, vₖ) += ∂ϕᵢ/∂vₖ h(vᵢ, vⱼ)
          for (size_t k = 0; k < num_args; ++k) {
            if (g[k]) {
              size_t idx_k = args[k]->idx;
              push_edge(j, idx_k, S(j == idx_k ? 2 : 1) * *g[k] * h_i_j);
            }
          }
        } else {
          // h(vₖ₁, vₖ₂) += ∂ϕᵢ/∂vₖ₁ ∂ϕᵢ/∂vₖ₂ h(vᵢ, vᵢ)
          for (size_t k1 = 0; k1 < num_args; ++k1) {
            for (size_t k2 = 0; k2 <= k1; ++k2) {
              if (g[k1] && g[k2]) {
                size_t idx_k1 = args[k1]->idx;
                size_t idx_k2 = args[k2]->idx;
                push_edge(idx_k1, idx_k2,
                          S(k1 != k2 && idx_k1 == idx_k2 ? 2 : 1) * *g[k1] *
                              *g[k2] * h_i_j);
              }
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
      if (v_i->type() > LINEAR) {
        detail::ArgArray<detail::ArgArray<std::optional<S>>> H;
        v_i->hess(H);

        // h(vⱼ, vₖ) += ∂²ϕᵢ/∂vⱼ∂vₖ a(vᵢ)
        //
        // Second derivatives are structurally zero if either first derivative
        // is.
        for (size_t k1 = 0; k1 < num_args; ++k1) {
          for (size_t k2 = 0; k2 <= k1; ++k2) {
            if (g[k1] && g[k2] && H[k1][k2]) {
              size_t idx_k1 = args[k1]->idx;
              size_t idx_k2 = args[k2]->idx;
              push_edge(idx_k1, idx_k2,
                        S(k1 != k2 && idx_k1 == idx_k2 ? 2 : 1) * *H[k1][k2] *
                            v_i->adj);
            }
          }
        }
      }
    }

#ifdef DEBUG
    matrix_build_prof.start();
#endif

    // Append Hessian triplets
    for (int row = 0; row < static_cast<int>(m_wrt.rows()); ++row) {
      for (const auto& [col_idx, value] : m_wrt[row].expr->hessian) {
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
};

// @cond Suppress Doxygen
extern template class EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
Hessian<double, Eigen::Lower | Eigen::Upper>;
// @endcond

}  // namespace slp
