// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <optional>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_eval.hpp"
#include "sleipnir/autodiff/expression_op.hpp"

// Vertex elimination on the linearized graph.
//
// The linearized graph of an expression graph has the same nodes, and an edge
// from each node vᵢ to each of its arguments vₖ weighted by the partial
// derivative ∂ϕᵢ/∂vₖ. Reverse accumulation and edge pushing both propagate
// along its edges: a node's adjoint operations and Hessian pushes are
// per-edge, so their cost grows with the number of edges.
//
// Eliminating a node v removes it from the linearized graph and replaces each
// pair of an in-edge u → v and an out-edge v → w with a composed edge u → w
// whose weight is the product of the two weights. The reduced graph has the
// same derivatives with respect to the nodes left in it. A node with p
// in-edges and c out-edges replaces p + c edges with p c.
//
// Only nodes whose out-edge weights are constant are eliminated here, so a
// composed edge's weight is the in-edge's partial derivative times a constant,
// its scale (see Edge). A composed edge goes through the same argument of u as
// the in-edge it replaces, and several can reach the same child.
//
// An elimination policy (see EliminationPolicy) says which nodes are eligible
// and when eliminating one is worthwhile. Each consumer adds which nodes it
// has to keep (e.g., elements of wrt) and counts p and c in the order it
// visits the nodes.

namespace slp::detail {

/// A node reached from an eliminated node by following out-edges of
/// eliminated nodes.
///
/// @tparam Scale The scale's type (see EliminationPolicy).
template <typename Scale>
struct ScaledChild {
  /// The node's index in the topological list.
  uint32_t k;

  /// The product of the weights of the out-edges followed.
  Scale scale;
};

/// An edge of the reduced linearized graph from a node vᵢ to a child vₖ.
///
/// Its weight is scale ∂ϕᵢ/∂a, where a is the argument of vᵢ's expression the
/// edge goes through. If no node was eliminated between them, the child is
/// that argument and the scale is 1.
///
/// @tparam Scale The scale's type (see EliminationPolicy).
template <typename Scale>
struct Edge {
  /// The child's index in the topological list.
  uint32_t k;

  /// The index of the argument of the node's expression the edge goes through.
  uint32_t arg;

  /// The product of the weights of the eliminated nodes's out-edges between
  /// the argument and the child (1 if the child is the argument).
  Scale scale;
};

/// A policy for which nodes vertex elimination eliminates.
///
/// * `Policy::Scalar` is the expression graph's scalar type.
/// * `Policy::Scale` is the type of an edge's scale (see Edge).
/// * `Policy::weights(expr)` returns a node's out-edge weights in visit_args()
///   order if the node is eligible for elimination, or std::nullopt if it
///   isn't. A node is only eligible if its partial derivatives are the same at
///   every argument value.
/// * `Policy::accepts(p, c)` returns whether eliminating an eligible node with
///   p in-edges and c out-edges is worthwhile.
///
/// @tparam Policy The policy.
template <typename Policy>
concept EliminationPolicy =
    requires(const Expression<typename Policy::Scalar>& expr, uint32_t p,
             uint32_t c, size_t arg) {
      {
        (*Policy::weights(expr))[arg]
      } -> std::convertible_to<typename Policy::Scale>;
      { Policy::accepts(p, c) } -> std::same_as<bool>;
    };

/// Eliminates nodes whose out-edge weights are all 1 or −1 (e.g., a + b,
/// a − b, and −a) if that doesn't add edges.
///
/// A composed edge's scale is then 1 or −1, so its weight is the in-edge's
/// partial derivative or its negation, and propagating along it needs no
/// multiplication by a constant. AdjointTape uses this, since its adjoint
/// operations only multiply by a partial derivative.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
struct UnitEliminationPolicy {
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// The sign of a composed edge's weight: 1 or −1.
  using Scale = int;

  /// Returns the signs of a node's partial derivatives if they're 1 or −1 at
  /// every argument value.
  ///
  /// These are operators whose partial derivatives are constant even if no
  /// argument is (see HAS_CONSTANT_PARTIALS), so they're found from the
  /// operator alone, without reading the arguments.
  ///
  /// @param expr The node's expression. Must not be a leaf.
  /// @return The signs in visit_args() order, or std::nullopt if the partial
  ///     derivatives aren't all 1 or −1.
  static std::optional<std::array<int, MAX_ARGS<Scalar>>> weights(
      const Expression<Scalar>& expr) {
    return visit_op<Scalar>(
        expr.op(),
        [&]<typename T>() -> std::optional<std::array<int, MAX_ARGS<Scalar>>> {
          if constexpr (HAS_CONSTANT_PARTIALS<T>) {
            if constexpr (T::has_constant_partials(
                              std::array<bool, T::NUM_ARGS>{})) {
              std::array<Scalar, T::NUM_ARGS> g{};
              op_grad<T>(expr, std::array<Scalar, T::NUM_ARGS>{}, Scalar(0), g);

              std::array<int, MAX_ARGS<Scalar>> signs{};
              for (size_t k = 0; k < T::NUM_ARGS; ++k) {
                if (g[k] == Scalar(1)) {
                  signs[k] = 1;
                } else if (g[k] == Scalar(-1)) {
                  signs[k] = -1;
                } else {
                  return std::nullopt;
                }
              }
              return signs;
            }
          }
          return std::nullopt;
        });
  }

  /// Returns whether eliminating a node leaves at most as many edges:
  /// p c ≤ p + c.
  ///
  /// @param p The number of the node's in-edges.
  /// @param c The number of the node's out-edges.
  static constexpr bool accepts(uint32_t p, uint32_t c) {
    return p * c <= p + c;
  }
};

/// Eliminates nodes whose out-edge weights are constant (e.g., a + b, a − b,
/// −a, and 2a) if they have at most one in-edge or at most one out-edge.
///
/// A composed edge's scale can be any constant, so propagating along it
/// multiplies by one. EdgePushingProgram uses this through LinearElimination,
/// since its operations already have a coefficient.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
struct ConstantEliminationPolicy {
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// A composed edge's scale.
  using Scale = Scalar;

  /// Returns a node's partial derivatives if they're the same at every
  /// argument value (see constant_partials()).
  ///
  /// @param expr The node's expression. Must not be a leaf.
  /// @return The partial derivatives in visit_args() order, or std::nullopt
  ///     if they aren't constant.
  static std::optional<std::array<Scalar, MAX_ARGS<Scalar>>> weights(
      const Expression<Scalar>& expr) {
    return constant_partials(expr);
  }

  /// Returns whether a node has at most one in-edge or at most one out-edge:
  /// p ≤ 1 or c ≤ 1.
  ///
  /// This is UnitEliminationPolicy's rule without p = c = 2, which replaces
  /// four edges with four.
  ///
  /// @param p The number of the node's in-edges.
  /// @param c The number of the node's out-edges.
  static constexpr bool accepts(uint32_t p, uint32_t c) {
    return p <= 1 || c <= 1;
  }
};

static_assert(EliminationPolicy<UnitEliminationPolicy<double>>);
static_assert(EliminationPolicy<ConstantEliminationPolicy<double>>);

}  // namespace slp::detail
