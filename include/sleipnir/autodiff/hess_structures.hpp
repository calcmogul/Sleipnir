// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <cstddef>
#include <span>

#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/adjoint_tape.hpp"
#include "sleipnir/autodiff/expression_eval.hpp"
#include "sleipnir/autodiff/expression_type.hpp"

namespace slp::detail {

/// Which second partial derivatives of each of an AdjointTape's nodes aren't
/// structurally zero.
///
/// They're stored flat, so a node with n arguments takes n(n + 1)/2 bytes
/// instead of MAX_ARGS² for the operator with the most arguments.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class HessStructures {
 public:
  /// Constructs an empty HessStructures.
  HessStructures() = default;

  /// Finds the second partial derivative structure of each node in a tape's
  /// nodes(). They're all zero for linear nodes, which get none.
  ///
  /// @param tape The tape.
  explicit HessStructures(const AdjointTape<Scalar>& tape) {
    const auto nodes = tape.nodes();
    m_first.reserve(nodes.size() + 1);
    m_first.push_back(0);
    for (const auto& node : nodes) {
      if (node.expr->type() > ExpressionType::LINEAR) {
        const auto H = hess_structure(*node.expr);
        for (size_t j = 0; j < node.num_args; ++j) {
          for (size_t k = 0; k <= j; ++k) {
            m_nonzero.push_back(H[j][k]);
          }
        }
      }
      m_first.push_back(static_cast<uint32_t>(m_nonzero.size()));
    }
  }

  /// Returns whether each second partial derivative ∂²ϕᵢ/∂aⱼ∂aₖ with j ≥ k of
  /// a node isn't structurally zero, at index j(j + 1)/2 + k, or an empty span
  /// if the node is linear.
  ///
  /// @param pos The node's position in the tape's nodes().
  /// @return The node's second partial derivative structure.
  std::span<const uint8_t> operator[](size_t pos) const {
    return {m_nonzero.data() + m_first[pos], m_first[pos + 1] - m_first[pos]};
  }

 private:
  /// Each node's first element in m_nonzero, indexed by position in nodes(),
  /// followed by m_nonzero's size.
  gch::small_vector<uint32_t> m_first;

  /// Each node's lower triangle of second partial derivatives (row-major),
  /// one if it isn't structurally zero and zero if it is.
  gch::small_vector<uint8_t> m_nonzero;
};

}  // namespace slp::detail
