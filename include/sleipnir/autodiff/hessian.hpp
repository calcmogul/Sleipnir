// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/hessian_expression_graph.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"

namespace slp {

/// This class calculates the Hessian of a variable with respect to a vector of
/// variables.
///
/// The gradient tree is cached so subsequent Hessian calculations are faster,
/// and the Hessian is only recomputed if the variable expression is nonlinear.
///
/// @tparam Scalar Scalar type.
/// @tparam UpLo Which part of the Hessian to compute (Lower or Lower | Upper).
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
      : m_variable{std::move(variable)}, m_wrt{std::move(wrt)}, m_graph{[&] {
          slp_assert(m_wrt.cols() == 1);

          // Initialize column each expression's adjoint occupies in the
          // Hessian
          for (size_t col = 0; col < m_wrt.size(); ++col) {
            m_wrt[col].expr->col = col;
          }

          return detail::HessianExpressionGraph<Scalar>{m_variable};
        }()} {
    // Reset col to -1
    for (auto& node : m_wrt) {
      node.expr->col = -1;
    }

    if (m_variable.type() <= ExpressionType::QUADRATIC) {
      m_graph.update_values();

      gch::small_vector<Eigen::Triplet<Scalar>> triplets;
      m_graph.template append_triplets<UpLo>(triplets, m_wrt);

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

    auto H = m_graph.template generate_tree<UpLo>(m_wrt);

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
      m_graph.update_values();

      gch::small_vector<Eigen::Triplet<Scalar>> triplets;
      m_graph.template append_triplets<UpLo>(triplets, m_wrt);

      m_H.setFromTriplets(triplets.begin(), triplets.end());
    }

    return m_H;
  }

 private:
  Variable<Scalar> m_variable;
  VariableMatrix<Scalar> m_wrt;

  detail::HessianExpressionGraph<Scalar> m_graph;

  Eigen::SparseMatrix<Scalar> m_H{m_wrt.rows(), m_wrt.rows()};
};

// @cond Suppress Doxygen
extern template class EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
Hessian<double, Eigen::Lower | Eigen::Upper>;
// @endcond

}  // namespace slp
