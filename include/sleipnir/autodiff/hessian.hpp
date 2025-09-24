// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/adjoint_expression_graph.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/**
 * This class calculates the Hessian of a variable with respect to a vector of
 * variables.
 *
 * The gradient tree is cached so subsequent Hessian calculations are faster,
 * and the Hessian is only recomputed if the variable expression is nonlinear.
 *
 * @tparam UpLo Which part of the Hessian to compute (Lower or Lower | Upper).
 */
template <int UpLo>
  requires(UpLo == Eigen::Lower) || (UpLo == (Eigen::Lower | Eigen::Upper))
class SLEIPNIR_DLLEXPORT Hessian {
 public:
  /**
   * Constructs a Hessian object.
   *
   * @param variable Variable of which to compute the Hessian.
   * @param wrt Variable with respect to which to compute the Hessian.
   */
  Hessian(Variable variable, Variable wrt)
      : Hessian{std::move(variable), VariableMatrix{std::move(wrt)}} {}

  /**
   * Constructs a Hessian object.
   *
   * @param variable Variable of which to compute the Hessian.
   * @param wrt Vector of variables with respect to which to compute the
   *   Hessian.
   */
  Hessian(Variable variable, SleipnirMatrixLike auto wrt)
      : m_wrt{wrt}, m_graph{[&] {
          slp_assert(m_wrt.cols() == 1);

          // Initialize column each expression's adjoint occupies in the
          // Jacobian
          for (size_t col = 0; col < m_wrt.size(); ++col) {
            m_wrt[col].expr->col = col;
          }

          return detail::AdjointExpressionGraph{variable};
        }()} {
    // Reset col to -1
    for (auto& node : m_wrt) {
      node.expr->col = -1;
    }
  }

  /**
   * Returns the Hessian as a VariableMatrix.
   *
   * This is useful when constructing optimization problems with derivatives in
   * them.
   *
   * @return The Hessian as a VariableMatrix.
   */
  VariableMatrix get() const {
#if 1
    return VariableMatrix{m_wrt.rows(), m_wrt.rows()};
#else
    VariableMatrix result{VariableMatrix::empty, m_wrt.rows(), m_wrt.rows()};

    for (int row = 0; row < m_wrt.rows(); ++row) {
      auto grad = m_graphs[row].generate_gradient_tree(m_wrt);
      for (int col = 0; col < m_wrt.rows(); ++col) {
        if (grad[col].expr != nullptr) {
          result[row, col] = std::move(grad[col]);
        } else {
          result[row, col] = Variable{0.0};
        }
      }
    }

    return result;
#endif
  }

  /**
   * Evaluates the Hessian at wrt's value.
   *
   * @return The Hessian at wrt's value.
   */
  const Eigen::SparseMatrix<double>& value() {
    m_graph.update_values();

    gch::small_vector<Eigen::Triplet<double>> triplets;
    m_graph.append_hessian_triplets<UpLo>(triplets, m_wrt);

    m_H.setFromTriplets(triplets.begin(), triplets.end());

    return m_H;
  }

 private:
  VariableMatrix m_wrt;

  detail::AdjointExpressionGraph m_graph;

  Eigen::SparseMatrix<double> m_H{m_wrt.rows(), m_wrt.rows()};
};

}  // namespace slp
