// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/adjoint_tape.hpp"
#include "sleipnir/autodiff/edge_pushing_program.hpp"
#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/empty.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/// This class calculates the Hessian of a variable with respect to a vector of
/// variables.
///
/// The Hessian is computed with edge pushing. The pushes are recorded once on
/// construction, then replayed with new values on each call to value() (see
/// detail::EdgePushingProgram).
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

    slp_assert(m_tape.nodes().empty() ||
               m_tape.top_list()[m_tape.nodes()[0].idx] ==
                   m_variable.expr.get());
    m_program = detail::EdgePushingProgram<Scalar, UpLo>{m_tape, m_wrt.rows()};

    if (m_variable.type() <= ExpressionType::QUADRATIC) {
      m_tape.template update_values<2>();
      m_program.compute(m_tape);
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

    // The program's nonzeros are laid out like value()'s
    auto values = m_program.symbolic_values(m_tape);
    const auto& pattern = m_program.value();
    slp_assert(pattern.isCompressed());
    for (int col = 0; col < pattern.outerSize(); ++col) {
      for (int idx = pattern.outerIndexPtr()[col];
           idx < pattern.outerIndexPtr()[col + 1]; ++idx) {
        result[pattern.innerIndexPtr()[idx], col] =
            Variable{std::move(values[idx])};
      }
    }

    // The program only has entries that aren't structurally zero
    for (int row = 0; row < m_wrt.rows(); ++row) {
      const int num_cols = UpLo == Eigen::Lower ? row + 1 : m_wrt.rows();
      for (int col = 0; col < num_cols; ++col) {
        if (result[row, col].expr == nullptr) {
          result[row, col] = Variable{Scalar(0)};
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
      m_tape.template update_values<2>();
      m_program.compute(m_tape);
    }

    return m_program.value();
  }

 private:
  Variable<Scalar> m_variable;
  VariableMatrix<Scalar> m_wrt;

  detail::AdjointTape<Scalar> m_tape;
  detail::EdgePushingProgram<Scalar, UpLo> m_program;
};

// @cond Suppress Doxygen
extern template class EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
Hessian<double, Eigen::Lower | Eigen::Upper>;
// @endcond

}  // namespace slp
