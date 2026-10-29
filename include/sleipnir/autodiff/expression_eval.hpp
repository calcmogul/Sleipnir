// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <utility>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_op.hpp"

// Evaluates a non-leaf expression's value and partial derivatives through its
// derived expression type's eval() and eval_grad(), as numbers, as
// expressions, or as which ones aren't structurally zero.

namespace slp::detail {

/// Returns a derived expression type's value at the given argument values.
///
/// Most derived expression types compute their value with a static function.
/// The expression is only used by ones whose value depends on its state.
///
/// @tparam T Derived expression type.
/// @tparam Scalar Scalar type.
/// @param expr The expression.
/// @param x The argument values in visit_args() order.
/// @return The expression's value.
template <typename T, typename Scalar>
Scalar op_value([[maybe_unused]] const Expression<Scalar>& expr,
                const std::array<Scalar, T::NUM_ARGS>& x) {
  return [&]<size_t... Is>(std::index_sequence<Is...>) {
    if constexpr (requires { T::eval(x[Is]...); }) {
      return T::eval(x[Is]...);
    } else {
      return static_cast<const T&>(expr).eval(x[Is]...);
    }
  }(std::make_index_sequence<T::NUM_ARGS>{});
}

/// op_grad() with the argument indices as a template parameter pack.
///
/// This is a function template instead of a generic lambda because MSVC treats
/// an invalid expression in a generic lambda's requires-expression as a hard
/// error instead of false.
template <typename T, typename Scalar, typename Value, typename Partial,
          size_t... Is>
void op_grad([[maybe_unused]] const Expression<Scalar>& expr,
             [[maybe_unused]] const std::array<Value, T::NUM_ARGS>& x,
             [[maybe_unused]] const Value& val,
             [[maybe_unused]] std::array<Partial, T::NUM_ARGS>& g,
             std::index_sequence<Is...>) {
  if constexpr (requires { T::eval_grad(g, val, x[Is]...); }) {
    T::eval_grad(g, val, x[Is]...);
  } else if constexpr (requires {
                         static_cast<const T&>(expr).eval_grad(g, val,
                                                               x[Is]...);
                       }) {
    static_cast<const T&>(expr).eval_grad(g, val, x[Is]...);
  }
}

/// Returns ∂/∂aₖ of a derived expression type for each argument aₖ at the
/// given argument values.
///
/// T computes them with `eval_grad<Partial, Value>(g, val, a₀, a₁, ...)`, which
/// takes this function's g and val followed by the argument values. It's static
/// unless the partial derivatives depend on the expression's state. If T
/// doesn't define it, all partial derivatives are structurally zero.
///
/// @tparam T Derived expression type.
/// @tparam Scalar Scalar type.
/// @tparam Value Scalar to evaluate the partial derivatives, or
///     ExpressionPtr<Scalar> to build them as expressions.
/// @tparam Partial The type each partial derivative is written as. Value, or
///     std::optional<Scalar> to find which ones aren't structurally zero
///     (elements left unwritten stay std::nullopt).
/// @param expr The expression.
/// @param x The arguments in visit_args() order.
/// @param val The expression.
/// @param g The partial derivatives in visit_args() order. Elements left
///     unwritten are structurally zero.
template <typename T, typename Scalar, typename Value, typename Partial>
void op_grad([[maybe_unused]] const Expression<Scalar>& expr,
             const std::array<Value, T::NUM_ARGS>& x, const Value& val,
             std::array<Partial, T::NUM_ARGS>& g) {
  op_grad<T>(expr, x, val, g, std::make_index_sequence<T::NUM_ARGS>{});
}

/// Returns which partial derivatives ∂/∂aₖ of an expression aren't
/// structurally zero.
///
/// Whether a partial derivative is structurally zero only depends on the
/// expression's operator, not its arguments's values.
///
/// @tparam Scalar Scalar type.
/// @param expr The expression. Must not be a leaf.
/// @return Whether each partial derivative isn't structurally zero, in
///     visit_args() order.
template <typename Scalar>
std::array<bool, MAX_ARGS<Scalar>> grad_structure(
    const Expression<Scalar>& expr) {
  std::array<bool, MAX_ARGS<Scalar>> nonzero{};
  visit_op<Scalar>(expr.op(), [&]<typename T>() {
    std::array<std::optional<Scalar>, T::NUM_ARGS> g;
    op_grad<T>(expr, std::array<Scalar, T::NUM_ARGS>{}, Scalar(0), g);
    for (size_t k = 0; k < T::NUM_ARGS; ++k) {
      nonzero[k] = g[k].has_value();
    }
  });
  return nonzero;
}

/// Returns ∂/∂aₖ of an expression as an Expression for each argument aₖ.
///
/// The partial derivatives aren't scaled by the adjoint.
///
/// @tparam Scalar Scalar type.
/// @param expr The expression. Must not be a leaf.
/// @param g The partial derivatives in visit_args() order. Elements left as
///     nullptr are structurally zero.
template <typename Scalar>
void grad_expr(Expression<Scalar>* expr,
               std::array<ExpressionPtr<Scalar>, MAX_ARGS<Scalar>>& g) {
  visit_op<Scalar>(expr->op(), [&]<typename T>() {
    std::array<ExpressionPtr<Scalar>, T::NUM_ARGS> op_g;
    op_grad<T>(*expr, static_cast<const T&>(*expr).args,
               ExpressionPtr<Scalar>{expr}, op_g);
    std::ranges::move(op_g, g.begin());
  });
}

}  // namespace slp::detail
