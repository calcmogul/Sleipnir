// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <utility>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_op.hpp"
#include "sleipnir/autodiff/expression_type.hpp"

// Evaluates a non-leaf expression's value, partial derivatives, and second
// partial derivatives through its derived expression type's eval(),
// eval_grad(), and eval_hess(), as numbers, as expressions, or as which ones
// aren't structurally zero.

namespace slp::detail {

/// The number of second partial derivatives of derived expression type T that
/// can be nonzero, stored as the lower triangle of the Hessian with respect to
/// its arguments. Zero if T's second partial derivatives are all structurally
/// zero.
///
/// @tparam T Derived expression type.
template <typename T>
inline constexpr size_t NUM_SECOND_PARTIALS = requires {
  &T::template eval_hess<typename T::Scalar, typename T::Scalar>;
} ? T::NUM_ARGS * (T::NUM_ARGS + 1) / 2 : 0;

/// Whether derived expression type T's partial derivatives can be the same at
/// every argument value (e.g., for a + b, or for ab if b is constant).
///
/// T declares when they are with a `static constexpr bool
/// has_constant_partials(const std::array<bool, NUM_ARGS>& is_constant)` member
/// function,
/// which takes whether each argument is constant. Partial derivatives with
/// respect to constant arguments don't need to be constant, since they're never
/// propagated.
///
/// @tparam T Derived expression type.
template <typename T>
inline constexpr bool HAS_CONSTANT_PARTIALS =
    requires { T::has_constant_partials(std::array<bool, T::NUM_ARGS>{}); };

/// Whether derived expression type T's second partial derivatives are the same
/// at every argument value (e.g., ∂²(ab)/∂a∂b = 1).
///
/// T declares this with a `static constexpr bool CONSTANT_SECOND_PARTIALS`
/// member that's true.
///
/// @tparam T Derived expression type.
template <typename T>
inline constexpr bool HAS_CONSTANT_SECOND_PARTIALS =
    requires { requires T::CONSTANT_SECOND_PARTIALS; };

/// The number of second partial derivatives of derived expression type T that
/// can be nonzero and vary with the argument values. Constant ones don't need
/// to be evaluated, since they can be folded into whatever they multiply.
///
/// @tparam T Derived expression type.
template <typename T>
inline constexpr size_t NUM_VARYING_SECOND_PARTIALS =
    HAS_CONSTANT_SECOND_PARTIALS<T> ? 0 : NUM_SECOND_PARTIALS<T>;

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

/// op_hess() with the argument indices as a template parameter pack.
///
/// See op_grad() for why this isn't a generic lambda.
template <typename T, typename Scalar, typename Value, typename Partial,
          size_t... Is>
void op_hess([[maybe_unused]] const Expression<Scalar>& expr,
             [[maybe_unused]] const std::array<Value, T::NUM_ARGS>& x,
             [[maybe_unused]] const Value& val,
             [[maybe_unused]] std::array<std::array<Partial, T::NUM_ARGS>,
                                         T::NUM_ARGS>& H,
             std::index_sequence<Is...>) {
  if constexpr (requires { T::eval_hess(H, val, x[Is]...); }) {
    T::eval_hess(H, val, x[Is]...);
  } else if constexpr (requires {
                         static_cast<const T&>(expr).eval_hess(H, val,
                                                               x[Is]...);
                       }) {
    static_cast<const T&>(expr).eval_hess(H, val, x[Is]...);
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

/// Returns ∂²/∂aⱼ∂aₖ of a derived expression type for each pair of arguments
/// aⱼ and aₖ at the given argument values.
///
/// T computes them with `eval_hess<Partial, Value>(H, val, a₀, a₁, ...)`, which
/// takes this function's H and val followed by the argument values. It's static
/// unless the second partial derivatives depend on the expression's state. If T
/// doesn't define it, all second partial derivatives are structurally zero.
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
/// @param H The second partial derivatives in visit_args() order. Only the
///     lower triangle (j ≥ k) is written. Elements left unwritten are
///     structurally zero.
template <typename T, typename Scalar, typename Value, typename Partial>
void op_hess([[maybe_unused]] const Expression<Scalar>& expr,
             const std::array<Value, T::NUM_ARGS>& x, const Value& val,
             std::array<std::array<Partial, T::NUM_ARGS>, T::NUM_ARGS>& H) {
  op_hess<T>(expr, x, val, H, std::make_index_sequence<T::NUM_ARGS>{});
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

/// Returns which second partial derivatives ∂²/∂aⱼ∂aₖ of an expression aren't
/// structurally zero.
///
/// Whether a second partial derivative is structurally zero only depends on
/// the expression's operator, not its arguments's values.
///
/// @tparam Scalar Scalar type.
/// @param expr The expression. Must not be a leaf.
/// @return Whether each second partial derivative isn't structurally zero, in
///     visit_args() order. Only the lower triangle (j ≥ k) is set.
template <typename Scalar>
std::array<std::array<bool, MAX_ARGS<Scalar>>, MAX_ARGS<Scalar>> hess_structure(
    const Expression<Scalar>& expr) {
  std::array<std::array<bool, MAX_ARGS<Scalar>>, MAX_ARGS<Scalar>> nonzero{};
  visit_op<Scalar>(expr.op(), [&]<typename T>() {
    std::array<std::array<std::optional<Scalar>, T::NUM_ARGS>, T::NUM_ARGS> H;
    op_hess<T>(expr, std::array<Scalar, T::NUM_ARGS>{}, Scalar(0), H);
    for (size_t j = 0; j < T::NUM_ARGS; ++j) {
      for (size_t k = 0; k < T::NUM_ARGS; ++k) {
        nonzero[j][k] = H[j][k].has_value();
      }
    }
  });
  return nonzero;
}

/// Returns the partial derivatives of an expression with respect to its
/// arguments if they're the same at every argument value, or std::nullopt if
/// they aren't (see HAS_CONSTANT_PARTIALS).
///
/// @tparam Scalar Scalar type.
/// @param expr The expression. Must not be a leaf.
/// @return The partial derivatives in visit_args() order, or std::nullopt.
template <typename Scalar>
std::optional<std::array<Scalar, MAX_ARGS<Scalar>>> constant_partials(
    const Expression<Scalar>& expr) {
  return visit_op<Scalar>(
      expr.op(),
      [&]<typename T>() -> std::optional<std::array<Scalar, MAX_ARGS<Scalar>>> {
        if constexpr (HAS_CONSTANT_PARTIALS<T>) {
          const auto& args = static_cast<const T&>(expr).args;
          std::array<Scalar, T::NUM_ARGS> x;
          std::array<bool, T::NUM_ARGS> is_constant;
          for (size_t k = 0; k < T::NUM_ARGS; ++k) {
            x[k] = args[k]->val;
            is_constant[k] = args[k]->type() == ExpressionType::CONSTANT;
          }
          if (!T::has_constant_partials(is_constant)) {
            return std::nullopt;
          }

          // The partial derivatives only depend on the values of constant
          // arguments, not on the expression's value
          std::array<Scalar, T::NUM_ARGS> g{};
          op_grad<T>(expr, x, Scalar(0), g);
          std::array<Scalar, MAX_ARGS<Scalar>> partials{};
          std::ranges::copy(g, partials.begin());
          return partials;
        } else {
          return std::nullopt;
        }
      });
}

/// Returns the second partial derivatives of an expression with respect to
/// its arguments if they're the same at every argument value, or std::nullopt
/// if they aren't (see HAS_CONSTANT_SECOND_PARTIALS).
///
/// @tparam Scalar Scalar type.
/// @param expr The expression. Must not be a leaf.
/// @return The second partial derivatives in visit_args() order (lower
///     triangle only), or std::nullopt.
template <typename Scalar>
auto constant_second_partials(const Expression<Scalar>& expr) {
  using SecondPartials =
      std::array<std::array<Scalar, MAX_ARGS<Scalar>>, MAX_ARGS<Scalar>>;

  return visit_op<Scalar>(
      expr.op(), [&]<typename T>() -> std::optional<SecondPartials> {
        if constexpr (HAS_CONSTANT_SECOND_PARTIALS<T>) {
          // The second partial derivatives don't depend on the arguments's
          // values or the expression's value
          std::array<std::array<Scalar, T::NUM_ARGS>, T::NUM_ARGS> op_H{};
          op_hess<T>(expr, std::array<Scalar, T::NUM_ARGS>{}, Scalar(0), op_H);
          SecondPartials H{};
          for (size_t j = 0; j < T::NUM_ARGS; ++j) {
            std::ranges::copy(op_H[j], H[j].begin());
          }
          return H;
        } else {
          return std::nullopt;
        }
      });
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

/// Returns ∂²/∂aⱼ∂aₖ of an expression as an Expression for each pair of
/// arguments aⱼ and aₖ.
///
/// The partial derivatives aren't scaled by the adjoint.
///
/// @tparam Scalar Scalar type.
/// @param expr The expression. Must not be a leaf.
/// @param H The second partial derivatives in visit_args() order. Only the
///     lower triangle (j ≥ k) is written. Elements left as nullptr are
///     structurally zero.
template <typename Scalar>
void hess_expr(Expression<Scalar>* expr,
               std::array<std::array<ExpressionPtr<Scalar>, MAX_ARGS<Scalar>>,
                          MAX_ARGS<Scalar>>& H) {
  visit_op<Scalar>(expr->op(), [&]<typename T>() {
    std::array<std::array<ExpressionPtr<Scalar>, T::NUM_ARGS>, T::NUM_ARGS>
        op_H;
    op_hess<T>(*expr, static_cast<const T&>(*expr).args,
               ExpressionPtr<Scalar>{expr}, op_H);
    for (size_t j = 0; j < T::NUM_ARGS; ++j) {
      std::ranges::move(op_H[j], H[j].begin());
    }
  });
}

}  // namespace slp::detail
