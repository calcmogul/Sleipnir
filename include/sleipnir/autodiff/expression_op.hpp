// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_functions.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/util/force_inline.hpp"

namespace slp::detail {
/// The derived expression types of the non-leaf operators, in ExpressionOp
/// order.
///
/// Derived expression types templated on ExpressionType are listed as their
/// nonlinear instantiation. Their eval functions don't depend on the
/// ExpressionType.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
using NonLeafExpressionTypes = std::tuple<
    AbsExpression<Scalar>, AcosExpression<Scalar>, AsinExpression<Scalar>,
    AtanExpression<Scalar>, Atan2Expression<Scalar>,
    BinaryMinusExpression<Scalar, ExpressionType::NONLINEAR>,
    BinaryPlusExpression<Scalar, ExpressionType::NONLINEAR>,
    CbrtExpression<Scalar>, CosExpression<Scalar>, CoshExpression<Scalar>,
    DivExpression<Scalar, ExpressionType::NONLINEAR>, ErfExpression<Scalar>,
    ExpExpression<Scalar>, Hypot2Expression<Scalar>, Hypot3Expression<Scalar>,
    IfElseExpression<Scalar>, LogExpression<Scalar>, Log10Expression<Scalar>,
    MaxExpression<Scalar>, MinExpression<Scalar>,
    MultExpression<Scalar, ExpressionType::NONLINEAR>,
    PowExpression<Scalar, ExpressionType::NONLINEAR>, SignExpression<Scalar>,
    SinExpression<Scalar>, SinhExpression<Scalar>, SqrtExpression<Scalar>,
    TanExpression<Scalar>, TanhExpression<Scalar>,
    UnaryMinusExpression<Scalar, ExpressionType::NONLINEAR>>;

/// Calls f.template operator()<T>() with the derived expression type T of a
/// non-leaf operator in [Begin, End).
///
/// @tparam Scalar Scalar type.
/// @tparam Begin First operator in the range.
/// @tparam End One past the last operator in the range.
/// @tparam F Function type.
/// @param op The operator.
/// @param f The function to call.
/// @return f's return value.
template <typename Scalar, size_t Begin, size_t End, typename F>
SLEIPNIR_FORCE_INLINE decltype(auto) visit_op_in(size_t op, F& f) {
  if constexpr (End - Begin == 1) {
    using T = std::tuple_element_t<Begin, NonLeafExpressionTypes<Scalar>>;
    static_assert(T::OP == static_cast<ExpressionOp>(Begin),
                  "NonLeafExpressionTypes must be in ExpressionOp order");
    return f.template operator()<T>();
  } else {
    constexpr size_t mid = Begin + (End - Begin) / 2;
    if (op < mid) {
      return visit_op_in<Scalar, Begin, mid>(op, f);
    } else {
      return visit_op_in<Scalar, mid, End>(op, f);
    }
  }
}

/// Calls f.template operator()<T>() with the derived expression type T of a
/// non-leaf operator.
///
/// The operator is found with a binary search of conditional branches instead
/// of a switch statement. Tapes often repeat a short sequence of operators
/// (e.g., one sequence per term of a sum), which conditional branch predictors
/// learn, but the single indirect branch of a switch statement's jump table
/// often doesn't. On the extended Rosenbrock function, a jump table
/// mispredicted about one in two operators and more than doubled tape
/// evaluation time.
///
/// @tparam Scalar Scalar type.
/// @tparam F Function type.
/// @param op The operator. Must not be a leaf operator.
/// @param f The function to call.
/// @return f's return value.
template <typename Scalar, typename F>
SLEIPNIR_FORCE_INLINE decltype(auto) visit_op(ExpressionOp op, F&& f) {
  return visit_op_in<Scalar, 0,
                     std::tuple_size_v<NonLeafExpressionTypes<Scalar>>>(
      static_cast<size_t>(op), f);
}

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
                const ArgArray<Scalar>& x) {
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
             [[maybe_unused]] const ArgArray<Value>& x,
             [[maybe_unused]] const Value& val,
             [[maybe_unused]] ArgArray<Partial>& g,
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
             const ArgArray<Value>& x, const Value& val, ArgArray<Partial>& g) {
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
ArgArray<bool> grad_structure(const Expression<Scalar>& expr) {
  ArgArray<std::optional<Scalar>> g;
  visit_op<Scalar>(expr.op(), [&]<typename T>() {
    op_grad<T>(expr, ArgArray<Scalar>{}, Scalar(0), g);
  });

  ArgArray<bool> nonzero{};
  for (size_t k = 0; k < MAX_ARGS; ++k) {
    nonzero[k] = g[k].has_value();
  }
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
void grad_expr(Expression<Scalar>* expr, ArgArray<ExpressionPtr<Scalar>>& g) {
  visit_op<Scalar>(expr->op(), [&]<typename T>() {
    ArgArray<ExpressionPtr<Scalar>> x;
    std::ranges::copy(static_cast<const T&>(*expr).args, x.begin());
    op_grad<T>(*expr, x, ExpressionPtr<Scalar>{expr}, g);
  });
}

}  // namespace slp::detail
