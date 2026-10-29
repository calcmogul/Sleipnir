// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_functions.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/util/force_inline.hpp"

namespace slp::detail {
/// A list of derived expression type templates.
///
/// @tparam Ts Derived expression type templates.
template <template <typename, ExpressionType> typename... Ts>
struct ExpressionTemplates {
  /// The number of templates in the list.
  static constexpr size_t SIZE = sizeof...(Ts);

  /// The derived expression types, each as its nonlinear instantiation.
  ///
  /// Their eval functions don't depend on the ExpressionType.
  ///
  /// @tparam Scalar Scalar type.
  template <typename Scalar>
  using Types = std::tuple<Ts<Scalar, ExpressionType::NONLINEAR>...>;

  /// The largest number of arguments of a derived expression type.
  ///
  /// @tparam Scalar Scalar type.
  template <typename Scalar>
  static constexpr size_t MAX_ARGS =
      std::max({Ts<Scalar, ExpressionType::NONLINEAR>::NUM_ARGS...});

  /// Returns T's index in the list, or SIZE if it isn't in the list.
  ///
  /// @tparam T Derived expression type template.
  template <template <typename, ExpressionType> typename T>
  static consteval size_t index_of() {
    size_t index = 0;
    for (bool match :
         {std::is_same_v<T<double, ExpressionType::NONLINEAR>,
                         Ts<double, ExpressionType::NONLINEAR>>...}) {
      if (match) {
        break;
      }
      ++index;
    }
    return index;
  }
};

/// The derived expression type templates of the non-leaf operators.
///
/// An operator is its template's index in this list (see Expression::op()).
using NonLeafExpressionTemplates = ExpressionTemplates<
    AbsExpression, AcosExpression, AsinExpression, AtanExpression,
    Atan2Expression, BinaryMinusExpression, BinaryPlusExpression,
    CbrtExpression, CosExpression, CoshExpression, DivExpression, ErfExpression,
    ExpExpression, Hypot2Expression, Hypot3Expression, IfElseExpression,
    LogExpression, Log10Expression, MaxExpression, MinExpression,
    MultExpression, PowExpression, SignExpression, SinExpression,
    SinhExpression, SqrtExpression, TanExpression, TanhExpression,
    UnaryMinusExpression>;

static_assert(NonLeafExpressionTemplates::SIZE <= UINT8_MAX + 1,
              "uint8_t is too small for the number of non-leaf operators");

template <template <typename, ExpressionType> typename T>
consteval uint8_t op_of() {
  constexpr size_t index = NonLeafExpressionTemplates::index_of<T>();
  static_assert(index < NonLeafExpressionTemplates::SIZE,
                "T must be in NonLeafExpressionTemplates");
  return static_cast<uint8_t>(index);
}

/// The largest number of arguments of a non-leaf operator.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
inline constexpr size_t MAX_ARGS = NonLeafExpressionTemplates::MAX_ARGS<Scalar>;

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
    using T =
        std::tuple_element_t<Begin, NonLeafExpressionTemplates::Types<Scalar>>;
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
SLEIPNIR_FORCE_INLINE decltype(auto) visit_op(uint8_t op, F&& f) {
  return visit_op_in<Scalar, 0, NonLeafExpressionTemplates::SIZE>(
      static_cast<size_t>(op), f);
}

}  // namespace slp::detail
