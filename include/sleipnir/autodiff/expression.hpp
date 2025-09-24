// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/util/force_inline.hpp"
#include "sleipnir/util/function_ref.hpp"
#include "sleipnir/util/intrusive_shared_ptr.hpp"
#include "sleipnir/util/pool.hpp"

namespace slp::detail {

#ifdef _WIN32
inline constexpr bool USE_POOL_ALLOCATOR = false;
#else
inline constexpr bool USE_POOL_ALLOCATOR = true;
#endif

/// Expression operators, one per derived expression type.
enum class ExpressionOp : uint8_t {
  /// abs().
  ABS,
  /// acos().
  ACOS,
  /// asin().
  ASIN,
  /// atan().
  ATAN,
  /// atan2().
  ATAN2,
  /// Binary minus operator.
  BINARY_MINUS,
  /// Binary plus operator.
  BINARY_PLUS,
  /// cbrt().
  CBRT,
  /// cos().
  COS,
  /// cosh().
  COSH,
  /// Binary division operator.
  DIV,
  /// erf().
  ERF,
  /// exp().
  EXP,
  /// Two-argument hypot().
  HYPOT2,
  /// Three-argument hypot().
  HYPOT3,
  /// if_else().
  IF_ELSE,
  /// log().
  LOG,
  /// log10().
  LOG10,
  /// max().
  MAX,
  /// min().
  MIN,
  /// Binary multiplication operator.
  MULT,
  /// pow().
  POW,
  /// sign().
  SIGN,
  /// sin().
  SIN,
  /// sinh().
  SINH,
  /// sqrt().
  SQRT,
  /// tan().
  TAN,
  /// tanh().
  TANH,
  /// Unary minus operator.
  UNARY_MINUS,
  /// Constant.
  CONSTANT,
  /// Decision variable.
  DECISION_VARIABLE,
};

template <typename Scalar>
struct Expression;

/// The maximum number of arguments an expression can have.
inline constexpr size_t MAX_ARGS = 4;

/// Array with one element per expression argument in visit_args() order.
///
/// @tparam T Element type.
template <typename T>
using ArgArray = std::array<T, MAX_ARGS>;

/// Typedef for intrusive shared pointer to Expression.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
using ExpressionPtr = IntrusiveSharedPtr<Expression<Scalar>>;

/// Creates an intrusive shared pointer to an expression from the global pool
/// allocator.
///
/// @tparam T The derived expression type.
/// @param args Constructor arguments for Expression.
template <typename T, typename... Args>
static ExpressionPtr<typename T::Scalar> make_expression_ptr(Args&&... args) {
  if constexpr (USE_POOL_ALLOCATOR) {
    return allocate_intrusive_shared<T>(global_pool_allocator<T>(),
                                        std::forward<Args>(args)...);
  } else {
    return make_intrusive_shared<T>(std::forward<Args>(args)...);
  }
}

template <typename Scalar, ExpressionType T>
struct BinaryMinusExpression;

template <typename Scalar, ExpressionType T>
struct BinaryPlusExpression;

template <typename Scalar, ExpressionType T>
struct DivExpression;

template <typename Scalar, ExpressionType T>
struct MultExpression;

template <typename Scalar, ExpressionType T>
struct UnaryMinusExpression;

/// Creates an intrusive shared pointer to a derived expression type templated
/// on ExpressionType, instantiated with the given expression type.
///
/// @tparam T Derived expression type template.
/// @tparam Scalar Scalar type.
/// @tparam Args Argument types.
/// @param type The expression's type. Types other than linear and quadratic
///     are instantiated as nonlinear.
/// @param args The arguments.
template <template <typename, ExpressionType> typename T, typename Scalar,
          typename... Args>
SLEIPNIR_FORCE_INLINE ExpressionPtr<Scalar> make_typed_expression_ptr(
    ExpressionType type, const Args&... args) {
  using enum ExpressionType;

  if (type == LINEAR) {
    return make_expression_ptr<T<Scalar, LINEAR>>(args...);
  } else if (type == QUADRATIC) {
    return make_expression_ptr<T<Scalar, QUADRATIC>>(args...);
  } else {
    return make_expression_ptr<T<Scalar, NONLINEAR>>(args...);
  }
}

/// Creates an intrusive shared pointer to a constant expression.
///
/// @tparam Scalar Scalar type.
/// @param value The expression value.
template <typename Scalar>
ExpressionPtr<Scalar> constant_ptr(Scalar value);

/// Returns a constant as the given value type.
///
/// Lets a derived expression type's derivative formulas be written once for
/// both Scalar values and ExpressionPtrs.
///
/// @tparam U Scalar or ExpressionPtr<Scalar>.
/// @tparam Scalar Scalar type.
/// @param value The constant.
/// @return The constant as a U.
template <typename U, typename Scalar>
U lit(Scalar value) {
  if constexpr (std::same_as<U, Scalar>) {
    return value;
  } else {
    return constant_ptr(value);
  }
}

/// An autodiff expression node.
///
/// @tparam Scalar Scalar type.
template <typename Scalar_>
struct Expression {
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// The value of the expression node.
  Scalar val{0};

  /// Scratch space for various graph algorithms.
  ///
  /// In expression_graph.hpp's topological_sort(), scratch counts incoming
  /// edges for this node, offset by -1 so -1 means no edges.
  ///
  /// In AdjointTape's constructor, scratch temporarily holds this expression's
  /// index in the topological list.
  ///
  /// All users restore it to -1 afterward, a shared default that avoids extra
  /// assignments.
  int32_t scratch = -1;

  /// Reference count for intrusive shared pointer.
  uint32_t ref_count : 31 = 0;

  /// True if the expression is a leaf node (a nullary expression).
  ///
  /// Graph traversals check this to avoid virtual calls on leaves. The check
  /// also feeds the branch predictor history that helps it predict the virtual
  /// call targets of non-leaf nodes.
  uint32_t is_leaf : 1 = 0;

  /// Constructs a constant expression with a value of zero.
  constexpr Expression() = default;

  /// Constructs a nullary expression (an operator with no arguments).
  ///
  /// @param value The expression value.
  explicit constexpr Expression(Scalar value) : val{value}, is_leaf{1} {}

  virtual ~Expression() = default;

  /// Returns true if the expression is the given constant.
  ///
  /// @param constant The constant.
  /// @return True if the expression is the given constant.
  constexpr bool is_constant(Scalar constant) const {
    return type() == ExpressionType::CONSTANT && val == constant;
  }

  /// Expression-Expression multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  friend ExpressionPtr<Scalar> operator*(const ExpressionPtr<Scalar>& lhs,
                                         const ExpressionPtr<Scalar>& rhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->is_constant(Scalar(0))) {
      // Return zero, which lhs currently is
      return lhs;
    } else if (rhs->is_constant(Scalar(0))) {
      // Return zero, which rhs currently is
      return rhs;
    } else if (lhs->is_constant(Scalar(1))) {
      // Return rhs unmodified
      return rhs;
    } else if (rhs->is_constant(Scalar(1))) {
      // Return lhs unmodified
      return lhs;
    }

    // Evaluate constant
    if (lhs->type() == CONSTANT && rhs->type() == CONSTANT) {
      return constant_ptr(lhs->val * rhs->val);
    }

    // Evaluate expression type. Scaling by a constant keeps the other
    // operand's type.
    auto type = NONLINEAR;
    if (lhs->type() == CONSTANT || rhs->type() == CONSTANT) {
      type = std::max(lhs->type(), rhs->type());
    } else if (lhs->type() == LINEAR && rhs->type() == LINEAR) {
      type = QUADRATIC;
    }
    return make_typed_expression_ptr<MultExpression, Scalar>(type, lhs, rhs);
  }

  /// Expression-Expression division operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  friend ExpressionPtr<Scalar> operator/(const ExpressionPtr<Scalar>& lhs,
                                         const ExpressionPtr<Scalar>& rhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->is_constant(Scalar(0))) {
      // Return zero, which lhs currently is
      return lhs;
    } else if (rhs->is_constant(Scalar(1))) {
      // Return lhs unmodified
      return lhs;
    }

    // Evaluate constant
    if (lhs->type() == CONSTANT && rhs->type() == CONSTANT) {
      return constant_ptr(lhs->val / rhs->val);
    }

    // Evaluate expression type. Dividing by a constant keeps the numerator's
    // type.
    auto type = rhs->type() == CONSTANT ? lhs->type() : NONLINEAR;
    return make_typed_expression_ptr<DivExpression, Scalar>(type, lhs, rhs);
  }

  /// Expression-Expression addition operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  friend ExpressionPtr<Scalar> operator+(const ExpressionPtr<Scalar>& lhs,
                                         const ExpressionPtr<Scalar>& rhs) {
    using enum ExpressionType;

    // Prune expression. We check for nullptr because operator+ is used in
    // adjoint accumulation, and child nodes can be null.
    if (lhs == nullptr || lhs->is_constant(Scalar(0))) {
      // Return rhs unmodified
      return rhs;
    } else if (rhs == nullptr || rhs->is_constant(Scalar(0))) {
      // Return lhs unmodified
      return lhs;
    }

    // Evaluate constant
    if (lhs->type() == CONSTANT && rhs->type() == CONSTANT) {
      return constant_ptr(lhs->val + rhs->val);
    }

    auto type = std::max(lhs->type(), rhs->type());
    return make_typed_expression_ptr<BinaryPlusExpression, Scalar>(type, lhs,
                                                                   rhs);
  }

  /// Expression-Expression compound addition operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  friend ExpressionPtr<Scalar> operator+=(ExpressionPtr<Scalar>& lhs,
                                          const ExpressionPtr<Scalar>& rhs) {
    return lhs = lhs + rhs;
  }

  /// Expression-Expression subtraction operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  friend ExpressionPtr<Scalar> operator-(const ExpressionPtr<Scalar>& lhs,
                                         const ExpressionPtr<Scalar>& rhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->is_constant(Scalar(0))) {
      if (rhs->is_constant(Scalar(0))) {
        // Return zero, which rhs currently is
        return rhs;
      } else {
        // Return rhs negated
        return -rhs;
      }
    } else if (rhs->is_constant(Scalar(0))) {
      // Return lhs unmodified
      return lhs;
    }

    // Evaluate constant
    if (lhs->type() == CONSTANT && rhs->type() == CONSTANT) {
      return constant_ptr(lhs->val - rhs->val);
    }

    auto type = std::max(lhs->type(), rhs->type());
    return make_typed_expression_ptr<BinaryMinusExpression, Scalar>(type, lhs,
                                                                    rhs);
  }

  /// Unary minus operator.
  ///
  /// @param lhs Operand of unary minus.
  friend ExpressionPtr<Scalar> operator-(const ExpressionPtr<Scalar>& lhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->is_constant(Scalar(0))) {
      // Return zero, which lhs currently is
      return lhs;
    }

    // Evaluate constant
    if (lhs->type() == CONSTANT) {
      return constant_ptr(-lhs->val);
    }

    return make_typed_expression_ptr<UnaryMinusExpression, Scalar>(lhs->type(),
                                                                   lhs);
  }

  /// Unary plus operator.
  ///
  /// @param lhs Operand of unary plus.
  friend ExpressionPtr<Scalar> operator+(const ExpressionPtr<Scalar>& lhs) {
    return lhs;
  }

  /// Runs a function on each argument.
  ///
  /// @param func The function to run.
  virtual void visit_args(
      [[maybe_unused]] function_ref<void(Expression<Scalar>* arg)> func) const {
  }

  /// Either nullary operator with no arguments, unary operator with one
  /// argument, or binary operator with two arguments. This operator is used to
  /// update the node's value.
  ///
  /// @return The node's value.
  virtual Scalar value() const = 0;

  /// Returns the type of this expression (constant, linear, quadratic, or
  /// nonlinear).
  ///
  /// @return The type of this expression.
  virtual ExpressionType type() const = 0;

  /// Returns the name of this expression.
  ///
  /// @return The name of this expression.
  virtual std::string_view name() const = 0;

  /// Returns the operator of this expression.
  ///
  /// @return The operator of this expression.
  virtual ExpressionOp op() const = 0;
};

/// Base of the derived expression types of non-leaf operators.
///
/// Implements Expression's virtual functions from the derived expression type's
/// eval() and NAME and the operator's properties.
///
/// @tparam Derived Derived expression type.
/// @tparam Scalar Scalar type.
/// @tparam Op The expression's operator.
/// @tparam N The number of arguments.
/// @tparam Type The expression's type.
template <typename Derived, typename Scalar, ExpressionOp Op, size_t N,
          ExpressionType Type = ExpressionType::NONLINEAR>
struct NonLeafExpression : Expression<Scalar> {
  /// The expression's operator.
  static constexpr ExpressionOp OP = Op;

  /// The number of arguments.
  static constexpr size_t NUM_ARGS = N;

  /// The arguments in visit_args() order.
  std::array<ExpressionPtr<Scalar>, N> args;

  /// Constructs an expression from its arguments.
  ///
  /// @param args The arguments in visit_args() order.
  template <typename... Args>
    requires(sizeof...(Args) == N &&
             (std::same_as<std::remove_cvref_t<Args>, ExpressionPtr<Scalar>> &&
              ...))
  explicit constexpr NonLeafExpression(Args&&... args)
      : args{std::forward<Args>(args)...} {}

  /// Runs a function on each argument.
  ///
  /// @param func The function to run.
  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const final {
    // A fold instead of a loop, since GCC doesn't unroll the loop. The loop
    // made Hessian setup 2% slower.
    [&]<size_t... I>(std::index_sequence<I...>) {
      (func(args[I].get()), ...);
    }(std::make_index_sequence<N>{});
  }

  /// Returns the node's value computed from its arguments's values.
  ///
  /// @return The node's value.
  Scalar value() const final {
    return [&]<size_t... I>(std::index_sequence<I...>) {
      return static_cast<const Derived&>(*this).eval(args[I]->val...);
    }(std::make_index_sequence<N>{});
  }

  /// Returns the type of this expression.
  ///
  /// @return The type of this expression.
  ExpressionType type() const final { return Type; }

  /// Returns the name of this expression.
  ///
  /// @return The name of this expression.
  std::string_view name() const final { return Derived::NAME; }

  /// Returns the operator of this expression.
  ///
  /// @return The operator of this expression.
  ExpressionOp op() const final { return OP; }
};

/// Creates an intrusive shared pointer to a derived expression type, or to a
/// constant if all its arguments are constant.
///
/// A unary operator that maps its constant argument to itself (e.g., sin(0) =
/// 0) returns the argument instead of allocating a new constant.
///
/// @tparam T Derived expression type.
/// @tparam Args Argument types.
/// @param args The arguments.
template <typename T, typename... Args>
ExpressionPtr<typename T::Scalar> fold_or_make(const Args&... args) {
  using Scalar = typename T::Scalar;

  if (((args->type() == ExpressionType::CONSTANT) && ...)) {
    Scalar val = T::eval(args->val...);
    if constexpr (sizeof...(Args) == 1) {
      const auto& x = std::get<0>(std::forward_as_tuple(args...));
      if (x->val == val) {
        return x;
      }
    }
    return constant_ptr(val);
  }

  return make_expression_ptr<T>(args...);
}

/// Derived expression type for binary minus operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct BinaryMinusExpression final
    : NonLeafExpression<BinaryMinusExpression<Scalar, T>, Scalar,
                        ExpressionOp::BINARY_MINUS, 2, T> {
  using BinaryMinusExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "binary minus";

  /// ∂(a − b)/∂a = 1 and ∂(a − b)/∂b = −1.
  static constexpr std::array<int, 2> UNIT_PARTIAL_SIGNS{1, -1};

  /// Returns the expression's value at the given argument values.
  ///
  /// @param lhs The value of lhs.
  /// @param rhs The value of rhs.
  /// @return The expression's value.
  static Scalar eval(Scalar lhs, Scalar rhs) { return lhs - rhs; }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        [[maybe_unused]] const U& lhs,
                        [[maybe_unused]] const U& rhs) {
    g[0] = lit<U>(Scalar(1));
    g[1] = lit<U>(Scalar(-1));
  }
};

/// Derived expression type for binary plus operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct BinaryPlusExpression final
    : NonLeafExpression<BinaryPlusExpression<Scalar, T>, Scalar,
                        ExpressionOp::BINARY_PLUS, 2, T> {
  using BinaryPlusExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "binary plus";

  /// ∂(a + b)/∂a = 1 and ∂(a + b)/∂b = 1.
  static constexpr std::array<int, 2> UNIT_PARTIAL_SIGNS{1, 1};

  /// Returns the expression's value at the given argument values.
  ///
  /// @param lhs The value of lhs.
  /// @param rhs The value of rhs.
  /// @return The expression's value.
  static Scalar eval(Scalar lhs, Scalar rhs) { return lhs + rhs; }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        [[maybe_unused]] const U& lhs,
                        [[maybe_unused]] const U& rhs) {
    g[0] = lit<U>(Scalar(1));
    g[1] = lit<U>(Scalar(1));
  }
};

/// Derived expression type for binary division operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct DivExpression final : NonLeafExpression<DivExpression<Scalar, T>, Scalar,
                                               ExpressionOp::DIV, 2, T> {
  using DivExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "division";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param lhs The value of lhs.
  /// @param rhs The value of rhs.
  /// @return The expression's value.
  static Scalar eval(Scalar lhs, Scalar rhs) { return lhs / rhs; }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& lhs, const U& rhs) {
    g[0] = lit<U>(Scalar(1)) / rhs;
    g[1] = -lhs / (rhs * rhs);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, [[maybe_unused]] const U& val,
                        const U& lhs, const U& rhs) {
    H[1][0] = lit<U>(Scalar(-1)) / (rhs * rhs);
    H[1][1] = lit<U>(Scalar(2)) * lhs / (rhs * rhs * rhs);
  }
};

/// Derived expression type for binary multiplication operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct MultExpression final
    : NonLeafExpression<MultExpression<Scalar, T>, Scalar, ExpressionOp::MULT,
                        2, T> {
  using MultExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "multiplication";

  /// ∂²(ab)/∂a∂b = 1 regardless of a and b.
  static constexpr bool CONSTANT_SECOND_PARTIALS = true;

  /// Returns the expression's value at the given argument values.
  ///
  /// @param lhs The value of lhs.
  /// @param rhs The value of rhs.
  /// @return The expression's value.
  static Scalar eval(Scalar lhs, Scalar rhs) { return lhs * rhs; }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& lhs, const U& rhs) {
    g[0] = rhs;
    g[1] = lhs;
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, [[maybe_unused]] const U& val,
                        [[maybe_unused]] const U& lhs,
                        [[maybe_unused]] const U& rhs) {
    H[1][0] = lit<U>(Scalar(1));
  }
};

/// Derived expression type for unary minus operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct UnaryMinusExpression final
    : NonLeafExpression<UnaryMinusExpression<Scalar, T>, Scalar,
                        ExpressionOp::UNARY_MINUS, 1, T> {
  using UnaryMinusExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "unary minus";

  /// ∂(−a)/∂a = −1.
  static constexpr std::array<int, 1> UNIT_PARTIAL_SIGNS{-1};

  /// Returns the expression's value at the given argument values.
  ///
  /// @param lhs The value of lhs.
  /// @return The expression's value.
  static Scalar eval(Scalar lhs) { return -lhs; }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        [[maybe_unused]] const U& lhs) {
    g[0] = lit<U>(Scalar(-1));
  }
};

/// Refcount increment for intrusive shared pointer.
///
/// @tparam Scalar Scalar type.
/// @param expr The shared pointer's managed object.
template <typename Scalar>
constexpr void inc_ref_count(Expression<Scalar>* expr) {
  ++expr->ref_count;
}

/// Refcount decrement for intrusive shared pointer.
///
/// @tparam Scalar Scalar type.
/// @param expr The shared pointer's managed object.
template <typename Scalar>
constexpr void dec_ref_count(Expression<Scalar>* expr) {
  // If a deeply nested tree is being deallocated all at once, calling the
  // Expression destructor when expr's refcount reaches zero can cause a stack
  // overflow. Instead, we iterate over its children to decrement their
  // refcounts and deallocate them.
  gch::small_vector<Expression<Scalar>*> stack;
  stack.emplace_back(expr);

  while (!stack.empty()) {
    auto elem = stack.back();
    stack.pop_back();

    // Decrement the current node's refcount. If it reaches zero, deallocate the
    // node and enqueue its children so their refcounts are decremented too.
    if (--elem->ref_count == 0) {
      elem->visit_args([&stack](const auto& arg) { stack.emplace_back(arg); });

      // Not calling the destructor here is safe because it only decrements
      // refcounts, which was already done above.
      if constexpr (USE_POOL_ALLOCATOR) {
        auto alloc = global_pool_allocator<Expression<Scalar>>();
        std::allocator_traits<decltype(alloc)>::deallocate(alloc, elem, 1);
      } else {
        operator delete(elem);
      }
    }
  }
}

/// Derived expression type for constant.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct ConstantExpression final : Expression<Scalar> {
  /// Constructs a nullary expression (an operator with no arguments).
  ///
  /// @param value The expression value.
  explicit constexpr ConstantExpression(Scalar value)
      : Expression<Scalar>{value} {}

  Scalar value() const override { return this->val; }

  ExpressionType type() const override { return ExpressionType::CONSTANT; }

  std::string_view name() const override { return "constant"; }

  /// The expression's operator.
  static constexpr ExpressionOp OP = ExpressionOp::CONSTANT;

  ExpressionOp op() const override { return OP; }
};

template <typename Scalar>
ExpressionPtr<Scalar> constant_ptr(Scalar value) {
  return make_expression_ptr<ConstantExpression<Scalar>>(value);
}

/// Derived expression type for decision variable.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct DecisionVariableExpression final : Expression<Scalar> {
  /// Constructs a decision variable expression with a value of zero.
  constexpr DecisionVariableExpression() : Expression<Scalar>{Scalar(0)} {}

  /// Constructs a nullary expression (an operator with no arguments).
  ///
  /// @param value The expression value.
  explicit constexpr DecisionVariableExpression(Scalar value)
      : Expression<Scalar>{value} {}

  Scalar value() const override { return this->val; }

  ExpressionType type() const override { return ExpressionType::LINEAR; }

  std::string_view name() const override { return "decision variable"; }

  /// The expression's operator.
  static constexpr ExpressionOp OP = ExpressionOp::DECISION_VARIABLE;

  ExpressionOp op() const override { return OP; }
};

}  // namespace slp::detail
