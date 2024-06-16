// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <utility>

#include <wpi/SmallVector.h>

#include "sleipnir/autodiff/ExpressionType.hpp"
#include "sleipnir/util/IntrusiveSharedPtr.hpp"
#include "sleipnir/util/Pool.hpp"

namespace sleipnir::detail {

// The global pool allocator uses a thread-local static pool resource, which
// isn't guaranteed to be initialized properly across DLL boundaries on Windows
#ifdef _WIN32
inline constexpr bool kUsePoolAllocator = false;
#else
inline constexpr bool kUsePoolAllocator = true;
#endif

struct Expression;

inline void IntrusiveSharedPtrIncRefCount(Expression* expr);
inline void IntrusiveSharedPtrDecRefCount(Expression* expr);

/**
 * Typedef for intrusive shared pointer to Expression.
 */
using ExpressionPtr = IntrusiveSharedPtr<Expression>;

/**
 * Creates an intrusive shared pointer to an expression from the global pool
 * allocator.
 *
 * @tparam T The derived expression type.
 * @param args Constructor arguments for Expression.
 */
template <typename T, typename... Args>
static ExpressionPtr MakeExpressionPtr(Args&&... args) {
  if constexpr (kUsePoolAllocator) {
    return AllocateIntrusiveShared<T>(GlobalPoolAllocator<T>(),
                                      std::forward<Args>(args)...);
  } else {
    return MakeIntrusiveShared<T>(std::forward<Args>(args)...);
  }
}

template <ExpressionType T>
struct BinaryMinusExpression;

template <ExpressionType T>
struct BinaryPlusExpression;

struct ConstExpression;

template <ExpressionType T>
struct DivExpression;

template <ExpressionType T>
struct MultExpression;

template <ExpressionType T>
struct UnaryMinusExpression;

/**
 * An autodiff expression node.
 */
struct Expression {
  /// The value of the expression node.
  double value = 0.0;

  /// The adjoint of the expression node used during autodiff.
  double adjoint = 0.0;

  /// Counts incoming edges for this node.
  uint32_t incomingEdges = 0;

  /// This expression's column in a Jacobian, or -1 otherwise.
  int32_t col = -1;

  /// The adjoint of the expression node used during gradient expression tree
  /// generation.
  ExpressionPtr adjointExpr;

  /// Reference count for intrusive shared pointer.
  uint32_t refCount = 0;

  /// Expression arguments.
  std::array<ExpressionPtr, 2> args{nullptr, nullptr};

  /**
   * Constructs a constant expression with a value of zero.
   */
  constexpr Expression() = default;

  /**
   * Constructs a nullary expression (an operator with no arguments).
   *
   * @param value The expression value.
   */
  explicit constexpr Expression(double value) : value{value} {}

  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit constexpr Expression(ExpressionPtr lhs)
      : args{std::move(lhs), nullptr} {}

  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr Expression(ExpressionPtr lhs, ExpressionPtr rhs)
      : args{std::move(lhs), std::move(rhs)} {}

  virtual ~Expression() = default;

  /**
   * Returns true if the expression is the given constant.
   *
   * @param constant The constant.
   */
  constexpr bool IsConstant(double constant) const {
    return Type() == ExpressionType::kConstant && value == constant;
  }

  /**
   * Expression-Expression multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend ExpressionPtr operator*(const ExpressionPtr& lhs,
                                 const ExpressionPtr& rhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->IsConstant(0.0)) {
      // Return zero
      return lhs;
    } else if (rhs->IsConstant(0.0)) {
      // Return zero
      return rhs;
    } else if (lhs->IsConstant(1.0)) {
      return rhs;
    } else if (rhs->IsConstant(1.0)) {
      return lhs;
    }

    // Evaluate constant
    if (lhs->Type() == kConstant && rhs->Type() == kConstant) {
      return MakeExpressionPtr<ConstExpression>(lhs->value * rhs->value);
    }

    // Evaluate expression type
    if (lhs->Type() == kConstant) {
      if (rhs->Type() == kLinear) {
        return MakeExpressionPtr<MultExpression<kLinear>>(lhs, rhs);
      } else if (rhs->Type() == kQuadratic) {
        return MakeExpressionPtr<MultExpression<kQuadratic>>(lhs, rhs);
      } else {
        return MakeExpressionPtr<MultExpression<kNonlinear>>(lhs, rhs);
      }
    } else if (rhs->Type() == kConstant) {
      if (lhs->Type() == kLinear) {
        return MakeExpressionPtr<MultExpression<kLinear>>(lhs, rhs);
      } else if (lhs->Type() == kQuadratic) {
        return MakeExpressionPtr<MultExpression<kQuadratic>>(lhs, rhs);
      } else {
        return MakeExpressionPtr<MultExpression<kNonlinear>>(lhs, rhs);
      }
    } else if (lhs->Type() == kLinear && rhs->Type() == kLinear) {
      return MakeExpressionPtr<MultExpression<kQuadratic>>(lhs, rhs);
    } else {
      return MakeExpressionPtr<MultExpression<kNonlinear>>(lhs, rhs);
    }
  }

  /**
   * Expression-Expression division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend ExpressionPtr operator/(const ExpressionPtr& lhs,
                                 const ExpressionPtr& rhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->IsConstant(0.0)) {
      // Return zero
      return lhs;
    } else if (rhs->IsConstant(1.0)) {
      return lhs;
    }

    // Evaluate constant
    if (lhs->Type() == kConstant && rhs->Type() == kConstant) {
      return MakeExpressionPtr<ConstExpression>(lhs->value / rhs->value);
    }

    // Evaluate expression type
    if (rhs->Type() == kConstant) {
      if (lhs->Type() == kLinear) {
        return MakeExpressionPtr<DivExpression<kLinear>>(lhs, rhs);
      } else if (lhs->Type() == kQuadratic) {
        return MakeExpressionPtr<DivExpression<kQuadratic>>(lhs, rhs);
      } else {
        return MakeExpressionPtr<DivExpression<kNonlinear>>(lhs, rhs);
      }
    } else {
      return MakeExpressionPtr<DivExpression<kNonlinear>>(lhs, rhs);
    }
  }

  /**
   * Expression-Expression addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend ExpressionPtr operator+(const ExpressionPtr& lhs,
                                 const ExpressionPtr& rhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs == nullptr || lhs->IsConstant(0.0)) {
      return rhs;
    } else if (rhs == nullptr || rhs->IsConstant(0.0)) {
      return lhs;
    }

    // Evaluate constant
    if (lhs->Type() == kConstant && rhs->Type() == kConstant) {
      return MakeExpressionPtr<ConstExpression>(lhs->value + rhs->value);
    }

    auto type = std::max(lhs->Type(), rhs->Type());
    if (type == kLinear) {
      return MakeExpressionPtr<BinaryPlusExpression<kLinear>>(lhs, rhs);
    } else if (type == kQuadratic) {
      return MakeExpressionPtr<BinaryPlusExpression<kQuadratic>>(lhs, rhs);
    } else {
      return MakeExpressionPtr<BinaryPlusExpression<kNonlinear>>(lhs, rhs);
    }
  }

  /**
   * Expression-Expression compound addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend ExpressionPtr operator+=(ExpressionPtr& lhs,
                                  const ExpressionPtr& rhs) {
    return lhs = lhs + rhs;
  }

  /**
   * Expression-Expression subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend ExpressionPtr operator-(const ExpressionPtr& lhs,
                                 const ExpressionPtr& rhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->IsConstant(0.0)) {
      if (rhs->IsConstant(0.0)) {
        // Return zero
        return rhs;
      } else {
        return -rhs;
      }
    } else if (rhs->IsConstant(0.0)) {
      return lhs;
    }

    // Evaluate constant
    if (lhs->Type() == kConstant && rhs->Type() == kConstant) {
      return MakeExpressionPtr<ConstExpression>(lhs->value - rhs->value);
    }

    auto type = std::max(lhs->Type(), rhs->Type());
    if (type == kLinear) {
      return MakeExpressionPtr<BinaryMinusExpression<kLinear>>(lhs, rhs);
    } else if (type == kQuadratic) {
      return MakeExpressionPtr<BinaryMinusExpression<kQuadratic>>(lhs, rhs);
    } else {
      return MakeExpressionPtr<BinaryMinusExpression<kNonlinear>>(lhs, rhs);
    }
  }

  /**
   * Unary minus operator.
   *
   * @param lhs Operand of unary minus.
   */
  friend ExpressionPtr operator-(const ExpressionPtr& lhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->IsConstant(0.0)) {
      // Return zero
      return lhs;
    }

    // Evaluate constant
    if (lhs->Type() == kConstant) {
      return MakeExpressionPtr<ConstExpression>(-lhs->value);
    }

    if (lhs->Type() == kLinear) {
      return MakeExpressionPtr<UnaryMinusExpression<kLinear>>(lhs);
    } else if (lhs->Type() == kQuadratic) {
      return MakeExpressionPtr<UnaryMinusExpression<kQuadratic>>(lhs);
    } else {
      return MakeExpressionPtr<UnaryMinusExpression<kNonlinear>>(lhs);
    }
  }

  /**
   * Unary plus operator.
   *
   * @param lhs Operand of unary plus.
   */
  friend ExpressionPtr operator+(const ExpressionPtr& lhs) { return lhs; }

  /**
   * Either nullary operator with no arguments, unary operator with one
   * argument, or binary operator with two arguments. This operator is used to
   * update the node's value.
   *
   * @param lhs Left argument to binary operator.
   * @param rhs Right argument to binary operator.
   */
  virtual double Value([[maybe_unused]] double lhs,
                       [[maybe_unused]] double rhs) const = 0;

  /**
   * Returns the type of this expression (constant, linear, quadratic, or
   * nonlinear).
   */
  virtual ExpressionType Type() const = 0;

  /**
   * Returns double adjoint of the left child expression.
   *
   * @param lhs Left argument to binary operator.
   * @param rhs Right argument to binary operator.
   * @param parentAdjoint Adjoint of parent expression.
   */
  virtual double GradL([[maybe_unused]] double lhs, [[maybe_unused]] double rhs,
                       [[maybe_unused]] double parentAdjoint) const {
    return 0.0;
  }

  /**
   * Returns double adjoint of the right child expression.
   *
   * @param lhs Left argument to binary operator.
   * @param rhs Right argument to binary operator.
   * @param parentAdjoint Adjoint of parent expression.
   */
  virtual double GradR([[maybe_unused]] double lhs, [[maybe_unused]] double rhs,
                       [[maybe_unused]] double parentAdjoint) const {
    return 0.0;
  }

  /**
   * Returns Expression adjoint of the left child expression.
   *
   * @param lhs Left argument to binary operator.
   * @param rhs Right argument to binary operator.
   * @param parentAdjoint Adjoint of parent expression.
   */
  virtual ExpressionPtr GradExprL(
      [[maybe_unused]] const ExpressionPtr& lhs,
      [[maybe_unused]] const ExpressionPtr& rhs,
      [[maybe_unused]] const ExpressionPtr& parentAdjoint) const {
    return MakeExpressionPtr<ConstExpression>();
  }

  /**
   * Returns Expression adjoint of the right child expression.
   *
   * @param lhs Left argument to binary operator.
   * @param rhs Right argument to binary operator.
   * @param parentAdjoint Adjoint of parent expression.
   */
  virtual ExpressionPtr GradExprR(
      [[maybe_unused]] const ExpressionPtr& lhs,
      [[maybe_unused]] const ExpressionPtr& rhs,
      [[maybe_unused]] const ExpressionPtr& parentAdjoint) const {
    return MakeExpressionPtr<ConstExpression>();
  }
};

template <ExpressionType T>
struct BinaryMinusExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr BinaryMinusExpression(ExpressionPtr lhs, ExpressionPtr rhs)
      : Expression{std::move(lhs), std::move(rhs)} {
    value = args[0]->value - args[1]->value;
  }

  double Value(double lhs, double rhs) const override { return lhs - rhs; }

  ExpressionType Type() const override { return T; }

  double GradL(double, double, double parentAdjoint) const override {
    return parentAdjoint;
  }

  double GradR(double, double, double parentAdjoint) const override {
    return -parentAdjoint;
  }

  ExpressionPtr GradExprL(const ExpressionPtr&, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint;
  }

  ExpressionPtr GradExprR(const ExpressionPtr&, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return -parentAdjoint;
  }
};

template <ExpressionType T>
struct BinaryPlusExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr BinaryPlusExpression(ExpressionPtr lhs, ExpressionPtr rhs)
      : Expression{std::move(lhs), std::move(rhs)} {
    value = args[0]->value + args[1]->value;
  }

  double Value(double lhs, double rhs) const override { return lhs + rhs; }

  ExpressionType Type() const override { return T; }

  double GradL(double, double, double parentAdjoint) const override {
    return parentAdjoint;
  }

  double GradR(double, double, double parentAdjoint) const override {
    return parentAdjoint;
  }

  ExpressionPtr GradExprL(const ExpressionPtr&, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint;
  }

  ExpressionPtr GradExprR(const ExpressionPtr&, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint;
  }
};

struct ConstExpression final : Expression {
  /**
   * Constructs a constant expression with a value of zero.
   */
  constexpr ConstExpression() = default;

  /**
   * Constructs a nullary expression (an operator with no arguments).
   *
   * @param value The expression value.
   */
  explicit constexpr ConstExpression(double value) : Expression{value} {}

  double Value(double, double) const override { return value; }

  ExpressionType Type() const override { return ExpressionType::kConstant; }
};

struct DecisionVariableExpression final : Expression {
  /**
   * Constructs a decision variable expression with a value of zero.
   */
  constexpr DecisionVariableExpression() = default;

  /**
   * Constructs a nullary expression (an operator with no arguments).
   *
   * @param value The expression value.
   */
  explicit constexpr DecisionVariableExpression(double value)
      : Expression{value} {}

  double Value(double, double) const override { return value; }

  ExpressionType Type() const override { return ExpressionType::kLinear; }
};

template <ExpressionType T>
struct DivExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr DivExpression(ExpressionPtr lhs, ExpressionPtr rhs)
      : Expression{std::move(lhs), std::move(rhs)} {
    value = args[0]->value / args[1]->value;
  }

  double Value(double lhs, double rhs) const override { return lhs / rhs; }

  ExpressionType Type() const override { return T; }

  double GradL(double, double rhs, double parentAdjoint) const override {
    return parentAdjoint / rhs;
  };

  double GradR(double lhs, double rhs, double parentAdjoint) const override {
    return parentAdjoint * -lhs / (rhs * rhs);
  }

  ExpressionPtr GradExprL(const ExpressionPtr&, const ExpressionPtr& rhs,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint / rhs;
  }

  ExpressionPtr GradExprR(const ExpressionPtr& lhs, const ExpressionPtr& rhs,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * -lhs / (rhs * rhs);
  }
};

template <ExpressionType T>
struct MultExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr MultExpression(ExpressionPtr lhs, ExpressionPtr rhs)
      : Expression{std::move(lhs), std::move(rhs)} {
    value = args[0]->value * args[1]->value;
  }

  double Value(double lhs, double rhs) const override { return lhs * rhs; }

  ExpressionType Type() const override { return T; }

  double GradL([[maybe_unused]] double lhs, double rhs,
               double parentAdjoint) const override {
    return parentAdjoint * rhs;
  }

  double GradR(double lhs, [[maybe_unused]] double rhs,
               double parentAdjoint) const override {
    return parentAdjoint * lhs;
  }

  ExpressionPtr GradExprL([[maybe_unused]] const ExpressionPtr& lhs,
                          const ExpressionPtr& rhs,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * rhs;
  }

  ExpressionPtr GradExprR(const ExpressionPtr& lhs,
                          [[maybe_unused]] const ExpressionPtr& rhs,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * lhs;
  }
};

template <ExpressionType T>
struct UnaryMinusExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit constexpr UnaryMinusExpression(ExpressionPtr lhs)
      : Expression{std::move(lhs)} {
    value = -args[0]->value;
  }

  double Value(double lhs, double) const override { return -lhs; }

  ExpressionType Type() const override { return T; }

  double GradL(double, double, double parentAdjoint) const override {
    return -parentAdjoint;
  }

  ExpressionPtr GradExprL(const ExpressionPtr&, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return -parentAdjoint;
  }
};

inline ExpressionPtr exp(const ExpressionPtr& x);
inline ExpressionPtr sin(const ExpressionPtr& x);
inline ExpressionPtr sinh(const ExpressionPtr& x);
inline ExpressionPtr sqrt(const ExpressionPtr& x);

/**
 * Refcount increment for intrusive shared pointer.
 *
 * @param expr The shared pointer's managed object.
 */
inline void IntrusiveSharedPtrIncRefCount(Expression* expr) {
  ++expr->refCount;
}

/**
 * Refcount decrement for intrusive shared pointer.
 *
 * @param expr The shared pointer's managed object.
 */
inline void IntrusiveSharedPtrDecRefCount(Expression* expr) {
  // If a deeply nested tree is being deallocated all at once, calling the
  // Expression destructor when expr's refcount reaches zero can cause a stack
  // overflow. Instead, we iterate over its children to decrement their
  // refcounts and deallocate them.
  wpi::SmallVector<Expression*> stack;
  stack.emplace_back(expr);

  while (!stack.empty()) {
    auto elem = stack.back();
    stack.pop_back();

    // Decrement the current node's refcount. If it reaches zero, deallocate the
    // node and enqueue its children so their refcounts are decremented too.
    if (--elem->refCount == 0) {
      if (elem->adjointExpr != nullptr) {
        stack.emplace_back(elem->adjointExpr.Get());
      }
      for (auto& arg : elem->args) {
        if (arg != nullptr) {
          stack.emplace_back(arg.Get());
        }
      }

      // Not calling the destructor here is safe because it only decrements
      // refcounts, which was already done above.
      if constexpr (kUsePoolAllocator) {
        auto alloc = GlobalPoolAllocator<Expression>();
        std::allocator_traits<decltype(alloc)>::deallocate(alloc, elem,
                                                           sizeof(Expression));
      }
    }
  }
}

struct AbsExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit constexpr AbsExpression(ExpressionPtr lhs)
      : Expression{std::move(lhs)} {
    value = args[0]->value < 0 ? -args[0]->value : args[0]->value;
  }

  double Value(double x, double) const override { return std::abs(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    if (x < 0.0) {
      return -parentAdjoint;
    } else if (x > 0.0) {
      return parentAdjoint;
    } else {
      return 0.0;
    }
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    if (x->value < 0.0) {
      return -parentAdjoint;
    } else if (x->value > 0.0) {
      return parentAdjoint;
    } else {
      // Return zero
      return MakeExpressionPtr<ConstExpression>();
    }
  }
};

/**
 * std::abs() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr abs(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::abs(x->value));
  }

  return MakeExpressionPtr<AbsExpression>(x);
}

struct AcosExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit AcosExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::acos(args[0]->value);
  }

  double Value(double x, double) const override { return std::acos(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return -parentAdjoint / std::sqrt(1.0 - x * x);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return -parentAdjoint /
           sleipnir::detail::sqrt(MakeExpressionPtr<ConstExpression>(1.0) -
                                  x * x);
  }
};

/**
 * std::acos() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr acos(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(std::numbers::pi / 2.0);
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::acos(x->value));
  }

  return MakeExpressionPtr<AcosExpression>(x);
}

struct AsinExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit AsinExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::asin(args[0]->value);
  }

  double Value(double x, double) const override { return std::asin(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint / std::sqrt(1.0 - x * x);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint / sleipnir::detail::sqrt(
                               MakeExpressionPtr<ConstExpression>(1.0) - x * x);
  }
};

/**
 * std::asin() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr asin(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::asin(x->value));
  }

  return MakeExpressionPtr<AsinExpression>(x);
}

struct AtanExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit AtanExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::atan(args[0]->value);
  }

  double Value(double x, double) const override { return std::atan(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint / (1.0 + x * x);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint / (MakeExpressionPtr<ConstExpression>(1.0) + x * x);
  }
};

/**
 * std::atan() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr atan(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::atan(x->value));
  }

  return MakeExpressionPtr<AtanExpression>(x);
}

struct Atan2Expression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  Atan2Expression(ExpressionPtr lhs, ExpressionPtr rhs)
      : Expression{std::move(lhs), std::move(rhs)} {
    value = std::atan2(args[0]->value, args[1]->value);
  }

  double Value(double y, double x) const override { return std::atan2(y, x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double y, double x, double parentAdjoint) const override {
    return parentAdjoint * x / (y * y + x * x);
  }

  double GradR(double y, double x, double parentAdjoint) const override {
    return parentAdjoint * -y / (y * y + x * x);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& y, const ExpressionPtr& x,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * x / (y * y + x * x);
  }

  ExpressionPtr GradExprR(const ExpressionPtr& y, const ExpressionPtr& x,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * -y / (y * y + x * x);
  }
};

/**
 * std::atan2() for Expressions.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
inline ExpressionPtr atan2(const ExpressionPtr& y, const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (y->IsConstant(0.0)) {
    // Return zero
    return y;
  } else if (x->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(std::numbers::pi / 2.0);
  }

  // Evaluate constant
  if (y->Type() == kConstant && x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::atan2(y->value, x->value));
  }

  return MakeExpressionPtr<Atan2Expression>(y, x);
}

struct CosExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit CosExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::cos(args[0]->value);
  }

  double Value(double x, double) const override { return std::cos(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return -parentAdjoint * std::sin(x);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * -sleipnir::detail::sin(x);
  }
};

/**
 * std::cos() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr cos(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(1.0);
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::cos(x->value));
  }

  return MakeExpressionPtr<CosExpression>(x);
}

struct CoshExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit CoshExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::cosh(args[0]->value);
  }

  double Value(double x, double) const override { return std::cosh(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint * std::sinh(x);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * sleipnir::detail::sinh(x);
  }
};

/**
 * std::cosh() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr cosh(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(1.0);
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::cosh(x->value));
  }

  return MakeExpressionPtr<CoshExpression>(x);
}

struct ErfExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit ErfExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::erf(args[0]->value);
  }

  double Value(double x, double) const override { return std::erf(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint * 2.0 * std::numbers::inv_sqrtpi * std::exp(-x * x);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint *
           MakeExpressionPtr<ConstExpression>(2.0 * std::numbers::inv_sqrtpi) *
           sleipnir::detail::exp(-x * x);
  }
};

/**
 * std::erf() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr erf(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::erf(x->value));
  }

  return MakeExpressionPtr<ErfExpression>(x);
}

struct ExpExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit ExpExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::exp(args[0]->value);
  }

  double Value(double x, double) const override { return std::exp(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint * std::exp(x);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * sleipnir::detail::exp(x);
  }
};

/**
 * std::exp() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr exp(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(1.0);
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::exp(x->value));
  }

  return MakeExpressionPtr<ExpExpression>(x);
}

inline ExpressionPtr hypot(const ExpressionPtr& x, const ExpressionPtr& y);

struct HypotExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  HypotExpression(ExpressionPtr lhs, ExpressionPtr rhs)
      : Expression{std::move(lhs), std::move(rhs)} {
    value = std::hypot(args[0]->value, args[1]->value);
  }

  double Value(double x, double y) const override { return std::hypot(x, y); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double y, double parentAdjoint) const override {
    return parentAdjoint * x / std::hypot(x, y);
  }

  double GradR(double x, double y, double parentAdjoint) const override {
    return parentAdjoint * y / std::hypot(x, y);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr& y,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * x / sleipnir::detail::hypot(x, y);
  }

  ExpressionPtr GradExprR(const ExpressionPtr& x, const ExpressionPtr& y,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * y / sleipnir::detail::hypot(x, y);
  }
};

/**
 * std::hypot() for Expressions.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
inline ExpressionPtr hypot(const ExpressionPtr& x, const ExpressionPtr& y) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return y;
  } else if (y->IsConstant(0.0)) {
    return x;
  }

  // Evaluate constant
  if (x->Type() == kConstant && y->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::hypot(x->value, y->value));
  }

  return MakeExpressionPtr<HypotExpression>(x, y);
}

struct LogExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit LogExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::log(args[0]->value);
  }

  double Value(double x, double) const override { return std::log(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint / x;
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint / x;
  }
};

/**
 * std::log() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr log(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::log(x->value));
  }

  return MakeExpressionPtr<LogExpression>(x);
}

struct Log10Expression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit Log10Expression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::log10(args[0]->value);
  }

  double Value(double x, double) const override { return std::log10(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint / (std::numbers::ln10 * x);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint /
           (MakeExpressionPtr<ConstExpression>(std::numbers::ln10) * x);
  }
};

/**
 * std::log10() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr log10(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::log10(x->value));
  }

  return MakeExpressionPtr<Log10Expression>(x);
}

inline ExpressionPtr pow(const ExpressionPtr& base, const ExpressionPtr& power);

template <ExpressionType T>
struct PowExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  PowExpression(ExpressionPtr lhs, ExpressionPtr rhs)
      : Expression{std::move(lhs), std::move(rhs)} {
    value = std::pow(args[0]->value, args[1]->value);
  }

  double Value(double base, double power) const override {
    return std::pow(base, power);
  }

  ExpressionType Type() const override { return T; }

  double GradL(double base, double power, double parentAdjoint) const override {
    return parentAdjoint * std::pow(base, power - 1) * power;
  }

  double GradR(double base, double power, double parentAdjoint) const override {
    // Since x * std::log(x) -> 0 as x -> 0
    if (base == 0.0) {
      return 0.0;
    } else {
      return parentAdjoint * std::pow(base, power - 1) * base * std::log(base);
    }
  }

  ExpressionPtr GradExprL(const ExpressionPtr& base, const ExpressionPtr& power,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint *
           sleipnir::detail::pow(
               base, power - MakeExpressionPtr<ConstExpression>(1.0)) *
           power;
  }

  ExpressionPtr GradExprR(const ExpressionPtr& base, const ExpressionPtr& power,
                          const ExpressionPtr& parentAdjoint) const override {
    // Since x * std::log(x) -> 0 as x -> 0
    if (base->value == 0.0) {
      // Return zero
      return base;
    } else {
      return parentAdjoint *
             sleipnir::detail::pow(
                 base, power - MakeExpressionPtr<ConstExpression>(1.0)) *
             base * sleipnir::detail::log(base);
    }
  }
};

/**
 * std::pow() for Expressions.
 *
 * @param base The base.
 * @param power The power.
 */
inline ExpressionPtr pow(const ExpressionPtr& base,
                         const ExpressionPtr& power) {
  using enum ExpressionType;

  // Prune expression
  if (base->IsConstant(0.0)) {
    // Return zero
    return base;
  } else if (base->IsConstant(1.0)) {
    // Return one
    return base;
  }
  if (power->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(1.0);
  } else if (power->IsConstant(1.0)) {
    return base;
  }

  // Evaluate constant
  if (base->Type() == kConstant && power->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(
        std::pow(base->value, power->value));
  }

  if (power->IsConstant(2.0)) {
    if (base->Type() == kLinear) {
      return MakeExpressionPtr<MultExpression<kQuadratic>>(base, base);
    } else {
      return MakeExpressionPtr<MultExpression<kNonlinear>>(base, base);
    }
  }

  return MakeExpressionPtr<PowExpression<kNonlinear>>(base, power);
}

struct SignExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit constexpr SignExpression(ExpressionPtr lhs)
      : Expression{std::move(lhs)} {
    if (args[0]->value < 0.0) {
      value = -1.0;
    } else if (args[0]->value == 0.0) {
      value = 0.0;
    } else {
      value = 1.0;
    }
  }

  double Value(double x, double) const override {
    if (x < 0.0) {
      return -1.0;
    } else if (x == 0.0) {
      return 0.0;
    } else {
      return 1.0;
    }
  }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double, double, double) const override { return 0.0; }

  ExpressionPtr GradExprL(const ExpressionPtr&, const ExpressionPtr&,
                          const ExpressionPtr&) const override {
    // Return zero
    return MakeExpressionPtr<ConstExpression>();
  }
};

/**
 * sign() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr sign(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Evaluate constant
  if (x->Type() == kConstant) {
    if (x->value < 0.0) {
      return MakeExpressionPtr<ConstExpression>(-1.0);
    } else if (x->value == 0.0) {
      // Return zero
      return x;
    } else {
      return MakeExpressionPtr<ConstExpression>(1.0);
    }
  }

  return MakeExpressionPtr<SignExpression>(x);
}

struct SinExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit SinExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::sin(args[0]->value);
  }

  double Value(double x, double) const override { return std::sin(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint * std::cos(x);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * sleipnir::detail::cos(x);
  }
};

/**
 * std::sin() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr sin(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::sin(x->value));
  }

  return MakeExpressionPtr<SinExpression>(x);
}

struct SinhExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit SinhExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::sinh(args[0]->value);
  }

  double Value(double x, double) const override { return std::sinh(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint * std::cosh(x);
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * sleipnir::detail::cosh(x);
  }
};

/**
 * std::sinh() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr sinh(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::sinh(x->value));
  }

  return MakeExpressionPtr<SinhExpression>(x);
}

struct SqrtExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit SqrtExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::sqrt(args[0]->value);
  }

  double Value(double x, double) const override { return std::sqrt(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint / (2.0 * std::sqrt(x));
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint / (MakeExpressionPtr<ConstExpression>(2.0) *
                            sleipnir::detail::sqrt(x));
  }
};

/**
 * std::sqrt() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr sqrt(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Evaluate constant
  if (x->Type() == kConstant) {
    if (x->value == 0.0) {
      // Return zero
      return x;
    } else if (x->value == 1.0) {
      return x;
    } else {
      return MakeExpressionPtr<ConstExpression>(std::sqrt(x->value));
    }
  }

  return MakeExpressionPtr<SqrtExpression>(x);
}

struct TanExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit TanExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::tan(args[0]->value);
  }

  double Value(double x, double) const override { return std::tan(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint / (std::cos(x) * std::cos(x));
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint /
           (sleipnir::detail::cos(x) * sleipnir::detail::cos(x));
  }
};

/**
 * std::tan() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr tan(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::tan(x->value));
  }

  return MakeExpressionPtr<TanExpression>(x);
}

struct TanhExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param lhs Unary operator's operand.
   */
  explicit TanhExpression(ExpressionPtr lhs) : Expression{std::move(lhs)} {
    value = std::tanh(args[0]->value);
  }

  double Value(double x, double) const override { return std::tanh(x); }

  ExpressionType Type() const override { return ExpressionType::kNonlinear; }

  double GradL(double x, double, double parentAdjoint) const override {
    return parentAdjoint / (std::cosh(x) * std::cosh(x));
  }

  ExpressionPtr GradExprL(const ExpressionPtr& x, const ExpressionPtr&,
                          const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint /
           (sleipnir::detail::cosh(x) * sleipnir::detail::cosh(x));
  }
};

/**
 * std::tanh() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr tanh(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->Type() == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::tanh(x->value));
  }

  return MakeExpressionPtr<TanhExpression>(x);
}

}  // namespace sleipnir::detail
