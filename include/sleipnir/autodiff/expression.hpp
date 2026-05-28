// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <string_view>
#include <utility>

#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/util/function_ref.hpp"
#include "sleipnir/util/intrusive_shared_ptr.hpp"
#include "sleipnir/util/pool.hpp"

namespace slp::detail {

// The global pool allocator uses a thread-local static pool resource, which
// isn't guaranteed to be initialized properly across DLL boundaries on Windows
#ifdef _WIN32
inline constexpr bool USE_POOL_ALLOCATOR = false;
#else
inline constexpr bool USE_POOL_ALLOCATOR = false;
#endif

template <typename Scalar>
struct Expression;

template <typename Scalar>
constexpr void inc_ref_count(Expression<Scalar>* expr);
template <typename Scalar>
constexpr void dec_ref_count(Expression<Scalar>* expr);

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
  // FIXME: Custom pool allocator is probably giving blocks of wrong size.
  //
  // Options to fix it:
  //
  // 1. Increase block size to fit binary expression
  // 2. Make separate custom pools for nullary, unary, and binary expressions
  // 3. Use polymorphic allocator to implement (2)
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

template <typename Scalar>
struct ConstantExpression;

template <typename Scalar, ExpressionType T>
struct DivExpression;

template <typename Scalar, ExpressionType T>
struct MultExpression;

template <typename Scalar, ExpressionType T>
struct UnaryMinusExpression;

/// Creates an intrusive shared pointer to a constant expression.
///
/// @tparam Scalar Scalar type.
/// @param value The expression value.
template <typename Scalar>
ExpressionPtr<Scalar> constant_ptr(Scalar value);

/// An autodiff expression node.
///
/// @tparam Scalar Scalar type.
template <typename Scalar_>
struct Expression {
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// The value of the expression node.
  Scalar val{0};

  /// The adjoint of the expression node, used during autodiff.
  Scalar adjoint{0};

  /// The adjoint of the expression node, used during gradient expression tree
  /// generation.
  ExpressionPtr<Scalar> adjoint_expr;

  /// Scratch space for various graph algorithms.
  ///
  /// In expression_graph.hpp's topological_sort(), scratch counts incoming
  /// edges for this node, offset by -1 so -1 means no edges.
  ///
  /// In Hessian and Jacobian constructors, scratch represents this expression's
  /// column in a Jacobian, or -1 otherwise.
  ///
  /// They share a default state of -1 to avoid extra assignments.
  int32_t scratch = -1;

  /// Reference count for intrusive shared pointer.
  uint32_t ref_count = 0;

  /// Constructs a constant expression with a value of zero.
  constexpr Expression() = default;

  /// Constructs a nullary expression (an operator with no arguments).
  ///
  /// @param value The expression value.
  explicit constexpr Expression(Scalar value) : val{value} {}

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

    // Evaluate expression type
    if (lhs->type() == CONSTANT) {
      if (rhs->type() == LINEAR) {
        return make_expression_ptr<MultExpression<Scalar, LINEAR>>(lhs, rhs);
      } else if (rhs->type() == QUADRATIC) {
        return make_expression_ptr<MultExpression<Scalar, QUADRATIC>>(lhs, rhs);
      } else {
        return make_expression_ptr<MultExpression<Scalar, NONLINEAR>>(lhs, rhs);
      }
    } else if (rhs->type() == CONSTANT) {
      if (lhs->type() == LINEAR) {
        return make_expression_ptr<MultExpression<Scalar, LINEAR>>(lhs, rhs);
      } else if (lhs->type() == QUADRATIC) {
        return make_expression_ptr<MultExpression<Scalar, QUADRATIC>>(lhs, rhs);
      } else {
        return make_expression_ptr<MultExpression<Scalar, NONLINEAR>>(lhs, rhs);
      }
    } else if (lhs->type() == LINEAR && rhs->type() == LINEAR) {
      return make_expression_ptr<MultExpression<Scalar, QUADRATIC>>(lhs, rhs);
    } else {
      return make_expression_ptr<MultExpression<Scalar, NONLINEAR>>(lhs, rhs);
    }
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

    // Evaluate expression type
    if (rhs->type() == CONSTANT) {
      if (lhs->type() == LINEAR) {
        return make_expression_ptr<DivExpression<Scalar, LINEAR>>(lhs, rhs);
      } else if (lhs->type() == QUADRATIC) {
        return make_expression_ptr<DivExpression<Scalar, QUADRATIC>>(lhs, rhs);
      } else {
        return make_expression_ptr<DivExpression<Scalar, NONLINEAR>>(lhs, rhs);
      }
    } else {
      return make_expression_ptr<DivExpression<Scalar, NONLINEAR>>(lhs, rhs);
    }
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
    if (type == LINEAR) {
      return make_expression_ptr<BinaryPlusExpression<Scalar, LINEAR>>(lhs,
                                                                       rhs);
    } else if (type == QUADRATIC) {
      return make_expression_ptr<BinaryPlusExpression<Scalar, QUADRATIC>>(lhs,
                                                                          rhs);
    } else {
      return make_expression_ptr<BinaryPlusExpression<Scalar, NONLINEAR>>(lhs,
                                                                          rhs);
    }
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
    if (type == LINEAR) {
      return make_expression_ptr<BinaryMinusExpression<Scalar, LINEAR>>(lhs,
                                                                        rhs);
    } else if (type == QUADRATIC) {
      return make_expression_ptr<BinaryMinusExpression<Scalar, QUADRATIC>>(lhs,
                                                                           rhs);
    } else {
      return make_expression_ptr<BinaryMinusExpression<Scalar, NONLINEAR>>(lhs,
                                                                           rhs);
    }
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

    if (lhs->type() == LINEAR) {
      return make_expression_ptr<UnaryMinusExpression<Scalar, LINEAR>>(lhs);
    } else if (lhs->type() == QUADRATIC) {
      return make_expression_ptr<UnaryMinusExpression<Scalar, QUADRATIC>>(lhs);
    } else {
      return make_expression_ptr<UnaryMinusExpression<Scalar, NONLINEAR>>(lhs);
    }
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

  /// Accumulates the child adjoints as Scalars.
  virtual void accumulate_adjoints() const {}

  /// Accumulates the child adjoints as Expressions.
  virtual void accumulate_adjoints_expr() const {}
};

template <typename Scalar>
ExpressionPtr<Scalar> constant_ptr(Scalar value) {
  return make_expression_ptr<ConstantExpression<Scalar>>(value);
}

template <typename Scalar>
ExpressionPtr<Scalar> cbrt(const ExpressionPtr<Scalar>& x);
template <typename Scalar>
ExpressionPtr<Scalar> exp(const ExpressionPtr<Scalar>& x);
template <typename Scalar>
ExpressionPtr<Scalar> sin(const ExpressionPtr<Scalar>& x);
template <typename Scalar>
ExpressionPtr<Scalar> sinh(const ExpressionPtr<Scalar>& x);
template <typename Scalar>
ExpressionPtr<Scalar> sqrt(const ExpressionPtr<Scalar>& x);

/// Derived expression type for binary minus operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct BinaryMinusExpression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> lhs;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> rhs;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param lhs Binary operator's left operand.
  /// @param rhs Binary operator's right operand.
  constexpr BinaryMinusExpression(ExpressionPtr<Scalar> lhs,
                                  ExpressionPtr<Scalar> rhs)
      : lhs{std::move(lhs)}, rhs{std::move(rhs)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(lhs.get());
    func(rhs.get());
  }

  Scalar value() const override { return lhs->val - rhs->val; }

  ExpressionType type() const override { return T; }

  std::string_view name() const override { return "binary minus"; }

  void accumulate_adjoints() const override {
    lhs->adjoint += grad_l();
    rhs->adjoint += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    lhs->adjoint_expr += grad_expr_l();
    rhs->adjoint_expr += grad_expr_r();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const { return this->adjoint; }

  /// Returns ∂/∂r as a Scalar.
  ///
  /// @return ∂/∂r as a Scalar.
  Scalar grad_r() const { return -this->adjoint; }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const { return this->adjoint_expr; }

  /// Returns ∂/∂r as an Expression.
  ///
  /// @return ∂/∂r as an Expression.
  ExpressionPtr<Scalar> grad_expr_r() const { return -this->adjoint_expr; }
};

/// Derived expression type for binary plus operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct BinaryPlusExpression final : Expression<Scalar> {
  /// @param lhs Binary operator's left operand.
  ExpressionPtr<Scalar> lhs;

  /// @param rhs Binary operator's right operand.
  ExpressionPtr<Scalar> rhs;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param lhs Binary operator's left operand.
  /// @param rhs Binary operator's right operand.
  constexpr BinaryPlusExpression(ExpressionPtr<Scalar> lhs,
                                 ExpressionPtr<Scalar> rhs)
      : lhs{std::move(lhs)}, rhs{std::move(rhs)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(lhs.get());
    func(rhs.get());
  }

  Scalar value() const override { return lhs->val + rhs->val; }

  ExpressionType type() const override { return T; }

  std::string_view name() const override { return "binary plus"; }

  void accumulate_adjoints() const override {
    lhs->adjoint += grad_l();
    rhs->adjoint += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    lhs->adjoint_expr += grad_expr_l();
    rhs->adjoint_expr += grad_expr_r();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const { return this->adjoint; }

  /// Returns ∂/∂r as a Scalar.
  ///
  /// @return ∂/∂r as a Scalar.
  Scalar grad_r() const { return this->adjoint; }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const { return this->adjoint_expr; }

  /// Returns ∂/∂r as an Expression.
  ///
  /// @return ∂/∂r as an Expression.
  ExpressionPtr<Scalar> grad_expr_r() const { return this->adjoint_expr; }
};

/// Derived expression type for cbrt().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct CbrtExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr CbrtExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::cbrt;
    return cbrt(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "cbrt"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::cbrt;

    Scalar c = cbrt(x->val);
    return this->adjoint / (Scalar(3) * c * c);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    auto c = cbrt(x);
    return this->adjoint_expr / (constant_ptr(Scalar(3)) * c * c);
  }
};

/// cbrt() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cbrt(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::cbrt;

  // Evaluate constant
  if (x->type() == CONSTANT) {
    if (x->val == Scalar(0)) {
      // Return zero
      return x;
    } else if (x->val == Scalar(-1) || x->val == Scalar(1)) {
      return x;
    } else {
      return constant_ptr(cbrt(x->val));
    }
  }

  return make_expression_ptr<CbrtExpression<Scalar>>(x);
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
};

/// Derived expression type for decision variable.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct DecisionVariableExpression final : Expression<Scalar> {
  /// Constructs a decision variable expression with a value of zero.
  constexpr DecisionVariableExpression() = default;

  /// Constructs a nullary expression (an operator with no arguments).
  ///
  /// @param value The expression value.
  explicit constexpr DecisionVariableExpression(Scalar value)
      : Expression<Scalar>{value} {}

  Scalar value() const override { return this->val; }

  ExpressionType type() const override { return ExpressionType::LINEAR; }

  std::string_view name() const override { return "decision variable"; }
};

/// Derived expression type for binary division operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct DivExpression final : Expression<Scalar> {
  /// @param lhs Binary operator's left operand.
  ExpressionPtr<Scalar> lhs;

  /// @param rhs Binary operator's right operand.
  ExpressionPtr<Scalar> rhs;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param lhs Binary operator's left operand.
  /// @param rhs Binary operator's right operand.
  constexpr DivExpression(ExpressionPtr<Scalar> lhs, ExpressionPtr<Scalar> rhs)
      : lhs{std::move(lhs)}, rhs{std::move(rhs)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(lhs.get());
    func(rhs.get());
  }

  Scalar value() const override { return lhs->val / rhs->val; }

  ExpressionType type() const override { return T; }

  std::string_view name() const override { return "division"; }

  void accumulate_adjoints() const override {
    lhs->adjoint += grad_l();
    rhs->adjoint += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    lhs->adjoint_expr += grad_expr_l();
    rhs->adjoint_expr += grad_expr_r();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const { return this->adjoint / rhs->val; };

  /// Returns ∂/∂r as a Scalar.
  ///
  /// @return ∂/∂r as a Scalar.
  Scalar grad_r() const {
    return this->adjoint * -lhs->val / (rhs->val * rhs->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const { return this->adjoint_expr / rhs; }

  /// Returns ∂/∂r as an Expression.
  ///
  /// @return ∂/∂r as an Expression.
  ExpressionPtr<Scalar> grad_expr_r() const {
    return this->adjoint_expr * -lhs / (rhs * rhs);
  }
};

/// Derived expression type for binary multiplication operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct MultExpression final : Expression<Scalar> {
  /// @param lhs Binary operator's left operand.
  ExpressionPtr<Scalar> lhs;

  /// @param rhs Binary operator's right operand.
  ExpressionPtr<Scalar> rhs;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param lhs Binary operator's left operand.
  /// @param rhs Binary operator's right operand.
  constexpr MultExpression(ExpressionPtr<Scalar> lhs, ExpressionPtr<Scalar> rhs)
      : lhs{std::move(lhs)}, rhs{std::move(rhs)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(lhs.get());
    func(rhs.get());
  }

  Scalar value() const override { return lhs->val * rhs->val; }

  ExpressionType type() const override { return T; }

  std::string_view name() const override { return "multiplication"; }

  void accumulate_adjoints() const override {
    lhs->adjoint += grad_l();
    rhs->adjoint += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    lhs->adjoint_expr += grad_expr_l();
    rhs->adjoint_expr += grad_expr_r();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const { return this->adjoint * rhs->val; }

  /// Returns ∂/∂r as a Scalar.
  ///
  /// @return ∂/∂r as a Scalar.
  Scalar grad_r() const { return this->adjoint * lhs->val; }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const { return this->adjoint_expr * rhs; }

  /// Returns ∂/∂r as an Expression.
  ///
  /// @return ∂/∂r as an Expression.
  ExpressionPtr<Scalar> grad_expr_r() const { return this->adjoint_expr * lhs; }
};

/// Derived expression type for unary minus operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct UnaryMinusExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> lhs;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param lhs Unary operator's operand.
  explicit constexpr UnaryMinusExpression(ExpressionPtr<Scalar> lhs)
      : lhs{std::move(lhs)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(lhs.get());
  }

  Scalar value() const override { return -lhs->val; }

  ExpressionType type() const override { return T; }

  std::string_view name() const override { return "unary minus"; }

  void accumulate_adjoints() const override { lhs->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    lhs->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const { return -this->adjoint; }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const { return -this->adjoint_expr; }
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
      if (elem->adjoint_expr != nullptr) {
        stack.emplace_back(elem->adjoint_expr.get());
      }
      elem->visit_args([&stack](const auto& arg) { stack.emplace_back(arg); });

      // Not calling the destructor here is safe because it only decrements
      // refcounts, which was already done above.
      if constexpr (USE_POOL_ALLOCATOR) {
        auto alloc = global_pool_allocator<Expression<Scalar>>();
        std::allocator_traits<decltype(alloc)>::deallocate(
            alloc, elem, sizeof(Expression<Scalar>));
      } else {
        operator delete(elem);
      }
    }
  }
}

/// Derived expression type for abs().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AbsExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr AbsExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::abs;
    return abs(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "abs"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    if (x->val < Scalar(0)) {
      return -this->adjoint;
    } else if (x->val > Scalar(0)) {
      return this->adjoint;
    } else {
      return Scalar(0);
    }
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    if (x->val < Scalar(0)) {
      return -this->adjoint_expr;
    } else if (x->val > Scalar(0)) {
      return this->adjoint_expr;
    } else {
      return constant_ptr(Scalar(0));
    }
  }
};

/// abs() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> abs(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::abs;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(abs(x->val));
  }

  return make_expression_ptr<AbsExpression<Scalar>>(x);
}

/// Derived expression type for acos().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AcosExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr AcosExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::acos;
    return acos(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "acos"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::sqrt;
    return -this->adjoint / sqrt(Scalar(1) - x->val * x->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return -this->adjoint_expr / sqrt(constant_ptr(Scalar(1)) - x * x);
  }
};

/// acos() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> acos(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::acos;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return constant_ptr(Scalar(std::numbers::pi) / Scalar(2));
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(acos(x->val));
  }

  return make_expression_ptr<AcosExpression<Scalar>>(x);
}

/// Derived expression type for asin().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AsinExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr AsinExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::asin;
    return asin(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "asin"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::sqrt;
    return this->adjoint / sqrt(Scalar(1) - x->val * x->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr / sqrt(constant_ptr(Scalar(1)) - x * x);
  }
};

/// asin() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> asin(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::asin;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(asin(x->val));
  }

  return make_expression_ptr<AsinExpression<Scalar>>(x);
}

/// Derived expression type for atan().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AtanExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr AtanExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::atan;
    return atan(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "atan"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    return this->adjoint / (Scalar(1) + x->val * x->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr / (constant_ptr(Scalar(1)) + x * x);
  }
};

/// atan() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> atan(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::atan;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(atan(x->val));
  }

  return make_expression_ptr<AtanExpression<Scalar>>(x);
}

/// Derived expression type for atan2().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Atan2Expression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> y;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> x;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param y Binary operator's left operand.
  /// @param x Binary operator's right operand.
  constexpr Atan2Expression(ExpressionPtr<Scalar> y, ExpressionPtr<Scalar> x)
      : y{std::move(y)}, x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(y.get());
    func(x.get());
  }

  Scalar value() const override {
    using std::atan2;
    return atan2(y->val, x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "atan2"; }

  void accumulate_adjoints() const override {
    y->adjoint += grad_l();
    x->adjoint += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    y->adjoint_expr += grad_expr_l();
    x->adjoint_expr += grad_expr_r();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    return this->adjoint * x->val / (y->val * y->val + x->val * x->val);
  }

  /// Returns ∂/∂r as a Scalar.
  ///
  /// @return ∂/∂r as a Scalar.
  Scalar grad_r() const {
    return this->adjoint * -y->val / (y->val * y->val + x->val * x->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr * x / (y * y + x * x);
  }

  /// Returns ∂/∂r as an Expression.
  ///
  /// @return ∂/∂r as an Expression.
  ExpressionPtr<Scalar> grad_expr_r() const {
    return this->adjoint_expr * -y / (y * y + x * x);
  }
};

/// atan2() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param y The y argument.
/// @param x The x argument.
template <typename Scalar>
ExpressionPtr<Scalar> atan2(const ExpressionPtr<Scalar>& y,
                            const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::atan2;

  // Prune expression
  if (y->is_constant(Scalar(0))) {
    // Return zero, which y currently is
    return y;
  } else if (x->is_constant(Scalar(0))) {
    return constant_ptr(Scalar(std::numbers::pi) / Scalar(2));
  }

  // Evaluate constant
  if (y->type() == CONSTANT && x->type() == CONSTANT) {
    return constant_ptr(atan2(y->val, x->val));
  }

  return make_expression_ptr<Atan2Expression<Scalar>>(y, x);
}

/// Derived expression type for cos().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct CosExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr CosExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::cos;
    return cos(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "cos"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::sin;
    return this->adjoint * -sin(x->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr * -sin(x);
  }
};

/// cos() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cos(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::cos;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return constant_ptr(Scalar(1));
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(cos(x->val));
  }

  return make_expression_ptr<CosExpression<Scalar>>(x);
}

/// Derived expression type for cosh().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct CoshExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr CoshExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::cosh;
    return cosh(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "cosh"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::sinh;
    return this->adjoint * sinh(x->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr * sinh(x);
  }
};

/// cosh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cosh(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::cosh;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return constant_ptr(Scalar(1));
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(cosh(x->val));
  }

  return make_expression_ptr<CoshExpression<Scalar>>(x);
}

/// Derived expression type for erf().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct ErfExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr ErfExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::erf;
    return erf(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "erf"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::exp;
    return this->adjoint * Scalar(2.0 * std::numbers::inv_sqrtpi) *
           exp(-x->val * x->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr *
           constant_ptr(Scalar(2.0 * std::numbers::inv_sqrtpi)) * exp(-x * x);
  }
};

/// erf() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> erf(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::erf;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(erf(x->val));
  }

  return make_expression_ptr<ErfExpression<Scalar>>(x);
}

/// Derived expression type for exp().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct ExpExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr ExpExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::exp;
    return exp(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "exp"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::exp;
    return this->adjoint * exp(x->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr * exp(x);
  }
};

/// exp() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> exp(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::exp;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return constant_ptr(Scalar(1));
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(exp(x->val));
  }

  return make_expression_ptr<ExpExpression<Scalar>>(x);
}

template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y);

/// Derived expression type for hypot().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct HypotExpression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> x;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> y;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param x Binary operator's left operand.
  /// @param y Binary operator's right operand.
  constexpr HypotExpression(ExpressionPtr<Scalar> x, ExpressionPtr<Scalar> y)
      : x{std::move(x)}, y{std::move(y)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
    func(y.get());
  }

  Scalar value() const override {
    using std::hypot;
    return hypot(x->val, y->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "hypot"; }

  void accumulate_adjoints() const override {
    x->adjoint += grad_l();
    y->adjoint += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
    y->adjoint_expr += grad_expr_r();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::hypot;
    return this->adjoint * x->val / hypot(x->val, y->val);
  }

  /// Returns ∂/∂r as a Scalar.
  ///
  /// @return ∂/∂r as a Scalar.
  Scalar grad_r() const {
    using std::hypot;
    return this->adjoint * y->val / hypot(x->val, y->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr * x / hypot(x, y);
  }

  /// Returns ∂/∂r as an Expression.
  ///
  /// @return ∂/∂r as an Expression.
  ExpressionPtr<Scalar> grad_expr_r() const {
    return this->adjoint_expr * y / hypot(x, y);
  }
};

/// hypot() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The x argument.
/// @param y The y argument.
template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y) {
  using enum ExpressionType;
  using std::hypot;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return y;
  } else if (y->is_constant(Scalar(0))) {
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT && y->type() == CONSTANT) {
    return constant_ptr(hypot(x->val, y->val));
  }

  return make_expression_ptr<HypotExpression<Scalar>>(x, y);
}

/// Derived expression type for log().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct LogExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr LogExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::log;
    return log(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "log"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const { return this->adjoint / x->val; }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const { return this->adjoint_expr / x; }
};

/// log() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> log(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::log;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(log(x->val));
  }

  return make_expression_ptr<LogExpression<Scalar>>(x);
}

/// Derived expression type for log10().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Log10Expression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr Log10Expression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::log10;
    return log10(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "log10"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    return this->adjoint / (Scalar(std::numbers::ln10) * x->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr / (constant_ptr(Scalar(std::numbers::ln10)) * x);
  }
};

/// log10() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> log10(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::log10;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(log10(x->val));
  }

  return make_expression_ptr<Log10Expression<Scalar>>(x);
}

/// Derived expression type for max().
///
/// Returns the greater of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct MaxExpression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> a;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> b;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param a Binary operator's left operand.
  /// @param b Binary operator's right operand.
  constexpr MaxExpression(ExpressionPtr<Scalar> a, ExpressionPtr<Scalar> b)
      : a{std::move(a)}, b{std::move(b)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(a.get());
    func(b.get());
  }

  Scalar value() const override {
    using std::max;
    return max(a->val, b->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "max"; }

  void accumulate_adjoints() const override {
    a->adjoint += grad_l();
    b->adjoint += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    a->adjoint_expr += grad_expr_l();
    b->adjoint_expr += grad_expr_r();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    if (a->val >= b->val) {
      return this->adjoint;
    } else {
      return Scalar(0);
    }
  }

  /// Returns ∂/∂r as a Scalar.
  ///
  /// @return ∂/∂r as a Scalar.
  Scalar grad_r() const {
    if (b->val > a->val) {
      return this->adjoint;
    } else {
      return Scalar(0);
    }
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    if (a->val >= b->val) {
      return this->adjoint_expr;
    } else {
      return constant_ptr(Scalar(0));
    }
  }

  /// Returns ∂/∂r as an Expression.
  ///
  /// @return ∂/∂r as an Expression.
  ExpressionPtr<Scalar> grad_expr_r() const {
    if (b->val > a->val) {
      return this->adjoint_expr;
    } else {
      return constant_ptr(Scalar(0));
    }
  }
};

/// max() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param a The a argument.
/// @param b The b argument.
template <typename Scalar>
ExpressionPtr<Scalar> max(const ExpressionPtr<Scalar>& a,
                          const ExpressionPtr<Scalar>& b) {
  using enum ExpressionType;
  using std::max;

  // Evaluate constant
  if (a->type() == CONSTANT && b->type() == CONSTANT) {
    return constant_ptr(max(a->val, b->val));
  }

  return make_expression_ptr<MaxExpression<Scalar>>(a, b);
}

/// Derived expression type for min().
///
/// Returns the lesser of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct MinExpression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> a;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> b;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param a Binary operator's left operand.
  /// @param b Binary operator's right operand.
  constexpr MinExpression(ExpressionPtr<Scalar> a, ExpressionPtr<Scalar> b)
      : a{std::move(a)}, b{std::move(b)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(a.get());
    func(b.get());
  }

  Scalar value() const override {
    using std::min;
    return min(a->val, b->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "min"; }

  void accumulate_adjoints() const override {
    a->adjoint += grad_l();
    b->adjoint += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    a->adjoint_expr += grad_expr_l();
    b->adjoint_expr += grad_expr_r();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    if (a->val <= b->val) {
      return this->adjoint;
    } else {
      return Scalar(0);
    }
  }

  /// Returns ∂/∂r as a Scalar.
  ///
  /// @return ∂/∂r as a Scalar.
  Scalar grad_r() const {
    if (b->val < a->val) {
      return this->adjoint;
    } else {
      return Scalar(0);
    }
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    if (a->val <= b->val) {
      return this->adjoint_expr;
    } else {
      return constant_ptr(Scalar(0));
    }
  }

  /// Returns ∂/∂r as an Expression.
  ///
  /// @return ∂/∂r as an Expression.
  ExpressionPtr<Scalar> grad_expr_r() const {
    if (b->val < a->val) {
      return this->adjoint_expr;
    } else {
      return constant_ptr(Scalar(0));
    }
  }
};

/// min() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param a The a argument.
/// @param b The b argument.
template <typename Scalar>
ExpressionPtr<Scalar> min(const ExpressionPtr<Scalar>& a,
                          const ExpressionPtr<Scalar>& b) {
  using enum ExpressionType;
  using std::min;

  // Evaluate constant
  if (a->type() == CONSTANT && b->type() == CONSTANT) {
    return constant_ptr(min(a->val, b->val));
  }

  return make_expression_ptr<MinExpression<Scalar>>(a, b);
}

template <typename Scalar>
ExpressionPtr<Scalar> pow(const ExpressionPtr<Scalar>& base,
                          const ExpressionPtr<Scalar>& power);

/// Derived expression type for pow().
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct PowExpression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> base;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> power;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param base Binary operator's left operand.
  /// @param power Binary operator's right operand.
  constexpr PowExpression(ExpressionPtr<Scalar> base,
                          ExpressionPtr<Scalar> power)
      : base{std::move(base)}, power{std::move(power)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(base.get());
    func(power.get());
  }

  Scalar value() const override {
    using std::pow;
    return pow(base->val, power->val);
  }

  ExpressionType type() const override { return T; }

  std::string_view name() const override { return "pow"; }

  void accumulate_adjoints() const override {
    base->adjoint += grad_l();
    power->adjoint += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    base->adjoint_expr += grad_expr_l();
    power->adjoint_expr += grad_expr_r();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::pow;
    return this->adjoint * pow(base->val, power->val - Scalar(1)) * power->val;
  }

  /// Returns ∂/∂r as a Scalar.
  ///
  /// @return ∂/∂r as a Scalar.
  Scalar grad_r() const {
    using std::log;
    using std::pow;

    // Since x log(x) -> 0 as x -> 0
    if (base->val == Scalar(0)) {
      return Scalar(0);
    } else {
      return this->adjoint * pow(base->val, power->val) * log(base->val);
    }
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr * pow(base, power - constant_ptr(Scalar(1))) *
           power;
  }

  /// Returns ∂/∂r as an Expression.
  ///
  /// @return ∂/∂r as an Expression.
  ExpressionPtr<Scalar> grad_expr_r() const {
    // Since x log(x) -> 0 as x -> 0
    if (base->val == Scalar(0)) {
      // Return zero
      return base;
    } else {
      return this->adjoint_expr * pow(base, power) * log(base);
    }
  }
};

/// pow() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param base The base.
/// @param power The power.
template <typename Scalar>
ExpressionPtr<Scalar> pow(const ExpressionPtr<Scalar>& base,
                          const ExpressionPtr<Scalar>& power) {
  using enum ExpressionType;
  using std::pow;

  // Prune expression
  if (base->is_constant(Scalar(0))) {
    // Return zero, which base currently is
    return base;
  } else if (base->is_constant(Scalar(1))) {
    // Return one, which base currently is
    return base;
  }
  if (power->is_constant(Scalar(0))) {
    return constant_ptr(Scalar(1));
  } else if (power->is_constant(Scalar(1))) {
    // Return base unmodified
    return base;
  }

  // Evaluate constant
  if (base->type() == CONSTANT && power->type() == CONSTANT) {
    return constant_ptr(pow(base->val, power->val));
  }

  if (power->is_constant(Scalar(2))) {
    if (base->type() == LINEAR) {
      return make_expression_ptr<MultExpression<Scalar, QUADRATIC>>(base, base);
    } else {
      return make_expression_ptr<MultExpression<Scalar, NONLINEAR>>(base, base);
    }
  }

  return make_expression_ptr<PowExpression<Scalar, NONLINEAR>>(base, power);
}

/// Derived expression type for sign().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SignExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr SignExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    if (x->val < Scalar(0)) {
      return Scalar(-1);
    } else if (x->val == Scalar(0)) {
      return Scalar(0);
    } else {
      return Scalar(1);
    }
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "sign"; }
};

/// sign() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sign(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;

  // Evaluate constant
  if (x->type() == CONSTANT) {
    if (x->val < Scalar(0)) {
      return constant_ptr(Scalar(-1));
    } else if (x->val == Scalar(0)) {
      // Return zero
      return x;
    } else {
      return constant_ptr(Scalar(1));
    }
  }

  return make_expression_ptr<SignExpression<Scalar>>(x);
}

/// Derived expression type for sin().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SinExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr SinExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::sin;
    return sin(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "sin"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::cos;
    return this->adjoint * cos(x->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr * cos(x);
  }
};

/// sin() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sin(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::sin;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(sin(x->val));
  }

  return make_expression_ptr<SinExpression<Scalar>>(x);
}

/// Derived expression type for sinh().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SinhExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr SinhExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::sinh;
    return sinh(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "sinh"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::cosh;
    return this->adjoint * cosh(x->val);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr * cosh(x);
  }
};

/// sinh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sinh(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::sinh;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(sinh(x->val));
  }

  return make_expression_ptr<SinhExpression<Scalar>>(x);
}

/// Derived expression type for sqrt().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SqrtExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr SqrtExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::sqrt;
    return sqrt(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "sqrt"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::sqrt;
    return this->adjoint / (Scalar(2) * sqrt(x->val));
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adjoint_expr / (constant_ptr(Scalar(2)) * sqrt(x));
  }
};

/// sqrt() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sqrt(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::sqrt;

  // Evaluate constant
  if (x->type() == CONSTANT) {
    if (x->val == Scalar(0)) {
      // Return zero
      return x;
    } else if (x->val == Scalar(1)) {
      return x;
    } else {
      return constant_ptr(sqrt(x->val));
    }
  }

  return make_expression_ptr<SqrtExpression<Scalar>>(x);
}

/// Derived expression type for tan().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct TanExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr TanExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::tan;
    return tan(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "tan"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::cos;

    auto c = cos(x->val);
    return this->adjoint / (c * c);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    auto c = cos(x);
    return this->adjoint_expr / (c * c);
  }
};

/// tan() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> tan(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::tan;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(tan(x->val));
  }

  return make_expression_ptr<TanExpression<Scalar>>(x);
}

/// Derived expression type for tanh().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct TanhExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr TanhExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::tanh;
    return tanh(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "tanh"; }

  void accumulate_adjoints() const override { x->adjoint += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adjoint_expr += grad_expr_l();
  }

 private:
  /// Returns ∂/∂l as a Scalar.
  ///
  /// @return ∂/∂l as a Scalar.
  Scalar grad_l() const {
    using std::cosh;

    auto c = cosh(x->val);
    return this->adjoint / (c * c);
  }

  /// Returns ∂/∂l as an Expression.
  ///
  /// @return ∂/∂l as an Expression.
  ExpressionPtr<Scalar> grad_expr_l() const {
    auto c = cosh(x);
    return this->adjoint_expr / (c * c);
  }
};

/// tanh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> tanh(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::tanh;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(tanh(x->val));
  }

  return make_expression_ptr<TanhExpression<Scalar>>(x);
}

}  // namespace slp::detail
