// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/util/flat_map.hpp"
#include "sleipnir/util/function_ref.hpp"
#include "sleipnir/util/intrusive_shared_ptr.hpp"
#include "sleipnir/util/pool.hpp"

namespace slp::detail {

// The global pool allocator uses a thread-local static pool resource, which
// isn't guaranteed to be initialized properly across DLL boundaries on Windows
#ifdef _WIN32
inline constexpr bool USE_POOL_ALLOCATOR = false;
#else
inline constexpr bool USE_POOL_ALLOCATOR = true;
#endif

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
  Scalar adj{0};

  /// The Hessian of the expression node's row, used during autodiff.
  ///
  /// Maps column index to value.
  flat_map<size_t, Scalar> hessian;

  /// This expression's index in the topological list.
  size_t idx = 0;

  /// The adjoint of the expression node, used during gradient expression tree
  /// generation.
  ExpressionPtr<Scalar> adj_expr;

  /// The Hessian of the expression node's row, used during Hessian expression
  /// tree generation.
  ///
  /// Maps column index to value.
  flat_map<size_t, ExpressionPtr<Scalar>> hessian_expr;

  /// True if the expression is a leaf node (a nullary expression).
  ///
  /// Graph traversals check this to avoid virtual calls on leaves. The check
  /// also feeds the branch predictor history that helps it predict the virtual
  /// call targets of non-leaf nodes.
  bool is_leaf = false;

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
  explicit constexpr Expression(Scalar value) : val{value}, is_leaf{true} {}

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

  /// Returns ∂/∂aₖ as a Scalar for each argument aₖ.
  ///
  /// Unlike accumulate_adjoints(), the partial derivatives aren't scaled by the
  /// adjoint.
  ///
  /// @param g The partial derivatives in visit_args() order. Elements left
  ///     empty are structurally zero.
  virtual void grad([[maybe_unused]] ArgArray<std::optional<Scalar>>& g) const {
  }

  /// Returns ∂/∂aₖ as an Expression for each argument aₖ.
  ///
  /// Unlike accumulate_adjoints_expr(), the partial derivatives aren't scaled
  /// by the adjoint.
  ///
  /// @param g The partial derivatives in visit_args() order. Elements left as
  ///     nullptr are structurally zero.
  virtual void grad_expr(
      [[maybe_unused]] ArgArray<ExpressionPtr<Scalar>>& g) const {}

  /// Returns ∂²/∂aⱼ∂aₖ as a Scalar for each pair of arguments aⱼ and aₖ.
  ///
  /// The partial derivatives aren't scaled by the adjoint.
  ///
  /// @param H The second partial derivatives in visit_args() order. Only the
  ///     lower triangle (j ≥ k) is written. Elements left empty are
  ///     structurally zero.
  virtual void hess(
      [[maybe_unused]] ArgArray<ArgArray<std::optional<Scalar>>>& H) const {}

  /// Returns ∂²/∂aⱼ∂aₖ as an Expression for each pair of arguments aⱼ and aₖ.
  ///
  /// The partial derivatives aren't scaled by the adjoint.
  ///
  /// @param H The second partial derivatives in visit_args() order. Only the
  ///     lower triangle (j ≥ k) is written. Elements left as nullptr are
  ///     structurally zero.
  virtual void hess_expr(
      [[maybe_unused]] ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const {}
};

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
    lhs->adj += grad_l();
    rhs->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    lhs->adj_expr += grad_expr_l();
    rhs->adj_expr += grad_expr_r();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    g[0] = Scalar(1);
    g[1] = Scalar(-1);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = constant_ptr(Scalar(1));
    g[1] = constant_ptr(Scalar(-1));
  }

 private:
  Scalar grad_l() const { return this->adj; }

  Scalar grad_r() const { return -this->adj; }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr; }

  ExpressionPtr<Scalar> grad_expr_r() const { return -this->adj_expr; }
};

/// Derived expression type for binary plus operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct BinaryPlusExpression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> lhs;

  /// Binary operator's right operand.
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
    lhs->adj += grad_l();
    rhs->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    lhs->adj_expr += grad_expr_l();
    rhs->adj_expr += grad_expr_r();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    g[0] = Scalar(1);
    g[1] = Scalar(1);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = constant_ptr(Scalar(1));
    g[1] = constant_ptr(Scalar(1));
  }

 private:
  Scalar grad_l() const { return this->adj; }

  Scalar grad_r() const { return this->adj; }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr; }

  ExpressionPtr<Scalar> grad_expr_r() const { return this->adj_expr; }
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::cbrt;

    Scalar c = cbrt(x->val);
    g[0] = Scalar(1) / (Scalar(3) * c * c);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    auto c = cbrt(x);
    g[0] = constant_ptr(Scalar(1)) / (constant_ptr(Scalar(3)) * c * c);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::cbrt;

    Scalar c = cbrt(x->val);
    H[0][0] = Scalar(-2) / (Scalar(9) * x->val * c * c);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    auto c = cbrt(x);
    H[0][0] = constant_ptr(Scalar(-2)) / (constant_ptr(Scalar(9)) * x * c * c);
  }

 private:
  Scalar grad_l() const {
    using std::cbrt;

    Scalar c = cbrt(x->val);
    return this->adj / (Scalar(3) * c * c);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    auto c = cbrt(x);
    return this->adj_expr / (constant_ptr(Scalar(3)) * c * c);
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
    lhs->adj += grad_l();
    rhs->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    lhs->adj_expr += grad_expr_l();
    rhs->adj_expr += grad_expr_r();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    g[0] = Scalar(1) / rhs->val;
    g[1] = -lhs->val / (rhs->val * rhs->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = constant_ptr(Scalar(1)) / rhs;
    g[1] = -lhs / (rhs * rhs);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    H[1][0] = Scalar(-1) / (rhs->val * rhs->val);
    H[1][1] = Scalar(2) * lhs->val / (rhs->val * rhs->val * rhs->val);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    H[1][0] = constant_ptr(Scalar(-1)) / (rhs * rhs);
    H[1][1] = constant_ptr(Scalar(2)) * lhs / (rhs * rhs * rhs);
  }

 private:
  Scalar grad_l() const { return this->adj / rhs->val; };

  Scalar grad_r() const {
    return this->adj * -lhs->val / (rhs->val * rhs->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr / rhs; }

  ExpressionPtr<Scalar> grad_expr_r() const {
    return this->adj_expr * -lhs / (rhs * rhs);
  }
};

/// Derived expression type for binary multiplication operator.
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct MultExpression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> lhs;

  /// Binary operator's right operand.
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
    lhs->adj += grad_l();
    rhs->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    lhs->adj_expr += grad_expr_l();
    rhs->adj_expr += grad_expr_r();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    g[0] = rhs->val;
    g[1] = lhs->val;
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = rhs;
    g[1] = lhs;
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    H[1][0] = Scalar(1);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    H[1][0] = constant_ptr(Scalar(1));
  }

 private:
  Scalar grad_l() const { return this->adj * rhs->val; }

  Scalar grad_r() const { return this->adj * lhs->val; }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * rhs; }

  ExpressionPtr<Scalar> grad_expr_r() const { return this->adj_expr * lhs; }
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

  void accumulate_adjoints() const override { lhs->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    lhs->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    g[0] = Scalar(-1);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = constant_ptr(Scalar(-1));
  }

 private:
  Scalar grad_l() const { return -this->adj; }

  ExpressionPtr<Scalar> grad_expr_l() const { return -this->adj_expr; }
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
      if (elem->adj_expr != nullptr) {
        stack.emplace_back(elem->adj_expr.get());
      }
      // Release ownership of Hessian entries so destroying the Hessian below
      // doesn't recursively decrement their refcounts
      for (auto&& [j, h_i_j] : elem->hessian_expr) {
        if (h_i_j != nullptr) {
          stack.emplace_back(h_i_j.release());
        }
      }
      elem->visit_args([&stack](const auto& arg) { stack.emplace_back(arg); });

      // The Hessians own heap storage, so they must be destroyed
      std::destroy_at(&elem->hessian);
      std::destroy_at(&elem->hessian_expr);

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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    if (x->val < Scalar(0)) {
      g[0] = Scalar(-1);
    } else if (x->val > Scalar(0)) {
      g[0] = Scalar(1);
    } else {
      g[0] = Scalar(0);
    }
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = sign(x);
  }

 private:
  Scalar grad_l() const {
    if (x->val < Scalar(0)) {
      return -this->adj;
    } else if (x->val > Scalar(0)) {
      return this->adj;
    } else {
      return Scalar(0);
    }
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * sign(x); }
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::sqrt;
    g[0] = Scalar(-1) / sqrt(Scalar(1) - x->val * x->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = constant_ptr(Scalar(-1)) / sqrt(constant_ptr(Scalar(1)) - x * x);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::sqrt;

    Scalar s = sqrt(Scalar(1) - x->val * x->val);
    H[0][0] = -x->val / (s * s * s);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    auto s = sqrt(constant_ptr(Scalar(1)) - x * x);
    H[0][0] = -x / (s * s * s);
  }

 private:
  Scalar grad_l() const {
    using std::sqrt;
    return -this->adj / sqrt(Scalar(1) - x->val * x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return -this->adj_expr / sqrt(constant_ptr(Scalar(1)) - x * x);
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::sqrt;
    g[0] = Scalar(1) / sqrt(Scalar(1) - x->val * x->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = constant_ptr(Scalar(1)) / sqrt(constant_ptr(Scalar(1)) - x * x);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::sqrt;

    Scalar s = sqrt(Scalar(1) - x->val * x->val);
    H[0][0] = x->val / (s * s * s);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    auto s = sqrt(constant_ptr(Scalar(1)) - x * x);
    H[0][0] = x / (s * s * s);
  }

 private:
  Scalar grad_l() const {
    using std::sqrt;
    return this->adj / sqrt(Scalar(1) - x->val * x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr / sqrt(constant_ptr(Scalar(1)) - x * x);
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    g[0] = Scalar(1) / (Scalar(1) + x->val * x->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = constant_ptr(Scalar(1)) / (constant_ptr(Scalar(1)) + x * x);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    Scalar t = Scalar(1) + x->val * x->val;
    H[0][0] = Scalar(-2) * x->val / (t * t);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    auto t = constant_ptr(Scalar(1)) + x * x;
    H[0][0] = constant_ptr(Scalar(-2)) * x / (t * t);
  }

 private:
  Scalar grad_l() const { return this->adj / (Scalar(1) + x->val * x->val); }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr / (constant_ptr(Scalar(1)) + x * x);
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
    y->adj += grad_l();
    x->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    y->adj_expr += grad_expr_l();
    x->adj_expr += grad_expr_r();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    Scalar d = y->val * y->val + x->val * x->val;
    g[0] = x->val / d;
    g[1] = -y->val / d;
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    auto d = y * y + x * x;
    g[0] = x / d;
    g[1] = -y / d;
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    Scalar y2 = y->val * y->val;
    Scalar x2 = x->val * x->val;
    Scalar d = y2 + x2;
    Scalar d2 = d * d;
    H[0][0] = Scalar(-2) * x->val * y->val / d2;
    H[1][0] = (y2 - x2) / d2;
    H[1][1] = Scalar(2) * x->val * y->val / d2;
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    auto y2 = y * y;
    auto x2 = x * x;
    auto d = y2 + x2;
    auto d2 = d * d;
    H[0][0] = constant_ptr(Scalar(-2)) * x * y / d2;
    H[1][0] = (y2 - x2) / d2;
    H[1][1] = constant_ptr(Scalar(2)) * x * y / d2;
  }

 private:
  Scalar grad_l() const {
    return this->adj * x->val / (y->val * y->val + x->val * x->val);
  }

  Scalar grad_r() const {
    return this->adj * -y->val / (y->val * y->val + x->val * x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr * x / (y * y + x * x);
  }

  ExpressionPtr<Scalar> grad_expr_r() const {
    return this->adj_expr * -y / (y * y + x * x);
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::sin;
    g[0] = -sin(x->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = -sin(x);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::cos;
    H[0][0] = -cos(x->val);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    H[0][0] = -cos(x);
  }

 private:
  Scalar grad_l() const {
    using std::sin;
    return this->adj * -sin(x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * -sin(x); }
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::sinh;
    g[0] = sinh(x->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = sinh(x);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::cosh;
    H[0][0] = cosh(x->val);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    H[0][0] = cosh(x);
  }

 private:
  Scalar grad_l() const {
    using std::sinh;
    return this->adj * sinh(x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * sinh(x); }
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::exp;
    g[0] = Scalar(2.0 * std::numbers::inv_sqrtpi) * exp(-x->val * x->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = constant_ptr(Scalar(2.0 * std::numbers::inv_sqrtpi)) * exp(-x * x);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::exp;
    H[0][0] = Scalar(-4.0 * std::numbers::inv_sqrtpi) * x->val *
              exp(-x->val * x->val);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    H[0][0] =
        constant_ptr(Scalar(-4.0 * std::numbers::inv_sqrtpi)) * x * exp(-x * x);
  }

 private:
  Scalar grad_l() const {
    using std::exp;
    return this->adj * Scalar(2.0 * std::numbers::inv_sqrtpi) *
           exp(-x->val * x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr *
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::exp;
    g[0] = exp(x->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = exp(x);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::exp;
    H[0][0] = exp(x->val);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    H[0][0] = exp(x);
  }

 private:
  Scalar grad_l() const {
    using std::exp;
    return this->adj * exp(x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * exp(x); }
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

/// Derived expression type for hypot() with two arguments.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Hypot2Expression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> x;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> y;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param x Binary operator's left operand.
  /// @param y Binary operator's right operand.
  constexpr Hypot2Expression(ExpressionPtr<Scalar> x, ExpressionPtr<Scalar> y)
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
    using std::hypot;
    Scalar norm = hypot(x->val, y->val);
    x->adj += this->adj * x->val / norm;
    y->adj += this->adj * y->val / norm;
  }

  void accumulate_adjoints_expr() const override {
    auto norm = hypot(x, y);
    x->adj_expr += this->adj_expr * x / norm;
    y->adj_expr += this->adj_expr * y / norm;
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::hypot;

    Scalar norm = hypot(x->val, y->val);
    g[0] = x->val / norm;
    g[1] = y->val / norm;
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    auto norm = hypot(x, y);
    g[0] = x / norm;
    g[1] = y / norm;
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::hypot;

    Scalar norm = hypot(x->val, y->val);
    Scalar norm3 = norm * norm * norm;
    H[0][0] = y->val * y->val / norm3;
    H[1][0] = -x->val * y->val / norm3;
    H[1][1] = x->val * x->val / norm3;
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    auto norm = hypot(x, y);
    auto norm3 = norm * norm * norm;
    H[0][0] = y * y / norm3;
    H[1][0] = -x * y / norm3;
    H[1][1] = x * x / norm3;
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
    return abs(y);
  } else if (y->is_constant(Scalar(0))) {
    return abs(x);
  }

  // Evaluate constant
  if (x->type() == CONSTANT && y->type() == CONSTANT) {
    return constant_ptr(hypot(x->val, y->val));
  }

  return make_expression_ptr<Hypot2Expression<Scalar>>(x, y);
}

template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y,
                            const ExpressionPtr<Scalar>& z);

/// Derived expression type for hypot() with three arguments.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Hypot3Expression final : Expression<Scalar> {
  /// Ternary operator's first operand.
  ExpressionPtr<Scalar> x;

  /// Ternary operator's second operand.
  ExpressionPtr<Scalar> y;

  /// Ternary operator's third operand.
  ExpressionPtr<Scalar> z;

  /// Constructs a ternary expression (an operator with three arguments).
  ///
  /// @param x Ternary operator's first operand.
  /// @param y Ternary operator's second operand.
  /// @param z Ternary operator's third operand.
  constexpr Hypot3Expression(ExpressionPtr<Scalar> x, ExpressionPtr<Scalar> y,
                             ExpressionPtr<Scalar> z)
      : x{std::move(x)}, y{std::move(y)}, z{std::move(z)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
    func(y.get());
    func(z.get());
  }

  Scalar value() const override {
    using std::hypot;
    return hypot(x->val, y->val, z->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "hypot"; }

  void accumulate_adjoints() const override {
    using std::hypot;
    Scalar norm = hypot(x->val, y->val, z->val);
    x->adj += this->adj * x->val / norm;
    y->adj += this->adj * y->val / norm;
    z->adj += this->adj * z->val / norm;
  }

  void accumulate_adjoints_expr() const override {
    auto norm = hypot(x, y, z);
    x->adj_expr += this->adj_expr * x / norm;
    y->adj_expr += this->adj_expr * y / norm;
    z->adj_expr += this->adj_expr * z / norm;
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::hypot;

    Scalar norm = hypot(x->val, y->val, z->val);
    g[0] = x->val / norm;
    g[1] = y->val / norm;
    g[2] = z->val / norm;
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    auto norm = hypot(x, y, z);
    g[0] = x / norm;
    g[1] = y / norm;
    g[2] = z / norm;
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::hypot;

    Scalar norm = hypot(x->val, y->val, z->val);
    Scalar norm3 = norm * norm * norm;
    H[0][0] = (y->val * y->val + z->val * z->val) / norm3;
    H[1][0] = -x->val * y->val / norm3;
    H[1][1] = (x->val * x->val + z->val * z->val) / norm3;
    H[2][0] = -x->val * z->val / norm3;
    H[2][1] = -y->val * z->val / norm3;
    H[2][2] = (x->val * x->val + y->val * y->val) / norm3;
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    auto norm = hypot(x, y, z);
    auto norm3 = norm * norm * norm;
    H[0][0] = (y * y + z * z) / norm3;
    H[1][0] = -x * y / norm3;
    H[1][1] = (x * x + z * z) / norm3;
    H[2][0] = -x * z / norm3;
    H[2][1] = -y * z / norm3;
    H[2][2] = (x * x + y * y) / norm3;
  }
};

/// hypot() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The x argument.
/// @param y The y argument.
/// @param z The z argument.
template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y,
                            const ExpressionPtr<Scalar>& z) {
  using enum ExpressionType;
  using std::hypot;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return hypot(y, z);
  } else if (y->is_constant(Scalar(0))) {
    return hypot(x, z);
  } else if (z->is_constant(Scalar(0))) {
    return hypot(x, y);
  }

  // Evaluate constant
  if (x->type() == CONSTANT && y->type() == CONSTANT && z->type() == CONSTANT) {
    return constant_ptr(hypot(x->val, y->val, z->val));
  }

  return make_expression_ptr<Hypot3Expression<Scalar>>(x, y, z);
}

/// Derived expression type for if_else().
///
/// Returns t if cond(a, b) is true, otherwise f.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct IfElseExpression final : Expression<Scalar> {
  /// Condition evaluated on a and b.
  bool (*cond)(Scalar a, Scalar b);

  /// The condition's first argument.
  ExpressionPtr<Scalar> a;

  /// The condition's second argument.
  ExpressionPtr<Scalar> b;

  /// Value selected when cond(a, b) is true.
  ExpressionPtr<Scalar> t;

  /// Value selected when cond(a, b) is false.
  ExpressionPtr<Scalar> f;

  /// Constructs an if-else expression.
  ///
  /// @param cond Condition evaluated on a and b.
  /// @param a The condition's first argument.
  /// @param b The condition's second argument.
  /// @param t Value selected when cond(a, b) is true.
  /// @param f Value selected when cond(a, b) is false.
  constexpr IfElseExpression(bool (*cond)(Scalar a, Scalar b),
                             ExpressionPtr<Scalar> a, ExpressionPtr<Scalar> b,
                             ExpressionPtr<Scalar> t, ExpressionPtr<Scalar> f)
      : cond{cond},
        a{std::move(a)},
        b{std::move(b)},
        t{std::move(t)},
        f{std::move(f)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(a.get());
    func(b.get());
    func(t.get());
    func(f.get());
  }

  Scalar value() const override {
    return cond(a->val, b->val) ? t->val : f->val;
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "if-else"; }

  // a and b have zero gradient, so only t and f accumulate adjoints
  void accumulate_adjoints() const override {
    if (cond(a->val, b->val)) {
      t->adj += this->adj;
    } else {
      f->adj += this->adj;
    }
  }

  void accumulate_adjoints_expr() const override {
    t->adj_expr += grad_expr_t();
    f->adj_expr += grad_expr_f();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    // a and b have zero gradient
    bool c = cond(a->val, b->val);
    g[2] = c ? Scalar(1) : Scalar(0);
    g[3] = c ? Scalar(0) : Scalar(1);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    // a and b have zero gradient
    auto zero = constant_ptr(Scalar(0));
    auto one = constant_ptr(Scalar(1));
    g[2] = if_else(cond, a, b, one, zero);
    g[3] = if_else(cond, a, b, zero, one);
  }

 private:
  ExpressionPtr<Scalar> grad_expr_t() const {
    // adjoint if cond(a, b), otherwise 0
    return if_else(cond, a, b, this->adj_expr, constant_ptr(Scalar(0)));
  }

  ExpressionPtr<Scalar> grad_expr_f() const {
    // 0 if cond(a, b), otherwise adjoint
    return if_else(cond, a, b, constant_ptr(Scalar(0)), this->adj_expr);
  }
};

/// Returns t if cond(a, b) is true, otherwise f.
///
/// @tparam Scalar Scalar type.
/// @param cond Condition evaluated on a and b.
/// @param a The condition's first argument.
/// @param b The condition's second argument.
/// @param t Value selected when cond(a, b) is true.
/// @param f Value selected when cond(a, b) is false.
template <typename Scalar>
ExpressionPtr<Scalar> if_else(
    std::type_identity_t<bool (*)(Scalar a, Scalar b)> cond,
    const ExpressionPtr<Scalar>& a, const ExpressionPtr<Scalar>& b,
    const ExpressionPtr<Scalar>& t, const ExpressionPtr<Scalar>& f) {
  using enum ExpressionType;

  // Prune expression
  if (t == f) {
    // Return t, which both branches currently are
    return t;
  } else if (t->type() == CONSTANT && f->type() == CONSTANT &&
             t->val == f->val) {
    // Return t, which both branches currently equal
    return t;
  }

  // Evaluate constant condition
  if (a->type() == CONSTANT && b->type() == CONSTANT) {
    return cond(a->val, b->val) ? t : f;
  }

  return make_expression_ptr<IfElseExpression<Scalar>>(cond, a, b, t, f);
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    g[0] = Scalar(1) / x->val;
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = constant_ptr(Scalar(1)) / x;
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    H[0][0] = Scalar(-1) / (x->val * x->val);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    H[0][0] = constant_ptr(Scalar(-1)) / (x * x);
  }

 private:
  Scalar grad_l() const { return this->adj / x->val; }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr / x; }
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    g[0] = Scalar(1) / (Scalar(std::numbers::ln10) * x->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = constant_ptr(Scalar(1)) /
           (constant_ptr(Scalar(std::numbers::ln10)) * x);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    H[0][0] = Scalar(-1) / (Scalar(std::numbers::ln10) * x->val * x->val);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    H[0][0] = constant_ptr(Scalar(-1)) /
              (constant_ptr(Scalar(std::numbers::ln10)) * x * x);
  }

 private:
  Scalar grad_l() const {
    return this->adj / (Scalar(std::numbers::ln10) * x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr / (constant_ptr(Scalar(std::numbers::ln10)) * x);
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
    a->adj += grad_l();
    b->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    a->adj_expr += grad_expr_l();
    b->adj_expr += grad_expr_r();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    bool c = a->val >= b->val;
    g[0] = c ? Scalar(1) : Scalar(0);
    g[1] = c ? Scalar(0) : Scalar(1);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    auto zero = constant_ptr(Scalar(0));
    auto one = constant_ptr(Scalar(1));
    auto cond = [](Scalar a, Scalar b) { return a >= b; };
    g[0] = if_else(cond, a, b, one, zero);
    g[1] = if_else(cond, a, b, zero, one);
  }

 private:
  Scalar grad_l() const { return a->val >= b->val ? this->adj : Scalar(0); }

  Scalar grad_r() const { return a->val >= b->val ? Scalar(0) : this->adj; }

  ExpressionPtr<Scalar> grad_expr_l() const {
    // adjoint if a >= b, otherwise 0
    return if_else([](Scalar a, Scalar b) { return a >= b; }, a, b,
                   this->adj_expr, constant_ptr(Scalar(0)));
  }

  ExpressionPtr<Scalar> grad_expr_r() const {
    // 0 if a >= b, otherwise adjoint
    return if_else([](Scalar a, Scalar b) { return a >= b; }, a, b,
                   constant_ptr(Scalar(0)), this->adj_expr);
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
    a->adj += grad_l();
    b->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    a->adj_expr += grad_expr_l();
    b->adj_expr += grad_expr_r();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    bool c = a->val <= b->val;
    g[0] = c ? Scalar(1) : Scalar(0);
    g[1] = c ? Scalar(0) : Scalar(1);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    auto zero = constant_ptr(Scalar(0));
    auto one = constant_ptr(Scalar(1));
    auto cond = [](Scalar a, Scalar b) { return a <= b; };
    g[0] = if_else(cond, a, b, one, zero);
    g[1] = if_else(cond, a, b, zero, one);
  }

 private:
  Scalar grad_l() const { return a->val <= b->val ? this->adj : Scalar(0); }

  Scalar grad_r() const { return a->val <= b->val ? Scalar(0) : this->adj; }

  ExpressionPtr<Scalar> grad_expr_l() const {
    // adjoint if a <= b, otherwise 0
    return if_else([](Scalar a, Scalar b) { return a <= b; }, a, b,
                   this->adj_expr, constant_ptr(Scalar(0)));
  }

  ExpressionPtr<Scalar> grad_expr_r() const {
    // 0 if a <= b, otherwise adjoint
    return if_else([](Scalar a, Scalar b) { return a <= b; }, a, b,
                   constant_ptr(Scalar(0)), this->adj_expr);
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
    base->adj += grad_l();
    power->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    base->adj_expr += grad_expr_l();
    power->adj_expr += grad_expr_r();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::log;
    using std::pow;

    g[0] = pow(base->val, power->val - Scalar(1)) * power->val;
    g[1] = pow(base->val, power->val) * log(base->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = pow(base, power - constant_ptr(Scalar(1))) * power;
    g[1] = pow(base, power) * log(base);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::log;
    using std::pow;

    Scalar l = log(base->val);
    H[0][0] = pow(base->val, power->val - Scalar(2)) *
              (power->val - Scalar(1)) * power->val;
    H[1][0] =
        pow(base->val, power->val - Scalar(1)) * (power->val * l + Scalar(1));
    H[1][1] = pow(base->val, power->val) * l * l;
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    auto one = constant_ptr(Scalar(1));
    auto l = log(base);
    H[0][0] =
        pow(base, power - constant_ptr(Scalar(2))) * (power - one) * power;
    H[1][0] = pow(base, power - one) * (power * l + one);
    H[1][1] = pow(base, power) * l * l;
  }

 private:
  Scalar grad_l() const {
    using std::pow;
    return this->adj * pow(base->val, power->val - Scalar(1)) * power->val;
  }

  Scalar grad_r() const {
    using std::log;
    using std::pow;

    return this->adj * pow(base->val, power->val) * log(base->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr * pow(base, power - constant_ptr(Scalar(1))) * power;
  }

  ExpressionPtr<Scalar> grad_expr_r() const {
    return this->adj_expr * pow(base, power) * log(base);
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::cos;
    g[0] = cos(x->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = cos(x);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::sin;
    H[0][0] = -sin(x->val);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    H[0][0] = -sin(x);
  }

 private:
  Scalar grad_l() const {
    using std::cos;
    return this->adj * cos(x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * cos(x); }
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::cosh;
    g[0] = cosh(x->val);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = cosh(x);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::sinh;
    H[0][0] = sinh(x->val);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    H[0][0] = sinh(x);
  }

 private:
  Scalar grad_l() const {
    using std::cosh;
    return this->adj * cosh(x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * cosh(x); }
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::sqrt;
    g[0] = Scalar(1) / (Scalar(2) * sqrt(x->val));
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    g[0] = constant_ptr(Scalar(1)) / (constant_ptr(Scalar(2)) * sqrt(x));
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::sqrt;

    Scalar s = sqrt(x->val);
    H[0][0] = Scalar(-1) / (Scalar(4) * s * s * s);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    auto s = sqrt(x);
    H[0][0] = constant_ptr(Scalar(-1)) / (constant_ptr(Scalar(4)) * s * s * s);
  }

 private:
  Scalar grad_l() const {
    using std::sqrt;
    return this->adj / (Scalar(2) * sqrt(x->val));
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr / (constant_ptr(Scalar(2)) * sqrt(x));
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::cos;

    Scalar c = cos(x->val);
    g[0] = Scalar(1) / (c * c);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    auto c = cos(x);
    g[0] = constant_ptr(Scalar(1)) / (c * c);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::cos;
    using std::tan;

    Scalar c = cos(x->val);
    H[0][0] = Scalar(2) * tan(x->val) / (c * c);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    auto c = cos(x);
    H[0][0] = constant_ptr(Scalar(2)) * tan(x) / (c * c);
  }

 private:
  Scalar grad_l() const {
    using std::cos;

    auto c = cos(x->val);
    return this->adj / (c * c);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    auto c = cos(x);
    return this->adj_expr / (c * c);
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

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

  void grad(ArgArray<std::optional<Scalar>>& g) const override {
    using std::cosh;

    Scalar c = cosh(x->val);
    g[0] = Scalar(1) / (c * c);
  }

  void grad_expr(ArgArray<ExpressionPtr<Scalar>>& g) const override {
    auto c = cosh(x);
    g[0] = constant_ptr(Scalar(1)) / (c * c);
  }

  void hess(ArgArray<ArgArray<std::optional<Scalar>>>& H) const override {
    using std::cosh;
    using std::tanh;

    Scalar c = cosh(x->val);
    H[0][0] = Scalar(-2) * tanh(x->val) / (c * c);
  }

  void hess_expr(ArgArray<ArgArray<ExpressionPtr<Scalar>>>& H) const override {
    auto c = cosh(x);
    H[0][0] = constant_ptr(Scalar(-2)) * tanh(x) / (c * c);
  }

 private:
  Scalar grad_l() const {
    using std::cosh;

    auto c = cosh(x->val);
    return this->adj / (c * c);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    auto c = cosh(x);
    return this->adj_expr / (c * c);
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
