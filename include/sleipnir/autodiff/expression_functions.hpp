// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <numbers>
#include <string_view>
#include <utility>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_type.hpp"

namespace slp::detail {
/// Derived expression type for abs().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct AbsExpression final : NonLeafExpression<AbsExpression, Scalar, Type, 1> {
  using AbsExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "abs";

  static Scalar eval(Scalar x) {
    using std::abs;
    return abs(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    if constexpr (std::same_as<Value, Scalar>) {
      if (x < Scalar(0)) {
        g[0] = Scalar(-1);
      } else if (x > Scalar(0)) {
        g[0] = Scalar(1);
      } else {
        g[0] = Scalar(0);
      }
    } else {
      g[0] = sign(x);
    }
  }
};

/// abs() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> abs(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<AbsExpression, Scalar>(x);
}

/// Derived expression type for acos().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct AcosExpression final
    : NonLeafExpression<AcosExpression, Scalar, Type, 1> {
  using AcosExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "acos";

  static Scalar eval(Scalar x) {
    using std::acos;
    return acos(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    using std::sqrt;
    g[0] = lit<Value>(Scalar(-1)) / sqrt(lit<Value>(Scalar(1)) - x * x);
  }
};

/// acos() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> acos(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<AcosExpression, Scalar>(x);
}

/// Derived expression type for asin().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct AsinExpression final
    : NonLeafExpression<AsinExpression, Scalar, Type, 1> {
  using AsinExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "asin";

  static Scalar eval(Scalar x) {
    using std::asin;
    return asin(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    using std::sqrt;
    g[0] = lit<Value>(Scalar(1)) / sqrt(lit<Value>(Scalar(1)) - x * x);
  }
};

/// asin() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> asin(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<AsinExpression, Scalar>(x);
}

/// Derived expression type for atan().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct AtanExpression final
    : NonLeafExpression<AtanExpression, Scalar, Type, 1> {
  using AtanExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "atan";

  static Scalar eval(Scalar x) {
    using std::atan;
    return atan(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    g[0] = lit<Value>(Scalar(1)) / (lit<Value>(Scalar(1)) + x * x);
  }
};

/// atan() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> atan(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<AtanExpression, Scalar>(x);
}

/// Derived expression type for atan2().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct Atan2Expression final
    : NonLeafExpression<Atan2Expression, Scalar, Type, 2> {
  using Atan2Expression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "atan2";

  static Scalar eval(Scalar y, Scalar x) {
    using std::atan2;
    return atan2(y, x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 2>& g,
                        [[maybe_unused]] const Value& val, const Value& y,
                        const Value& x) {
    Value d = y * y + x * x;
    g[0] = x / d;
    g[1] = -y / d;
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
  return fold_or_make<Atan2Expression, Scalar>(y, x);
}

/// Derived expression type for cbrt().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct CbrtExpression final
    : NonLeafExpression<CbrtExpression, Scalar, Type, 1> {
  using CbrtExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "cbrt";

  static Scalar eval(Scalar x) {
    using std::cbrt;
    return cbrt(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g, const Value& val,
                        [[maybe_unused]] const Value& x) {
    g[0] = lit<Value>(Scalar(1)) / (lit<Value>(Scalar(3)) * val * val);
  }
};

/// cbrt() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cbrt(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<CbrtExpression, Scalar>(x);
}

/// Derived expression type for cos().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct CosExpression final : NonLeafExpression<CosExpression, Scalar, Type, 1> {
  using CosExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "cos";

  static Scalar eval(Scalar x) {
    using std::cos;
    return cos(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    using std::sin;
    g[0] = -sin(x);
  }
};

/// cos() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cos(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<CosExpression, Scalar>(x);
}

/// Derived expression type for cosh().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct CoshExpression final
    : NonLeafExpression<CoshExpression, Scalar, Type, 1> {
  using CoshExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "cosh";

  static Scalar eval(Scalar x) {
    using std::cosh;
    return cosh(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    using std::sinh;
    g[0] = sinh(x);
  }
};

/// cosh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cosh(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<CoshExpression, Scalar>(x);
}

/// Derived expression type for erf().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct ErfExpression final : NonLeafExpression<ErfExpression, Scalar, Type, 1> {
  using ErfExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "erf";

  static Scalar eval(Scalar x) {
    using std::erf;
    return erf(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    using std::exp;
    g[0] = lit<Value>(Scalar(2.0 * std::numbers::inv_sqrtpi)) * exp(-x * x);
  }
};

/// erf() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> erf(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<ErfExpression, Scalar>(x);
}

/// Derived expression type for exp().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct ExpExpression final : NonLeafExpression<ExpExpression, Scalar, Type, 1> {
  using ExpExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "exp";

  static Scalar eval(Scalar x) {
    using std::exp;
    return exp(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g, const Value& val,
                        [[maybe_unused]] const Value& x) {
    g[0] = val;
  }
};

/// exp() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> exp(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<ExpExpression, Scalar>(x);
}

template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y);

/// Derived expression type for hypot() with two arguments.
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct Hypot2Expression final
    : NonLeafExpression<Hypot2Expression, Scalar, Type, 2> {
  using Hypot2Expression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "hypot";

  static Scalar eval(Scalar x, Scalar y) {
    using std::hypot;
    return hypot(x, y);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 2>& g, const Value& val,
                        const Value& x, const Value& y) {
    g[0] = x / val;
    g[1] = y / val;
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
  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return abs(y);
  } else if (y->is_constant(Scalar(0))) {
    return abs(x);
  }

  return fold_or_make<Hypot2Expression, Scalar>(x, y);
}

template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y,
                            const ExpressionPtr<Scalar>& z);

/// Derived expression type for hypot() with three arguments.
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct Hypot3Expression final
    : NonLeafExpression<Hypot3Expression, Scalar, Type, 3> {
  using Hypot3Expression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "hypot";

  static Scalar eval(Scalar x, Scalar y, Scalar z) {
    using std::hypot;
    return hypot(x, y, z);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 3>& g, const Value& val,
                        const Value& x, const Value& y, const Value& z) {
    g[0] = x / val;
    g[1] = y / val;
    g[2] = z / val;
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
  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return hypot(y, z);
  } else if (y->is_constant(Scalar(0))) {
    return hypot(x, z);
  } else if (z->is_constant(Scalar(0))) {
    return hypot(x, y);
  }

  return fold_or_make<Hypot3Expression, Scalar>(x, y, z);
}

/// The condition if_else() checks on its argument c.
enum class IfElseCondition : uint8_t {
  /// c = 0.
  ZERO,
  /// c ≥ 0.
  NONNEGATIVE,
  /// c > 0.
  POSITIVE
};

/// Derived expression type for if_else().
///
/// Returns t if c satisfies the condition, otherwise f.
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct IfElseExpression final
    : NonLeafExpression<IfElseExpression, Scalar, Type, 3> {
  /// The expression's name.
  static constexpr std::string_view NAME = "if-else";

  /// The condition checked on c.
  IfElseCondition cond;

  /// Constructs an if-else expression.
  ///
  /// @param cond The condition checked on c.
  /// @param c The condition's argument.
  /// @param t Value selected when c satisfies the condition.
  /// @param f Value selected when c violates the condition.
  constexpr IfElseExpression(IfElseCondition cond, ExpressionPtr<Scalar> c,
                             ExpressionPtr<Scalar> t, ExpressionPtr<Scalar> f)
      : IfElseExpression::NonLeafExpression{std::move(c), std::move(t),
                                            std::move(f)},
        cond{cond} {}

  /// Returns true if c satisfies the condition.
  ///
  /// @param cond The condition checked on c.
  /// @param c The condition's argument.
  /// @return True if c satisfies the condition.
  static bool satisfied(IfElseCondition cond, Scalar c) {
    using enum IfElseCondition;

    switch (cond) {
      case ZERO:
        return c == Scalar(0);
      case NONNEGATIVE:
        return c >= Scalar(0);
      case POSITIVE:
        return c > Scalar(0);
      default:
        std::unreachable();
    }
  }

  Scalar eval(Scalar c, Scalar t, Scalar f) const {
    return satisfied(cond, c) ? t : f;
  }

  template <typename Partial, typename Value>
  void eval_grad(std::array<Partial, 3>& g, [[maybe_unused]] const Value& val,
                 const Value& c, [[maybe_unused]] const Value& t,
                 [[maybe_unused]] const Value& f) const {
    // c has zero gradient
    if constexpr (std::same_as<Value, Scalar>) {
      bool s = satisfied(cond, c);
      g[1] = s ? Scalar(1) : Scalar(0);
      g[2] = s ? Scalar(0) : Scalar(1);
    } else {
      auto zero = constant_ptr(Scalar(0));
      auto one = constant_ptr(Scalar(1));
      g[1] = if_else(cond, c, one, zero);
      g[2] = if_else(cond, c, zero, one);
    }
  }
};

/// Returns t if c satisfies the condition, otherwise f.
///
/// @tparam Scalar Scalar type.
/// @param cond The condition checked on c.
/// @param c The condition's argument.
/// @param t Value selected when c satisfies the condition.
/// @param f Value selected when c violates the condition.
template <typename Scalar>
ExpressionPtr<Scalar> if_else(IfElseCondition cond,
                              const ExpressionPtr<Scalar>& c,
                              const ExpressionPtr<Scalar>& t,
                              const ExpressionPtr<Scalar>& f) {
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
  if (c->type() == CONSTANT) {
    return IfElseExpression<Scalar, NONLINEAR>::satisfied(cond, c->val) ? t : f;
  }

  if (c->type() <= PARAMETER && t->type() <= PARAMETER &&
      f->type() <= PARAMETER) {
    return make_expression_ptr<IfElseExpression<Scalar, PARAMETER>>(cond, c, t,
                                                                    f);
  } else {
    return make_expression_ptr<IfElseExpression<Scalar, NONLINEAR>>(cond, c, t,
                                                                    f);
  }
}

/// Derived expression type for log().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct LogExpression final : NonLeafExpression<LogExpression, Scalar, Type, 1> {
  using LogExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "log";

  static Scalar eval(Scalar x) {
    using std::log;
    return log(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    g[0] = lit<Value>(Scalar(1)) / x;
  }
};

/// log() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> log(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<LogExpression, Scalar>(x);
}

/// Derived expression type for log10().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct Log10Expression final
    : NonLeafExpression<Log10Expression, Scalar, Type, 1> {
  using Log10Expression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "log10";

  static Scalar eval(Scalar x) {
    using std::log10;
    return log10(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    g[0] = lit<Value>(Scalar(1)) / (lit<Value>(Scalar(std::numbers::ln10)) * x);
  }
};

/// log10() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> log10(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<Log10Expression, Scalar>(x);
}

/// Derived expression type for max().
///
/// Returns the greater of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct MaxExpression final : NonLeafExpression<MaxExpression, Scalar, Type, 2> {
  using MaxExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "max";

  static Scalar eval(Scalar a, Scalar b) {
    using std::max;
    return max(a, b);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 2>& g,
                        [[maybe_unused]] const Value& val, const Value& a,
                        const Value& b) {
    if constexpr (std::same_as<Value, Scalar>) {
      bool c = a >= b;
      g[0] = c ? Scalar(1) : Scalar(0);
      g[1] = c ? Scalar(0) : Scalar(1);
    } else {
      auto zero = constant_ptr(Scalar(0));
      auto one = constant_ptr(Scalar(1));
      auto c = a - b;
      g[0] = if_else(IfElseCondition::NONNEGATIVE, c, one, zero);
      g[1] = if_else(IfElseCondition::NONNEGATIVE, c, zero, one);
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
  return fold_or_make<MaxExpression, Scalar>(a, b);
}

/// Derived expression type for min().
///
/// Returns the lesser of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct MinExpression final : NonLeafExpression<MinExpression, Scalar, Type, 2> {
  using MinExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "min";

  static Scalar eval(Scalar a, Scalar b) {
    using std::min;
    return min(a, b);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 2>& g,
                        [[maybe_unused]] const Value& val, const Value& a,
                        const Value& b) {
    if constexpr (std::same_as<Value, Scalar>) {
      bool c = a <= b;
      g[0] = c ? Scalar(1) : Scalar(0);
      g[1] = c ? Scalar(0) : Scalar(1);
    } else {
      auto zero = constant_ptr(Scalar(0));
      auto one = constant_ptr(Scalar(1));
      auto c = b - a;
      g[0] = if_else(IfElseCondition::NONNEGATIVE, c, one, zero);
      g[1] = if_else(IfElseCondition::NONNEGATIVE, c, zero, one);
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
  return fold_or_make<MinExpression, Scalar>(a, b);
}

template <typename Scalar>
ExpressionPtr<Scalar> pow(const ExpressionPtr<Scalar>& base,
                          const ExpressionPtr<Scalar>& power);

template <typename Scalar>
ExpressionPtr<Scalar> sqrt(const ExpressionPtr<Scalar>& x);

/// Derived expression type for pow().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct PowExpression final : NonLeafExpression<PowExpression, Scalar, Type, 2> {
  using PowExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "pow";

  static Scalar eval(Scalar base, Scalar power) {
    using std::pow;
    return pow(base, power);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 2>& g, const Value& val,
                        const Value& base, const Value& power) {
    using std::log;
    using std::pow;

    g[0] = pow(base, power - lit<Value>(Scalar(1))) * power;
    g[1] = val * log(base);
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
    auto type = base->type() == PARAMETER ? PARAMETER
                : base->type() == LINEAR  ? QUADRATIC
                                          : NONLINEAR;
    return make_typed_expression_ptr<MultExpression, Scalar>(type, base, base);
  } else if (power->is_constant(Scalar(0.5))) {
    return sqrt(base);
  }

  if (base->type() <= PARAMETER && power->type() <= PARAMETER) {
    return make_expression_ptr<PowExpression<Scalar, PARAMETER>>(base, power);
  } else {
    return make_expression_ptr<PowExpression<Scalar, NONLINEAR>>(base, power);
  }
}

/// Derived expression type for sign().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct SignExpression final
    : NonLeafExpression<SignExpression, Scalar, Type, 1> {
  using SignExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "sign";

  static Scalar eval(Scalar x) {
    if (x < Scalar(0)) {
      return Scalar(-1);
    } else if (x == Scalar(0)) {
      return Scalar(0);
    } else {
      return Scalar(1);
    }
  }
};

/// sign() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sign(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<SignExpression, Scalar>(x);
}

/// Derived expression type for sin().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct SinExpression final : NonLeafExpression<SinExpression, Scalar, Type, 1> {
  using SinExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "sin";

  static Scalar eval(Scalar x) {
    using std::sin;
    return sin(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    using std::cos;
    g[0] = cos(x);
  }
};

/// sin() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sin(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<SinExpression, Scalar>(x);
}

/// Derived expression type for sinh().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct SinhExpression final
    : NonLeafExpression<SinhExpression, Scalar, Type, 1> {
  using SinhExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "sinh";

  static Scalar eval(Scalar x) {
    using std::sinh;
    return sinh(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    using std::cosh;
    g[0] = cosh(x);
  }
};

/// sinh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sinh(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<SinhExpression, Scalar>(x);
}

/// Derived expression type for sqrt().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct SqrtExpression final
    : NonLeafExpression<SqrtExpression, Scalar, Type, 1> {
  using SqrtExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "sqrt";

  static Scalar eval(Scalar x) {
    using std::sqrt;
    return sqrt(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g, const Value& val,
                        [[maybe_unused]] const Value& x) {
    g[0] = lit<Value>(Scalar(1)) / (lit<Value>(Scalar(2)) * val);
  }
};

/// sqrt() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sqrt(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<SqrtExpression, Scalar>(x);
}

/// Derived expression type for tan().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct TanExpression final : NonLeafExpression<TanExpression, Scalar, Type, 1> {
  using TanExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "tan";

  static Scalar eval(Scalar x) {
    using std::tan;
    return tan(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    using std::cos;

    Value c = cos(x);
    g[0] = lit<Value>(Scalar(1)) / (c * c);
  }
};

/// tan() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> tan(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<TanExpression, Scalar>(x);
}

/// Derived expression type for tanh().
///
/// @tparam Scalar Scalar type.
/// @tparam Type Expression type.
template <typename Scalar, ExpressionType Type>
struct TanhExpression final
    : NonLeafExpression<TanhExpression, Scalar, Type, 1> {
  using TanhExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "tanh";

  static Scalar eval(Scalar x) {
    using std::tanh;
    return tanh(x);
  }

  template <typename Partial, typename Value>
  static void eval_grad(std::array<Partial, 1>& g,
                        [[maybe_unused]] const Value& val, const Value& x) {
    using std::cosh;

    Value c = cosh(x);
    g[0] = lit<Value>(Scalar(1)) / (c * c);
  }
};

/// tanh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> tanh(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<TanhExpression, Scalar>(x);
}

}  // namespace slp::detail
