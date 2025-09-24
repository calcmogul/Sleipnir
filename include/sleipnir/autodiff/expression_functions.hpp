// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <numbers>
#include <string_view>
#include <type_traits>
#include <utility>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_type.hpp"

namespace slp::detail {
/// Derived expression type for abs().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AbsExpression final
    : NonLeafExpression<AbsExpression<Scalar>, Scalar, ExpressionOp::ABS, 1> {
  using AbsExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "abs";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::abs;
    return abs(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    if constexpr (std::same_as<U, Scalar>) {
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
  return fold_or_make<AbsExpression<Scalar>>(x);
}

/// Derived expression type for acos().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AcosExpression final
    : NonLeafExpression<AcosExpression<Scalar>, Scalar, ExpressionOp::ACOS, 1> {
  using AcosExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "acos";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::acos;
    return acos(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::sqrt;
    g[0] = lit<U>(Scalar(-1)) / sqrt(lit<U>(Scalar(1)) - x * x);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::sqrt;

    U s = sqrt(lit<U>(Scalar(1)) - x * x);
    H[0][0] = -x / (s * s * s);
  }
};

/// acos() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> acos(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<AcosExpression<Scalar>>(x);
}

/// Derived expression type for asin().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AsinExpression final
    : NonLeafExpression<AsinExpression<Scalar>, Scalar, ExpressionOp::ASIN, 1> {
  using AsinExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "asin";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::asin;
    return asin(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::sqrt;
    g[0] = lit<U>(Scalar(1)) / sqrt(lit<U>(Scalar(1)) - x * x);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::sqrt;

    U s = sqrt(lit<U>(Scalar(1)) - x * x);
    H[0][0] = x / (s * s * s);
  }
};

/// asin() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> asin(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<AsinExpression<Scalar>>(x);
}

/// Derived expression type for atan().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AtanExpression final
    : NonLeafExpression<AtanExpression<Scalar>, Scalar, ExpressionOp::ATAN, 1> {
  using AtanExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "atan";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::atan;
    return atan(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    g[0] = lit<U>(Scalar(1)) / (lit<U>(Scalar(1)) + x * x);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, [[maybe_unused]] const U& val,
                        const U& x) {
    U t = lit<U>(Scalar(1)) + x * x;
    H[0][0] = lit<U>(Scalar(-2)) * x / (t * t);
  }
};

/// atan() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> atan(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<AtanExpression<Scalar>>(x);
}

/// Derived expression type for atan2().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Atan2Expression final
    : NonLeafExpression<Atan2Expression<Scalar>, Scalar, ExpressionOp::ATAN2,
                        2> {
  using Atan2Expression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "atan2";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param y The value of y.
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar y, Scalar x) {
    using std::atan2;
    return atan2(y, x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& y, const U& x) {
    U d = y * y + x * x;
    g[0] = x / d;
    g[1] = -y / d;
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, [[maybe_unused]] const U& val,
                        const U& y, const U& x) {
    U y2 = y * y;
    U x2 = x * x;
    U d = y2 + x2;
    U d2 = d * d;
    H[0][0] = lit<U>(Scalar(-2)) * x * y / d2;
    H[1][0] = (y2 - x2) / d2;
    H[1][1] = lit<U>(Scalar(2)) * x * y / d2;
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
  return fold_or_make<Atan2Expression<Scalar>>(y, x);
}

/// Derived expression type for cbrt().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct CbrtExpression final
    : NonLeafExpression<CbrtExpression<Scalar>, Scalar, ExpressionOp::CBRT, 1> {
  using CbrtExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "cbrt";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::cbrt;
    return cbrt(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, const U& val,
                        [[maybe_unused]] const U& x) {
    g[0] = lit<U>(Scalar(1)) / (lit<U>(Scalar(3)) * val * val);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val, const U& x) {
    H[0][0] = lit<U>(Scalar(-2)) / (lit<U>(Scalar(9)) * x * val * val);
  }
};

/// cbrt() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cbrt(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<CbrtExpression<Scalar>>(x);
}

/// Derived expression type for cos().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct CosExpression final
    : NonLeafExpression<CosExpression<Scalar>, Scalar, ExpressionOp::COS, 1> {
  using CosExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "cos";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::cos;
    return cos(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::sin;
    g[0] = -sin(x);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val,
                        [[maybe_unused]] const U& x) {
    H[0][0] = -val;
  }
};

/// cos() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cos(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<CosExpression<Scalar>>(x);
}

/// Derived expression type for cosh().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct CoshExpression final
    : NonLeafExpression<CoshExpression<Scalar>, Scalar, ExpressionOp::COSH, 1> {
  using CoshExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "cosh";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::cosh;
    return cosh(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::sinh;
    g[0] = sinh(x);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val,
                        [[maybe_unused]] const U& x) {
    H[0][0] = val;
  }
};

/// cosh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cosh(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<CoshExpression<Scalar>>(x);
}

/// Derived expression type for erf().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct ErfExpression final
    : NonLeafExpression<ErfExpression<Scalar>, Scalar, ExpressionOp::ERF, 1> {
  using ErfExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "erf";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::erf;
    return erf(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::exp;
    g[0] = lit<U>(Scalar(2.0 * std::numbers::inv_sqrtpi)) * exp(-x * x);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::exp;
    H[0][0] = lit<U>(Scalar(-4.0 * std::numbers::inv_sqrtpi)) * x * exp(-x * x);
  }
};

/// erf() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> erf(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<ErfExpression<Scalar>>(x);
}

/// Derived expression type for exp().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct ExpExpression final
    : NonLeafExpression<ExpExpression<Scalar>, Scalar, ExpressionOp::EXP, 1> {
  using ExpExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "exp";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::exp;
    return exp(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, const U& val,
                        [[maybe_unused]] const U& x) {
    g[0] = val;
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val,
                        [[maybe_unused]] const U& x) {
    H[0][0] = val;
  }
};

/// exp() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> exp(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<ExpExpression<Scalar>>(x);
}

template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y);

/// Derived expression type for hypot() with two arguments.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Hypot2Expression final
    : NonLeafExpression<Hypot2Expression<Scalar>, Scalar, ExpressionOp::HYPOT2,
                        2> {
  using Hypot2Expression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "hypot";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @param y The value of y.
  /// @return The expression's value.
  static Scalar eval(Scalar x, Scalar y) {
    using std::hypot;
    return hypot(x, y);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, const U& val, const U& x, const U& y) {
    g[0] = x / val;
    g[1] = y / val;
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val, const U& x,
                        const U& y) {
    U norm3 = val * val * val;
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
  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return abs(y);
  } else if (y->is_constant(Scalar(0))) {
    return abs(x);
  }

  return fold_or_make<Hypot2Expression<Scalar>>(x, y);
}

template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y,
                            const ExpressionPtr<Scalar>& z);

/// Derived expression type for hypot() with three arguments.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Hypot3Expression final
    : NonLeafExpression<Hypot3Expression<Scalar>, Scalar, ExpressionOp::HYPOT3,
                        3> {
  using Hypot3Expression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "hypot";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @param y The value of y.
  /// @param z The value of z.
  /// @return The expression's value.
  static Scalar eval(Scalar x, Scalar y, Scalar z) {
    using std::hypot;
    return hypot(x, y, z);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, const U& val, const U& x, const U& y,
                        const U& z) {
    g[0] = x / val;
    g[1] = y / val;
    g[2] = z / val;
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val, const U& x,
                        const U& y, const U& z) {
    U norm3 = val * val * val;
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
  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return hypot(y, z);
  } else if (y->is_constant(Scalar(0))) {
    return hypot(x, z);
  } else if (z->is_constant(Scalar(0))) {
    return hypot(x, y);
  }

  return fold_or_make<Hypot3Expression<Scalar>>(x, y, z);
}

/// Derived expression type for if_else().
///
/// Returns t if cond(a, b) is true, otherwise f.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct IfElseExpression final
    : NonLeafExpression<IfElseExpression<Scalar>, Scalar, ExpressionOp::IF_ELSE,
                        4> {
  /// The expression's name.
  static constexpr std::string_view NAME = "if-else";

  /// Condition evaluated on a and b.
  bool (*cond)(Scalar a, Scalar b);

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
      : IfElseExpression::NonLeafExpression{std::move(a), std::move(b),
                                            std::move(t), std::move(f)},
        cond{cond} {}

  /// Returns the expression's value at the given argument values.
  ///
  /// @param a The value of a.
  /// @param b The value of b.
  /// @param t The value of t.
  /// @param f The value of f.
  /// @return The expression's value.
  Scalar eval(Scalar a, Scalar b, Scalar t, Scalar f) const {
    return cond(a, b) ? t : f;
  }

  template <typename G, typename U>
  void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val, const U& a,
                 const U& b, [[maybe_unused]] const U& t,
                 [[maybe_unused]] const U& f) const {
    // a and b have zero gradient
    if constexpr (std::same_as<U, Scalar>) {
      bool c = cond(a, b);
      g[2] = c ? Scalar(1) : Scalar(0);
      g[3] = c ? Scalar(0) : Scalar(1);
    } else {
      auto zero = constant_ptr(Scalar(0));
      auto one = constant_ptr(Scalar(1));
      g[2] = if_else(cond, a, b, one, zero);
      g[3] = if_else(cond, a, b, zero, one);
    }
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
struct LogExpression final
    : NonLeafExpression<LogExpression<Scalar>, Scalar, ExpressionOp::LOG, 1> {
  using LogExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "log";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::log;
    return log(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    g[0] = lit<U>(Scalar(1)) / x;
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, [[maybe_unused]] const U& val,
                        const U& x) {
    H[0][0] = lit<U>(Scalar(-1)) / (x * x);
  }
};

/// log() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> log(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<LogExpression<Scalar>>(x);
}

/// Derived expression type for log10().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Log10Expression final
    : NonLeafExpression<Log10Expression<Scalar>, Scalar, ExpressionOp::LOG10,
                        1> {
  using Log10Expression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "log10";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::log10;
    return log10(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    g[0] = lit<U>(Scalar(1)) / (lit<U>(Scalar(std::numbers::ln10)) * x);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, [[maybe_unused]] const U& val,
                        const U& x) {
    H[0][0] = lit<U>(Scalar(-1)) / (lit<U>(Scalar(std::numbers::ln10)) * x * x);
  }
};

/// log10() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> log10(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<Log10Expression<Scalar>>(x);
}

/// Derived expression type for max().
///
/// Returns the greater of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct MaxExpression final
    : NonLeafExpression<MaxExpression<Scalar>, Scalar, ExpressionOp::MAX, 2> {
  using MaxExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "max";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param a The value of a.
  /// @param b The value of b.
  /// @return The expression's value.
  static Scalar eval(Scalar a, Scalar b) {
    using std::max;
    return max(a, b);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& a, const U& b) {
    if constexpr (std::same_as<U, Scalar>) {
      bool c = a >= b;
      g[0] = c ? Scalar(1) : Scalar(0);
      g[1] = c ? Scalar(0) : Scalar(1);
    } else {
      auto zero = constant_ptr(Scalar(0));
      auto one = constant_ptr(Scalar(1));
      auto cond = [](Scalar a, Scalar b) { return a >= b; };
      g[0] = if_else(cond, a, b, one, zero);
      g[1] = if_else(cond, a, b, zero, one);
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
  return fold_or_make<MaxExpression<Scalar>>(a, b);
}

/// Derived expression type for min().
///
/// Returns the lesser of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct MinExpression final
    : NonLeafExpression<MinExpression<Scalar>, Scalar, ExpressionOp::MIN, 2> {
  using MinExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "min";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param a The value of a.
  /// @param b The value of b.
  /// @return The expression's value.
  static Scalar eval(Scalar a, Scalar b) {
    using std::min;
    return min(a, b);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& a, const U& b) {
    if constexpr (std::same_as<U, Scalar>) {
      bool c = a <= b;
      g[0] = c ? Scalar(1) : Scalar(0);
      g[1] = c ? Scalar(0) : Scalar(1);
    } else {
      auto zero = constant_ptr(Scalar(0));
      auto one = constant_ptr(Scalar(1));
      auto cond = [](Scalar a, Scalar b) { return a <= b; };
      g[0] = if_else(cond, a, b, one, zero);
      g[1] = if_else(cond, a, b, zero, one);
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
  return fold_or_make<MinExpression<Scalar>>(a, b);
}

template <typename Scalar>
ExpressionPtr<Scalar> pow(const ExpressionPtr<Scalar>& base,
                          const ExpressionPtr<Scalar>& power);

template <typename Scalar>
ExpressionPtr<Scalar> sqrt(const ExpressionPtr<Scalar>& x);

/// Derived expression type for pow().
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct PowExpression final : NonLeafExpression<PowExpression<Scalar, T>, Scalar,
                                               ExpressionOp::POW, 2, T> {
  using PowExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "pow";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param base The value of base.
  /// @param power The value of power.
  /// @return The expression's value.
  static Scalar eval(Scalar base, Scalar power) {
    using std::pow;
    return pow(base, power);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, const U& val, const U& base,
                        const U& power) {
    using std::log;
    using std::pow;

    g[0] = pow(base, power - lit<U>(Scalar(1))) * power;
    g[1] = val * log(base);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val, const U& base,
                        const U& power) {
    using std::log;
    using std::pow;

    U l = log(base);
    H[0][0] = pow(base, power - lit<U>(Scalar(2))) *
              (power - lit<U>(Scalar(1))) * power;
    H[1][0] =
        pow(base, power - lit<U>(Scalar(1))) * (power * l + lit<U>(Scalar(1)));
    H[1][1] = val * l * l;
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
    auto type = base->type() == LINEAR ? QUADRATIC : NONLINEAR;
    return make_typed_expression_ptr<MultExpression, Scalar>(type, base, base);
  } else if (power->is_constant(Scalar(0.5))) {
    return sqrt(base);
  }

  return make_expression_ptr<PowExpression<Scalar, NONLINEAR>>(base, power);
}

/// Derived expression type for sign().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SignExpression final
    : NonLeafExpression<SignExpression<Scalar>, Scalar, ExpressionOp::SIGN, 1> {
  using SignExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "sign";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
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
  return fold_or_make<SignExpression<Scalar>>(x);
}

/// Derived expression type for sin().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SinExpression final
    : NonLeafExpression<SinExpression<Scalar>, Scalar, ExpressionOp::SIN, 1> {
  using SinExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "sin";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::sin;
    return sin(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::cos;
    g[0] = cos(x);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val,
                        [[maybe_unused]] const U& x) {
    H[0][0] = -val;
  }
};

/// sin() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sin(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<SinExpression<Scalar>>(x);
}

/// Derived expression type for sinh().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SinhExpression final
    : NonLeafExpression<SinhExpression<Scalar>, Scalar, ExpressionOp::SINH, 1> {
  using SinhExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "sinh";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::sinh;
    return sinh(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::cosh;
    g[0] = cosh(x);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val,
                        [[maybe_unused]] const U& x) {
    H[0][0] = val;
  }
};

/// sinh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sinh(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<SinhExpression<Scalar>>(x);
}

/// Derived expression type for sqrt().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SqrtExpression final
    : NonLeafExpression<SqrtExpression<Scalar>, Scalar, ExpressionOp::SQRT, 1> {
  using SqrtExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "sqrt";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::sqrt;
    return sqrt(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, const U& val,
                        [[maybe_unused]] const U& x) {
    g[0] = lit<U>(Scalar(1)) / (lit<U>(Scalar(2)) * val);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val,
                        [[maybe_unused]] const U& x) {
    H[0][0] = lit<U>(Scalar(-1)) / (lit<U>(Scalar(4)) * val * val * val);
  }
};

/// sqrt() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sqrt(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<SqrtExpression<Scalar>>(x);
}

/// Derived expression type for tan().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct TanExpression final
    : NonLeafExpression<TanExpression<Scalar>, Scalar, ExpressionOp::TAN, 1> {
  using TanExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "tan";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::tan;
    return tan(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::cos;

    U c = cos(x);
    g[0] = lit<U>(Scalar(1)) / (c * c);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val, const U& x) {
    using std::cos;

    U c = cos(x);
    H[0][0] = lit<U>(Scalar(2)) * val / (c * c);
  }
};

/// tan() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> tan(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<TanExpression<Scalar>>(x);
}

/// Derived expression type for tanh().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct TanhExpression final
    : NonLeafExpression<TanhExpression<Scalar>, Scalar, ExpressionOp::TANH, 1> {
  using TanhExpression::NonLeafExpression::NonLeafExpression;

  /// The expression's name.
  static constexpr std::string_view NAME = "tanh";

  /// Returns the expression's value at the given argument values.
  ///
  /// @param x The value of x.
  /// @return The expression's value.
  static Scalar eval(Scalar x) {
    using std::tanh;
    return tanh(x);
  }

  template <typename G, typename U>
  static void eval_grad(ArgArray<G>& g, [[maybe_unused]] const U& val,
                        const U& x) {
    using std::cosh;

    U c = cosh(x);
    g[0] = lit<U>(Scalar(1)) / (c * c);
  }

  template <typename G, typename U>
  static void eval_hess(ArgArray<ArgArray<G>>& H, const U& val, const U& x) {
    using std::cosh;

    U c = cosh(x);
    H[0][0] = lit<U>(Scalar(-2)) * val / (c * c);
  }
};

/// tanh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> tanh(const ExpressionPtr<Scalar>& x) {
  return fold_or_make<TanhExpression<Scalar>>(x);
}

}  // namespace slp::detail
