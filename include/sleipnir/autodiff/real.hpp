// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <ranges>
#include <source_location>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_functions.hpp"
#include "sleipnir/autodiff/expression_graph.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/sleipnir_base.hpp"
#include "sleipnir/util/assert.hpp"
#include "sleipnir/util/concepts.hpp"

#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
#include "sleipnir/util/print.hpp"
#endif

namespace slp {

// Forward declarations for friend declarations in Real

template <typename Scalar>
class RealMatrix;

template <typename Scalar>
struct EqualityConstraints;

template <typename Scalar>
struct InequalityConstraints;

template <typename Scalar>
class Real;

namespace detail {

enum class ExpressionRole : uint8_t;

template <typename Scalar>
std::vector<std::string_view> replaceable_nonsmooth_functions(
    const Real<Scalar>& x, ExpressionRole role);

template <typename Scalar>
RealMatrix<Scalar> gradient_tree(
    const detail::ExpressionGraph<Scalar>& top_list,
    const RealMatrix<Scalar>& wrt);

}  // namespace detail

template <typename Scalar, int UpLo = Eigen::Lower | Eigen::Upper>
  requires(UpLo == Eigen::Lower) || (UpLo == (Eigen::Lower | Eigen::Upper))
class Hessian;

template <typename Scalar>
class Jacobian;

/// A real-valued autodiff expression.
///
/// @tparam Scalar_ Scalar type.
template <typename Scalar_>
class Real : public SleipnirBase {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a linear Real with a value of zero.
  Real() = default;

  /// Constructs an empty Real.
  explicit constexpr Real(std::nullptr_t) : expr{nullptr} {}

  /// Constructs a Real from a scalar type.
  ///
  /// @param value The value of the Real.
  // NOLINTNEXTLINE (google-explicit-constructor)
  Real(Scalar value)
    requires(!MatrixLike<Scalar>)
      : expr{detail::make_expression_ptr<detail::ConstantExpression<Scalar>>(
            value)} {}

  /// Constructs a Real from a scalar type.
  ///
  /// @param value The value of the Real.
  // NOLINTNEXTLINE (google-explicit-constructor)
  Real(SleipnirMatrixLike<Scalar> auto value) : expr{value[0, 0].expr} {
    slp_assert(value.rows() == 1 && value.cols() == 1);
  }

  /// Constructs a Real from a floating-point type.
  ///
  /// @param value The value of the Real.
  // NOLINTNEXTLINE (google-explicit-constructor)
  Real(std::floating_point auto value)
      : expr{detail::make_expression_ptr<detail::ConstantExpression<Scalar>>(
            Scalar(value))} {}

  /// Constructs a Real from an integral type.
  ///
  /// @param value The value of the Real.
  // NOLINTNEXTLINE (google-explicit-constructor)
  Real(std::integral auto value)
      : expr{detail::make_expression_ptr<detail::ConstantExpression<Scalar>>(
            Scalar(value))} {}

  /// Constructs a Real pointing to the specified expression.
  ///
  /// @param expr The autodiff variable.
  explicit Real(const detail::ExpressionPtr<Scalar>& expr) : expr{expr} {}

  /// Constructs a Real pointing to the specified expression.
  ///
  /// @param expr The autodiff variable.
  explicit constexpr Real(detail::ExpressionPtr<Scalar>&& expr)
      : expr{std::move(expr)} {}

  /// Assignment operator for scalar.
  ///
  /// @param value The value of the Real.
  /// @return This variable.
  Real<Scalar>& operator=(ScalarLike auto value) {
    expr =
        detail::make_expression_ptr<detail::ConstantExpression<Scalar>>(value);
    m_graph_initialized = false;

    return *this;
  }

  /// Sets Real's internal value.
  ///
  /// @param value The value of the Real.
  void set_value(Scalar value) {
#ifndef SLEIPNIR_DISABLE_DIAGNOSTICS
    if (!expr->is_leaf) {
      auto location = std::source_location::current();
      slp::println(
          stderr,
          "WARNING: {}:{}: {}: Modified the value of a dependent variable",
          location.file_name(), location.line(), location.function_name());
    }
#endif
    expr->val = Scalar(value);
  }

  /// Returns the value of this variable.
  ///
  /// @return The value of this variable.
  Scalar value() {
    if (!m_graph_initialized) {
      m_graph = detail::topological_sort(expr);
      m_graph_initialized = true;
    }
    detail::update_values(m_graph);

    return Scalar(expr->val);
  }

  /// Returns the type of this expression (constant, linear, quadratic, or
  /// nonlinear).
  ///
  /// @return The type of this expression.
  ExpressionType type() const { return expr->type(); }

  /// Real-scalar multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of multiplication.
  template <ScalarLike LHS, SleipnirScalarLike<Scalar> RHS>
  friend Real<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
    return Real{Real<Scalar>{lhs}.expr * rhs.expr};
  }

  /// Real-scalar multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of multiplication.
  template <SleipnirScalarLike<Scalar> LHS, ScalarLike RHS>
  friend Real<Scalar> operator*(const LHS& lhs, const RHS& rhs) {
    return Real{lhs.expr * Real<Scalar>{rhs}.expr};
  }

  /// Real-scalar multiplication operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of multiplication.
  friend Real<Scalar> operator*(const Real<Scalar>& lhs,
                                const Real<Scalar>& rhs) {
    return Real{lhs.expr * rhs.expr};
  }

  /// Real-Real compound multiplication operator.
  ///
  /// @param rhs Operator right-hand side.
  /// @return Result of multiplication.
  Real<Scalar>& operator*=(const Real<Scalar>& rhs) {
    *this = *this * rhs;
    return *this;
  }

  /// Real-Real division operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of division.
  friend Real<Scalar> operator/(const Real<Scalar>& lhs,
                                const Real<Scalar>& rhs) {
    return Real{lhs.expr / rhs.expr};
  }

  /// Real-Real compound division operator.
  ///
  /// @param rhs Operator right-hand side.
  /// @return Result of division.
  Real<Scalar>& operator/=(const Real<Scalar>& rhs) {
    *this = *this / rhs;
    return *this;
  }

  /// Real-Real addition operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of addition.
  friend Real<Scalar> operator+(const Real<Scalar>& lhs,
                                const Real<Scalar>& rhs) {
    return Real{lhs.expr + rhs.expr};
  }

  /// Real-Real compound addition operator.
  ///
  /// @param rhs Operator right-hand side.
  /// @return Result of addition.
  Real<Scalar>& operator+=(const Real<Scalar>& rhs) {
    *this = *this + rhs;
    return *this;
  }

  /// Real-Real subtraction operator.
  ///
  /// @param lhs Operator left-hand side.
  /// @param rhs Operator right-hand side.
  /// @return Result of subtraction.
  friend Real<Scalar> operator-(const Real<Scalar>& lhs,
                                const Real<Scalar>& rhs) {
    return Real{lhs.expr - rhs.expr};
  }

  /// Real-Real compound subtraction operator.
  ///
  /// @param rhs Operator right-hand side.
  /// @return Result of subtraction.
  Real<Scalar>& operator-=(const Real<Scalar>& rhs) {
    *this = *this - rhs;
    return *this;
  }

  /// Unary minus operator.
  ///
  /// @param lhs Operand for unary minus.
  friend Real<Scalar> operator-(const Real<Scalar>& lhs) {
    return Real{-lhs.expr};
  }

  /// Unary plus operator.
  ///
  /// @param lhs Operand for unary plus.
  friend Real<Scalar> operator+(const Real<Scalar>& lhs) {
    return Real{+lhs.expr};
  }

 private:
  /// The expression node
  detail::ExpressionPtr<Scalar> expr =
      detail::make_expression_ptr<detail::DecisionVariableExpression<Scalar>>();

  /// Used to update the value of this variable based on the values of its
  /// dependent variables
  detail::ExpressionGraph<Scalar> m_graph;

  /// Used for lazy initialization of m_graph
  bool m_graph_initialized = false;

  template <typename Scalar>
  friend Real<Scalar> abs(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> acos(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> asin(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> atan(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> atan2(const ScalarLike auto& y, const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> atan2(const Real<Scalar>& y, const ScalarLike auto& x);
  template <typename Scalar>
  friend Real<Scalar> atan2(const Real<Scalar>& y, const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> cbrt(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> cos(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> cosh(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> erf(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> exp(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> hypot(const ScalarLike auto& x, const Real<Scalar>& y);
  template <typename Scalar>
  friend Real<Scalar> hypot(const Real<Scalar>& x, const ScalarLike auto& y);
  template <typename Scalar>
  friend Real<Scalar> hypot(const Real<Scalar>& x, const Real<Scalar>& y);
  template <typename Scalar>
  friend Real<Scalar> if_else(const EqualityConstraints<Scalar>& cond,
                              const std::type_identity_t<Real<Scalar>>& t,
                              const std::type_identity_t<Real<Scalar>>& f);
  template <typename Scalar>
  friend Real<Scalar> if_else(const InequalityConstraints<Scalar>& cond,
                              const std::type_identity_t<Real<Scalar>>& t,
                              const std::type_identity_t<Real<Scalar>>& f);
  template <typename Scalar>
  friend Real<Scalar> log(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> log10(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> max(const ScalarLike auto& a, const Real<Scalar>& b);
  template <typename Scalar>
  friend Real<Scalar> max(const Real<Scalar>& a, const ScalarLike auto& b);
  template <typename Scalar>
  friend Real<Scalar> max(const Real<Scalar>& a, const Real<Scalar>& b);
  template <typename Scalar>
  friend Real<Scalar> min(const ScalarLike auto& a, const Real<Scalar>& b);
  template <typename Scalar>
  friend Real<Scalar> min(const Real<Scalar>& a, const ScalarLike auto& b);
  template <typename Scalar>
  friend Real<Scalar> min(const Real<Scalar>& a, const Real<Scalar>& b);
  template <typename Scalar>
  friend Real<Scalar> pow(const ScalarLike auto& base,
                          const Real<Scalar>& power);
  template <typename Scalar>
  friend Real<Scalar> pow(const Real<Scalar>& base,
                          const ScalarLike auto& power);
  template <typename Scalar>
  friend Real<Scalar> pow(const Real<Scalar>& base, const Real<Scalar>& power);
  template <typename Scalar>
  friend Real<Scalar> sign(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> sin(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> sinh(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> sqrt(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> tan(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> tanh(const Real<Scalar>& x);
  template <typename Scalar>
  friend Real<Scalar> hypot(const Real<Scalar>& x, const Real<Scalar>& y,
                            const Real<Scalar>& z);

  template <typename Scalar>
  friend RealMatrix<Scalar> detail::gradient_tree(
      const detail::ExpressionGraph<Scalar>& top_list,
      const RealMatrix<Scalar>& wrt);
  template <typename Scalar, int UpLo>
    requires(UpLo == Eigen::Lower) || (UpLo == (Eigen::Lower | Eigen::Upper))
  friend class Hessian;
  template <typename Scalar>
  friend class Jacobian;
  template <typename Scalar>
  friend std::vector<std::string_view> detail::replaceable_nonsmooth_functions(
      const Real<Scalar>& x, detail::ExpressionRole role);
};

template <template <typename> typename T, typename Scalar>
  requires SleipnirMatrixLike<T<Scalar>, Scalar>
Real(T<Scalar>) -> Real<Scalar>;

template <std::floating_point T>
Real(T) -> Real<T>;

template <std::integral T>
Real(T) -> Real<T>;

/// abs() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> abs(const Real<Scalar>& x) {
  return Real{detail::abs(x.expr)};
}

/// acos() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> acos(const Real<Scalar>& x) {
  return Real{detail::acos(x.expr)};
}

/// asin() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> asin(const Real<Scalar>& x) {
  return Real{detail::asin(x.expr)};
}

/// atan() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> atan(const Real<Scalar>& x) {
  return Real{detail::atan(x.expr)};
}

/// atan2() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param y The y argument.
/// @param x The x argument.
template <typename Scalar>
Real<Scalar> atan2(const ScalarLike auto& y, const Real<Scalar>& x) {
  return Real{detail::atan2(Real<Scalar>(y).expr, x.expr)};
}

/// atan2() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param y The y argument.
/// @param x The x argument.
template <typename Scalar>
Real<Scalar> atan2(const Real<Scalar>& y, const ScalarLike auto& x) {
  return Real{detail::atan2(y.expr, Real<Scalar>(x).expr)};
}

/// atan2() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param y The y argument.
/// @param x The x argument.
template <typename Scalar>
Real<Scalar> atan2(const Real<Scalar>& y, const Real<Scalar>& x) {
  return Real{detail::atan2(y.expr, x.expr)};
}

/// cbrt() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> cbrt(const Real<Scalar>& x) {
  return Real{detail::cbrt(x.expr)};
}

/// cos() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> cos(const Real<Scalar>& x) {
  return Real{detail::cos(x.expr)};
}

/// cosh() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> cosh(const Real<Scalar>& x) {
  return Real{detail::cosh(x.expr)};
}

/// erf() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> erf(const Real<Scalar>& x) {
  return Real{detail::erf(x.expr)};
}

/// exp() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> exp(const Real<Scalar>& x) {
  return Real{detail::exp(x.expr)};
}

/// hypot() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The x argument.
/// @param y The y argument.
template <typename Scalar>
Real<Scalar> hypot(const ScalarLike auto& x, const Real<Scalar>& y) {
  return Real{detail::hypot(Real<Scalar>(x).expr, y.expr)};
}

/// hypot() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The x argument.
/// @param y The y argument.
template <typename Scalar>
Real<Scalar> hypot(const Real<Scalar>& x, const ScalarLike auto& y) {
  return Real{detail::hypot(x.expr, Real<Scalar>(y).expr)};
}

/// hypot() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The x argument.
/// @param y The y argument.
template <typename Scalar>
Real<Scalar> hypot(const Real<Scalar>& x, const Real<Scalar>& y) {
  return Real{detail::hypot(x.expr, y.expr)};
}

/// log() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> log(const Real<Scalar>& x) {
  return Real{detail::log(x.expr)};
}

/// log10() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> log10(const Real<Scalar>& x) {
  return Real{detail::log10(x.expr)};
}

/// max() for Reals.
///
/// Returns the greater of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
/// @param a The a argument.
/// @param b The b argument.
template <typename Scalar>
Real<Scalar> max(const ScalarLike auto& a, const Real<Scalar>& b) {
  return Real{detail::max(Real<Scalar>(a).expr, b.expr)};
}

/// max() for Reals.
///
/// Returns the greater of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
/// @param a The a argument.
/// @param b The b argument.
template <typename Scalar>
Real<Scalar> max(const Real<Scalar>& a, const ScalarLike auto& b) {
  return Real{detail::max(a.expr, Real<Scalar>(b).expr)};
}

/// max() for Reals.
///
/// Returns the greater of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
/// @param a The a argument.
/// @param b The b argument.
template <typename Scalar>
Real<Scalar> max(const Real<Scalar>& a, const Real<Scalar>& b) {
  return Real{detail::max(a.expr, b.expr)};
}

/// min() for Reals.
///
/// Returns the lesser of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
/// @param a The a argument.
/// @param b The b argument.
template <typename Scalar>
Real<Scalar> min(const ScalarLike auto& a, const Real<Scalar>& b) {
  return Real{detail::min(Real<Scalar>(a).expr, b.expr)};
}

/// min() for Reals.
///
/// Returns the lesser of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
/// @param a The a argument.
/// @param b The b argument.
template <typename Scalar>
Real<Scalar> min(const Real<Scalar>& a, const ScalarLike auto& b) {
  return Real{detail::min(a.expr, Real<Scalar>(b).expr)};
}

/// min() for Reals.
///
/// Returns the lesser of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
/// @param a The a argument.
/// @param b The b argument.
template <typename Scalar>
Real<Scalar> min(const Real<Scalar>& a, const Real<Scalar>& b) {
  return Real{detail::min(a.expr, b.expr)};
}

/// pow() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param base The base.
/// @param power The power.
template <typename Scalar>
Real<Scalar> pow(const ScalarLike auto& base, const Real<Scalar>& power) {
  return Real{detail::pow(Real<Scalar>(base).expr, power.expr)};
}

/// pow() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param base The base.
/// @param power The power.
template <typename Scalar>
Real<Scalar> pow(const Real<Scalar>& base, const ScalarLike auto& power) {
  return Real{detail::pow(base.expr, Real<Scalar>(power).expr)};
}

/// pow() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param base The base.
/// @param power The power.
template <typename Scalar>
Real<Scalar> pow(const Real<Scalar>& base, const Real<Scalar>& power) {
  return Real{detail::pow(base.expr, power.expr)};
}

/// sign() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> sign(const Real<Scalar>& x) {
  return Real{detail::sign(x.expr)};
}

/// sin() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> sin(const Real<Scalar>& x) {
  return Real{detail::sin(x.expr)};
}

/// sinh() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> sinh(const Real<Scalar>& x) {
  return Real{detail::sinh(x.expr)};
}

/// sqrt() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> sqrt(const Real<Scalar>& x) {
  return Real{detail::sqrt(x.expr)};
}

/// tan() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> tan(const Real<Scalar>& x) {
  return Real{detail::tan(x.expr)};
}

/// tanh() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
Real<Scalar> tanh(const Real<Scalar>& x) {
  return Real{detail::tanh(x.expr)};
}

/// hypot() for Reals.
///
/// @tparam Scalar Scalar type.
/// @param x The x argument.
/// @param y The y argument.
/// @param z The z argument.
template <typename Scalar>
Real<Scalar> hypot(const Real<Scalar>& x, const Real<Scalar>& y,
                   const Real<Scalar>& z) {
  return Real{detail::hypot(x.expr, y.expr, z.expr)};
}

// The standard form for equality constraints is c(x) = 0, and the standard form
// for inequality constraints is c(x) ≥ 0. make_constraints() takes constraints
// of the form lhs = rhs or lhs ≥ rhs and converts them to lhs - rhs = 0 or
// lhs - rhs ≥ 0.

template <typename Scalar, ScalarLike LHS, ScalarLike RHS>
  requires SleipnirScalarLike<LHS, Scalar> || SleipnirScalarLike<RHS, Scalar>
auto make_constraints(LHS&& lhs, RHS&& rhs) {
  gch::small_vector<Real<Scalar>> constraints;
  constraints.emplace_back(lhs - rhs);

  return constraints;
}

template <typename Scalar, ScalarLike LHS, MatrixLike RHS>
  requires SleipnirScalarLike<LHS, Scalar> || SleipnirMatrixLike<RHS, Scalar>
auto make_constraints(LHS&& lhs, RHS&& rhs) {
  gch::small_vector<Real<Scalar>> constraints;
  constraints.reserve(rhs.rows() * rhs.cols());

  for (int row = 0; row < rhs.rows(); ++row) {
    for (int col = 0; col < rhs.cols(); ++col) {
      // Make right-hand side zero
      constraints.emplace_back(lhs - rhs[row, col]);
    }
  }

  return constraints;
}

template <typename Scalar, MatrixLike LHS, ScalarLike RHS>
  requires SleipnirMatrixLike<LHS, Scalar> || SleipnirScalarLike<RHS, Scalar>
auto make_constraints(LHS&& lhs, RHS&& rhs) {
  gch::small_vector<Real<Scalar>> constraints;
  constraints.reserve(lhs.rows() * lhs.cols());

  for (int row = 0; row < lhs.rows(); ++row) {
    for (int col = 0; col < lhs.cols(); ++col) {
      // Make right-hand side zero
      constraints.emplace_back(lhs[row, col] - rhs);
    }
  }

  return constraints;
}

template <typename Scalar, MatrixLike LHS, MatrixLike RHS>
  requires SleipnirMatrixLike<LHS, Scalar> || SleipnirMatrixLike<RHS, Scalar>
auto make_constraints(LHS&& lhs, RHS&& rhs) {
  slp_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());

  gch::small_vector<Real<Scalar>> constraints;
  constraints.reserve(lhs.rows() * lhs.cols());

  for (int row = 0; row < lhs.rows(); ++row) {
    for (int col = 0; col < lhs.cols(); ++col) {
      // Make right-hand side zero
      constraints.emplace_back(lhs[row, col] - rhs[row, col]);
    }
  }

  return constraints;
}

/// A vector of equality constraints of the form cₑ(x) = 0.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct EqualityConstraints {
  /// A vector of scalar equality constraints.
  gch::small_vector<Real<Scalar>> constraints;

  /// Concatenates multiple equality constraints.
  ///
  /// @param equality_constraints The list of EqualityConstraints to
  ///     concatenate.
  EqualityConstraints(
      std::initializer_list<EqualityConstraints> equality_constraints) {
    for (const auto& elem : equality_constraints) {
      constraints.insert(constraints.end(), elem.constraints.begin(),
                         elem.constraints.end());
    }
  }

  /// Concatenates multiple equality constraints.
  ///
  /// This overload is for Python bindings only.
  ///
  /// @param equality_constraints The list of EqualityConstraints to
  ///     concatenate.
  explicit EqualityConstraints(
      const std::vector<EqualityConstraints>& equality_constraints) {
    for (const auto& elem : equality_constraints) {
      constraints.insert(constraints.end(), elem.constraints.begin(),
                         elem.constraints.end());
    }
  }

  /// Constructs an equality constraint from a left and right side.
  ///
  /// The standard form for equality constraints is c(x) = 0. This function
  /// takes a constraint of the form lhs = rhs and converts it to lhs - rhs = 0.
  ///
  /// @param lhs Left-hand side.
  /// @param rhs Right-hand side.
  template <typename LHS, typename RHS>
    requires(ScalarLike<LHS> || MatrixLike<LHS>) &&
            (ScalarLike<RHS> || MatrixLike<RHS>) &&
            (SleipnirType<LHS> || SleipnirType<RHS>)
  EqualityConstraints(LHS&& lhs, RHS&& rhs)
      : constraints{make_constraints<Scalar>(lhs, rhs)} {}

  /// Implicit conversion operator to bool.
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator bool() {
    return std::ranges::all_of(constraints, [](auto& constraint) {
      return constraint.value() == Scalar(0);
    });
  }
};

/// A vector of inequality constraints of the form cᵢ(x) ≥ 0.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct InequalityConstraints {
  /// A vector of scalar inequality constraints.
  gch::small_vector<Real<Scalar>> constraints;

  /// Whether each scalar inequality constraint came from a strict comparison
  /// (< or >).
  ///
  /// The solver treats strict and non-strict inequality constraints the same.
  /// if_else() and the conversion to bool distinguish them.
  gch::small_vector<bool> strict;

  /// Concatenates multiple inequality constraints.
  ///
  /// @param inequality_constraints The list of InequalityConstraints to
  ///     concatenate.
  InequalityConstraints(
      std::initializer_list<InequalityConstraints> inequality_constraints) {
    for (const auto& elem : inequality_constraints) {
      constraints.insert(constraints.end(), elem.constraints.begin(),
                         elem.constraints.end());
      strict.insert(strict.end(), elem.strict.begin(), elem.strict.end());
    }
  }

  /// Concatenates multiple inequality constraints.
  ///
  /// This overload is for Python bindings only.
  ///
  /// @param inequality_constraints The list of InequalityConstraints to
  ///     concatenate.
  explicit InequalityConstraints(
      const std::vector<InequalityConstraints>& inequality_constraints) {
    for (const auto& elem : inequality_constraints) {
      constraints.insert(constraints.end(), elem.constraints.begin(),
                         elem.constraints.end());
      strict.insert(strict.end(), elem.strict.begin(), elem.strict.end());
    }
  }

  /// Constructs an inequality constraint from a left and right side.
  ///
  /// The standard form for inequality constraints is c(x) ≥ 0. This function
  /// takes a constraints of the form lhs ≥ rhs and converts it to lhs - rhs ≥
  /// 0.
  ///
  /// @param lhs Left-hand side.
  /// @param rhs Right-hand side.
  /// @param strict Whether the comparison is strict (lhs > rhs).
  template <typename LHS, typename RHS>
    requires(ScalarLike<LHS> || MatrixLike<LHS>) &&
                (ScalarLike<RHS> || MatrixLike<RHS>) &&
                (SleipnirType<LHS> || SleipnirType<RHS>)
  InequalityConstraints(LHS&& lhs, RHS&& rhs, bool strict = false)
      : constraints{make_constraints<Scalar>(lhs, rhs)},
        strict(constraints.size(), strict) {}

  /// Implicit conversion operator to bool.
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator bool() {
    return std::ranges::all_of(
        std::views::zip(constraints, strict), [](auto&& elem) {
          auto& [constraint, is_strict] = elem;
          return is_strict ? constraint.value() > Scalar(0)
                           : constraint.value() >= Scalar(0);
        });
  }
};

/// Equality operator that returns an equality constraint for two Reals.
///
/// @param lhs Left-hand side.
/// @param rhs Left-hand side.
template <typename LHS, typename RHS>
  requires(ScalarLike<LHS> || MatrixLike<LHS>) && SleipnirType<LHS> &&
          (ScalarLike<RHS> || MatrixLike<RHS>) && (!SleipnirType<RHS>) &&
          (!SymmetricMatrixLike<LHS>) && (!SymmetricMatrixLike<RHS>) &&
          (!UnitVectorLike<LHS>) && (!UnitVectorLike<RHS>)
auto operator==(LHS&& lhs, RHS&& rhs) {
  return EqualityConstraints<typename std::decay_t<LHS>::Scalar>{lhs, rhs};
}

/// Equality operator that returns an equality constraint for two Reals.
///
/// @param lhs Left-hand side.
/// @param rhs Left-hand side.
template <typename LHS, typename RHS>
  requires(ScalarLike<LHS> || MatrixLike<LHS>) && (!SleipnirType<LHS>) &&
          (ScalarLike<RHS> || MatrixLike<RHS>) && SleipnirType<RHS> &&
          (!SymmetricMatrixLike<LHS>) && (!SymmetricMatrixLike<RHS>) &&
          (!UnitVectorLike<LHS>) && (!UnitVectorLike<RHS>)
auto operator==(LHS&& lhs, RHS&& rhs) {
  return EqualityConstraints<typename std::decay_t<RHS>::Scalar>{lhs, rhs};
}

/// Equality operator that returns an equality constraint for two Reals.
///
/// @param lhs Left-hand side.
/// @param rhs Left-hand side.
template <typename LHS, typename RHS>
  requires(ScalarLike<LHS> || MatrixLike<LHS>) && SleipnirType<LHS> &&
          (ScalarLike<RHS> || MatrixLike<RHS>) && SleipnirType<RHS> &&
          (!SymmetricMatrixLike<LHS>) && (!SymmetricMatrixLike<RHS>) &&
          (!UnitVectorLike<LHS>) && (!UnitVectorLike<RHS>)
auto operator==(LHS&& lhs, RHS&& rhs) {
  return EqualityConstraints<typename std::decay_t<LHS>::Scalar>{lhs, rhs};
}

/// Less-than comparison operator that returns an inequality constraint for two
/// Reals.
///
/// @param lhs Left-hand side.
/// @param rhs Left-hand side.
template <typename LHS, typename RHS>
  requires(ScalarLike<LHS> || MatrixLike<LHS>) &&
          (ScalarLike<RHS> || MatrixLike<RHS>) &&
          (SleipnirType<LHS> || SleipnirType<RHS>) &&
          (!SymmetricMatrixLike<LHS>) && (!SymmetricMatrixLike<RHS>)
auto operator<(LHS&& lhs, RHS&& rhs) {
  return decltype(rhs >= lhs){rhs, lhs, true};
}

/// Less-than-or-equal-to comparison operator that returns an inequality
/// constraint for two Reals.
///
/// @param lhs Left-hand side.
/// @param rhs Left-hand side.
template <typename LHS, typename RHS>
  requires(ScalarLike<LHS> || MatrixLike<LHS>) &&
          (ScalarLike<RHS> || MatrixLike<RHS>) &&
          (SleipnirType<LHS> || SleipnirType<RHS>) &&
          (!SymmetricMatrixLike<LHS>) && (!SymmetricMatrixLike<RHS>)
auto operator<=(LHS&& lhs, RHS&& rhs) {
  return rhs >= lhs;
}

/// Greater-than comparison operator that returns an inequality constraint for
/// two Reals.
///
/// @param lhs Left-hand side.
/// @param rhs Left-hand side.
template <typename LHS, typename RHS>
  requires(ScalarLike<LHS> || MatrixLike<LHS>) &&
          (ScalarLike<RHS> || MatrixLike<RHS>) &&
          (SleipnirType<LHS> || SleipnirType<RHS>) &&
          (!SymmetricMatrixLike<LHS>) && (!SymmetricMatrixLike<RHS>)
auto operator>(LHS&& lhs, RHS&& rhs) {
  return decltype(lhs >= rhs){lhs, rhs, true};
}

/// Greater-than-or-equal-to comparison operator that returns an inequality
/// constraint for two Reals.
///
/// @param lhs Left-hand side.
/// @param rhs Left-hand side.
template <typename LHS, typename RHS>
  requires(ScalarLike<LHS> || MatrixLike<LHS>) && SleipnirType<LHS> &&
          (ScalarLike<RHS> || MatrixLike<RHS>) && (!SleipnirType<RHS>) &&
          (!SymmetricMatrixLike<LHS>) && (!SymmetricMatrixLike<RHS>)
auto operator>=(LHS&& lhs, RHS&& rhs) {
  return InequalityConstraints<typename std::decay_t<LHS>::Scalar>{lhs, rhs};
}

/// Greater-than-or-equal-to comparison operator that returns an inequality
/// constraint for two Reals.
///
/// @param lhs Left-hand side.
/// @param rhs Left-hand side.
template <typename LHS, typename RHS>
  requires(ScalarLike<LHS> || MatrixLike<LHS>) && (!SleipnirType<LHS>) &&
          (ScalarLike<RHS> || MatrixLike<RHS>) && SleipnirType<RHS> &&
          (!SymmetricMatrixLike<LHS>) && (!SymmetricMatrixLike<RHS>)
auto operator>=(LHS&& lhs, RHS&& rhs) {
  return InequalityConstraints<typename std::decay_t<RHS>::Scalar>{lhs, rhs};
}

/// Greater-than-or-equal-to comparison operator that returns an inequality
/// constraint for two Reals.
///
/// @param lhs Left-hand side.
/// @param rhs Left-hand side.
template <typename LHS, typename RHS>
  requires(ScalarLike<LHS> || MatrixLike<LHS>) && SleipnirType<LHS> &&
          (ScalarLike<RHS> || MatrixLike<RHS>) && SleipnirType<RHS> &&
          (!SymmetricMatrixLike<LHS>) && (!SymmetricMatrixLike<RHS>)
auto operator>=(LHS&& lhs, RHS&& rhs) {
  return InequalityConstraints<typename std::decay_t<LHS>::Scalar>{lhs, rhs};
}

/// Returns t if the equality constraint cond is satisfied, otherwise f.
///
/// For example, if_else(x == y, t, f) returns t if x = y, otherwise f.
///
/// The condition is evaluated on the current values of its operands each time
/// the result is evaluated, and the gradient only flows through the selected
/// branch. Both branches are evaluated regardless of which one is selected.
///
/// @tparam Scalar Scalar type.
/// @param cond A scalar equality constraint.
/// @param t Value selected when cond is satisfied.
/// @param f Value selected when cond is violated.
template <typename Scalar>
Real<Scalar> if_else(const EqualityConstraints<Scalar>& cond,
                     const std::type_identity_t<Real<Scalar>>& t,
                     const std::type_identity_t<Real<Scalar>>& f) {
  slp_assert(cond.constraints.size() == 1);

  return Real{detail::if_else(detail::IfElseCondition::ZERO,
                              cond.constraints[0].expr, t.expr, f.expr)};
}

/// Returns t if the inequality constraint cond is satisfied, otherwise f.
///
/// For example, if_else(x < y, t, f) returns t if x < y, otherwise f. Strict
/// comparisons (< and >) exclude equality and non-strict comparisons (≤ and ≥)
/// include it.
///
/// The condition is evaluated on the current values of its operands each time
/// the result is evaluated, and the gradient only flows through the selected
/// branch. Both branches are evaluated regardless of which one is selected.
///
/// @tparam Scalar Scalar type.
/// @param cond A scalar inequality constraint.
/// @param t Value selected when cond is satisfied.
/// @param f Value selected when cond is violated.
template <typename Scalar>
Real<Scalar> if_else(const InequalityConstraints<Scalar>& cond,
                     const std::type_identity_t<Real<Scalar>>& t,
                     const std::type_identity_t<Real<Scalar>>& f) {
  slp_assert(cond.constraints.size() == 1);

  return Real{detail::if_else(cond.strict[0]
                                  ? detail::IfElseCondition::POSITIVE
                                  : detail::IfElseCondition::NONNEGATIVE,
                              cond.constraints[0].expr, t.expr, f.expr)};
}

/// Helper function for creating bound constraints.
///
/// @param l Lower bound.
/// @param x Real to bound.
/// @param u Upper bound.
template <typename L, typename X, typename U>
  requires(ScalarLike<L> || MatrixLike<L>) && SleipnirType<X> &&
          (ScalarLike<U> || MatrixLike<U>)
auto bounds(L&& l, X&& x, U&& u) {
  return InequalityConstraints{l <= x, x <= u};
}

}  // namespace slp

namespace Eigen {

/// NumTraits specialization that allows instantiating Eigen types with
/// Real.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct NumTraits<slp::Real<Scalar>> : NumTraits<Scalar> {
  /// Real type.
  using Real = slp::Real<Scalar>;
  /// Non-integer type.
  using NonInteger = slp::Real<Scalar>;
  /// Nested type.
  using Nested = slp::Real<Scalar>;
  /// Literal type.
  using Literal = slp::Real<Scalar>;

  /// Is complex.
  static constexpr int IsComplex = 0;
  /// Is integer.
  static constexpr int IsInteger = 0;
  /// Is signed.
  static constexpr int IsSigned = 1;
  /// Require initialization.
  static constexpr int RequireInitialization = 1;
  /// Read cost.
  static constexpr int ReadCost = 1;
  /// Add cost.
  static constexpr int AddCost = 3;
  /// Multiply cost.
  static constexpr int MulCost = 3;
};

}  // namespace Eigen
