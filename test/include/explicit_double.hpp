// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <concepts>
#include <limits>
#include <ostream>

#include <Eigen/Core>
#include <fmt/base.h>

/// A scalar type not implicitly convertible from/to floating-point or integral
/// types.
///
/// Instantiating Sleipnir with this scalar type ensures casts are used in the
/// appropriate places. It's a wrapper around double, so it has the same
/// performance characteristics and accuracy as double.
class ExplicitDouble {
 public:
  // Construction
  constexpr ExplicitDouble() noexcept = default;
  constexpr explicit ExplicitDouble(std::floating_point auto value) noexcept
      : m_value(value) {}
  constexpr explicit ExplicitDouble(std::integral auto value) noexcept
      : m_value(value) {}

  // Observers
  double value() const { return m_value; }

  // Assignment
  constexpr ExplicitDouble& operator+=(const ExplicitDouble& rhs) {
    m_value += rhs.m_value;
    return *this;
  }
  constexpr ExplicitDouble& operator-=(const ExplicitDouble& rhs) {
    m_value -= rhs.m_value;
    return *this;
  }
  constexpr ExplicitDouble& operator*=(const ExplicitDouble& rhs) {
    m_value *= rhs.m_value;
    return *this;
  }
  constexpr ExplicitDouble& operator/=(const ExplicitDouble& rhs) {
    m_value /= rhs.m_value;
    return *this;
  }

  // Increment/decrement
  constexpr ExplicitDouble& operator++() noexcept {
    ++m_value;
    return *this;
  }
  constexpr ExplicitDouble& operator--() noexcept {
    --m_value;
    return *this;
  }
  constexpr ExplicitDouble operator++(int) noexcept {
    ExplicitDouble retval = *this;
    ++(*this);
    return retval;
  }
  constexpr ExplicitDouble operator--(int) noexcept {
    ExplicitDouble retval = *this;
    --(*this);
    return retval;
  }

  // Arithmetic
  constexpr ExplicitDouble operator+() const noexcept { return *this; }
  constexpr ExplicitDouble operator-() const noexcept {
    return ExplicitDouble{-m_value};
  }
  friend constexpr ExplicitDouble operator+(
      const ExplicitDouble& lhs, const ExplicitDouble& rhs) noexcept {
    return ExplicitDouble{lhs.m_value + rhs.m_value};
  }
  friend constexpr ExplicitDouble operator-(
      const ExplicitDouble& lhs, const ExplicitDouble& rhs) noexcept {
    return ExplicitDouble{lhs.m_value - rhs.m_value};
  }
  friend constexpr ExplicitDouble operator*(
      const ExplicitDouble& lhs, const ExplicitDouble& rhs) noexcept {
    return ExplicitDouble{lhs.m_value * rhs.m_value};
  }
  friend constexpr ExplicitDouble operator/(
      const ExplicitDouble& lhs, const ExplicitDouble& rhs) noexcept {
    return ExplicitDouble{lhs.m_value / rhs.m_value};
  }

  // Comparison
  friend constexpr auto operator<=>(const ExplicitDouble&,
                                    const ExplicitDouble&) noexcept = default;

  // Explicit conversion
  constexpr explicit operator double() const noexcept { return m_value; }
  constexpr explicit operator int() const noexcept {
    return static_cast<int>(m_value);
  }

  // cmath
  friend ExplicitDouble abs(const ExplicitDouble& x) {
    return ExplicitDouble{std::abs(x.m_value)};
  }
  friend ExplicitDouble acos(const ExplicitDouble& x) {
    return ExplicitDouble{std::acos(x.m_value)};
  }
  friend ExplicitDouble asin(const ExplicitDouble& x) {
    return ExplicitDouble{std::asin(x.m_value)};
  }
  friend ExplicitDouble atan(const ExplicitDouble& x) {
    return ExplicitDouble{std::atan(x.m_value)};
  }
  friend ExplicitDouble atan2(const ExplicitDouble& y,
                              const ExplicitDouble& x) {
    return ExplicitDouble{std::atan2(y.m_value, x.m_value)};
  }
  friend ExplicitDouble cbrt(const ExplicitDouble& x) {
    return ExplicitDouble{std::cbrt(x.m_value)};
  }
  friend ExplicitDouble cos(const ExplicitDouble& x) {
    return ExplicitDouble{std::cos(x.m_value)};
  }
  friend ExplicitDouble cosh(const ExplicitDouble& x) {
    return ExplicitDouble{std::cosh(x.m_value)};
  }
  friend ExplicitDouble erf(const ExplicitDouble& x) {
    return ExplicitDouble{std::erf(x.m_value)};
  }
  friend ExplicitDouble exp(const ExplicitDouble& x) {
    return ExplicitDouble{std::exp(x.m_value)};
  }
  friend ExplicitDouble hypot(const ExplicitDouble& x,
                              const ExplicitDouble& y) {
    return ExplicitDouble{std::hypot(x.m_value, y.m_value)};
  }
  friend ExplicitDouble hypot(const ExplicitDouble& x, const ExplicitDouble& y,
                              const ExplicitDouble& z) {
    return ExplicitDouble{std::hypot(x.m_value, y.m_value, z.m_value)};
  }
  friend bool isfinite(const ExplicitDouble& num) {
    return std::isfinite(num.m_value);
  }
  friend ExplicitDouble log(const ExplicitDouble& x) {
    return ExplicitDouble{std::log(x.m_value)};
  }
  friend ExplicitDouble log10(const ExplicitDouble& x) {
    return ExplicitDouble{std::log10(x.m_value)};
  }
  friend ExplicitDouble pow(const ExplicitDouble& base,
                            const ExplicitDouble& power) {
    return ExplicitDouble{std::pow(base.m_value, power.m_value)};
  }
  friend ExplicitDouble sin(const ExplicitDouble& x) {
    return ExplicitDouble{std::sin(x.m_value)};
  }
  friend ExplicitDouble sinh(const ExplicitDouble& x) {
    return ExplicitDouble{std::sinh(x.m_value)};
  }
  friend ExplicitDouble sqrt(const ExplicitDouble& x) {
    return ExplicitDouble{std::sqrt(x.m_value)};
  }
  friend ExplicitDouble tan(const ExplicitDouble& x) {
    return ExplicitDouble{std::tan(x.m_value)};
  }
  friend ExplicitDouble tanh(const ExplicitDouble& x) {
    return ExplicitDouble{std::tanh(x.m_value)};
  }

  // Formatted output
  template <typename CharT, typename Traits>
  friend std::basic_ostream<CharT, Traits>& operator<<(
      std::basic_ostream<CharT, Traits>& os, const ExplicitDouble& d) {
    return os << d.m_value;
  }

 private:
  double m_value;
};

template <>
struct fmt::formatter<ExplicitDouble> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return m_underlying.parse(ctx);
  }

  template <typename FmtContext>
  auto format(const ExplicitDouble& d, FmtContext& ctx) const {
    return m_underlying.format(d.value(), ctx);
  }

 private:
  fmt::formatter<double> m_underlying;
};

namespace std {

template <>
struct numeric_limits<ExplicitDouble> {
  static constexpr bool is_specialized = numeric_limits<double>::is_specialized;

  static constexpr ExplicitDouble min() noexcept {
    return ExplicitDouble{numeric_limits<double>::min()};
  }

  static constexpr ExplicitDouble max() noexcept {
    return ExplicitDouble{numeric_limits<double>::max()};
  }

  static constexpr ExplicitDouble lowest() noexcept {
    return ExplicitDouble{numeric_limits<double>::lowest()};
  }

  static constexpr int digits = numeric_limits<double>::digits;
  static constexpr int digits10 = numeric_limits<double>::digits10;
  static constexpr int max_digits10 = numeric_limits<double>::max_digits10;
  static constexpr bool is_signed = numeric_limits<double>::is_signed;
  static constexpr bool is_integer = numeric_limits<double>::is_integer;
  static constexpr bool is_exact = numeric_limits<double>::is_exact;
  static constexpr int radix = numeric_limits<double>::radix;

  static constexpr ExplicitDouble epsilon() noexcept {
    return ExplicitDouble{numeric_limits<double>::epsilon()};
  }

  static constexpr ExplicitDouble round_error() noexcept {
    return ExplicitDouble{numeric_limits<double>::round_error()};
  }

  static constexpr int min_exponent = numeric_limits<double>::min_exponent;
  static constexpr int min_exponent10 = numeric_limits<double>::min_exponent10;
  static constexpr int max_exponent = numeric_limits<double>::max_exponent;
  static constexpr int max_exponent10 = numeric_limits<double>::max_exponent10;

  static constexpr bool has_infinity = numeric_limits<double>::has_infinity;
  static constexpr bool has_quiet_NaN = numeric_limits<double>::has_quiet_NaN;
  static constexpr bool has_signaling_NaN =
      numeric_limits<double>::has_signaling_NaN;

  static constexpr ExplicitDouble infinity() noexcept {
    return ExplicitDouble{numeric_limits<double>::infinity()};
  }

  static constexpr ExplicitDouble quiet_NaN() noexcept {
    return ExplicitDouble{numeric_limits<double>::quiet_NaN()};
  }

  static constexpr ExplicitDouble signaling_NaN() noexcept {
    return ExplicitDouble{numeric_limits<double>::signaling_NaN()};
  }

  static constexpr ExplicitDouble denorm_min() noexcept {
    return ExplicitDouble{numeric_limits<double>::denorm_min()};
  }

  static constexpr bool is_iec559 = numeric_limits<double>::is_iec559;
  static constexpr bool is_bounded = numeric_limits<double>::is_bounded;
  static constexpr bool is_modulo = numeric_limits<double>::is_modulo;

  static constexpr bool traps = numeric_limits<double>::traps;
  static constexpr bool tinyness_before =
      numeric_limits<double>::tinyness_before;
  static constexpr float_round_style round_style =
      numeric_limits<double>::round_style;
};

}  // namespace std

namespace Eigen {

/// NumTraits specialization that allows instantiating Eigen types with
/// ExplicitDouble.
template <>
struct NumTraits<ExplicitDouble> : GenericNumTraits<ExplicitDouble> {
  /// Is complex.
  static constexpr int IsComplex = NumTraits<double>::IsComplex;
  /// Is integer.
  static constexpr int IsInteger = NumTraits<double>::IsInteger;
  /// Is signed.
  static constexpr int IsSigned = NumTraits<double>::IsSigned;
  /// Require initialization.
  static constexpr int RequireInitialization =
      NumTraits<double>::RequireInitialization;
  /// Read cost.
  static constexpr int ReadCost = NumTraits<double>::ReadCost;
  /// Add cost.
  static constexpr int AddCost = NumTraits<double>::AddCost;
  /// Multiply cost.
  static constexpr int MulCost = NumTraits<double>::MulCost;

  static constexpr Real epsilon() { return Real(NumTraits<double>::epsilon()); }
  static constexpr Real dummy_precision() {
    return Real(NumTraits<double>::dummy_precision());
  }
  static constexpr int digits10() { return NumTraits<double>::digits10(); }
  static constexpr int max_digits10() {
    return NumTraits<double>::max_digits10();
  }
};

}  // namespace Eigen
