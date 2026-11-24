// Copyright (c) Sleipnir contributors

#pragma once

#include <complex>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/util/concepts.hpp"

namespace slp {

/// A complex number of autodiff expressions.
///
/// A complex number z = a + bi stores its real part a and imaginary part b as
/// two Reals, so the solver sees two real decision variables per complex
/// decision variable. Costs must be real-valued, so a cost is written with
/// real(), imag(), abs2(), or similar.
///
/// An equality constraint between complex numbers constrains the real and
/// imaginary parts separately. Complex numbers aren't ordered, so inequality
/// constraints are written on real-valued expressions like abs2() or real().
///
/// Complex numbers are created by Problem::complex_decision_variable(), from
/// constants, and from expressions.
///
/// @tparam Scalar_ Scalar type. std::complex is only specified for
///     floating-point types.
template <std::floating_point Scalar_>
class Complex {
 public:
  /// Scalar type alias.
  using Scalar = Scalar_;

  /// Constructs a complex zero.
  Complex() : m_real{Scalar(0)}, m_imag{Scalar(0)} {}

  /// Constructs an empty complex number.
  explicit Complex(std::nullptr_t) : m_real{nullptr}, m_imag{nullptr} {}

  /// Constructs a complex number from its real and imaginary parts.
  ///
  /// @param real The real part.
  /// @param imag The imaginary part.
  Complex(Real<Scalar> real, Real<Scalar> imag)
      : m_real{std::move(real)}, m_imag{std::move(imag)} {}

  /// Constructs a complex number with a zero imaginary part.
  ///
  /// @param real The real part.
  template <ScalarLike T>
  // NOLINTNEXTLINE (google-explicit-constructor)
  Complex(const T& real) : m_real{real}, m_imag{Scalar(0)} {}

  /// Constructs a constant complex number.
  ///
  /// @param value The value.
  // NOLINTNEXTLINE (google-explicit-constructor)
  Complex(const std::complex<Scalar>& value)
      : m_real{value.real()}, m_imag{value.imag()} {}

  /// Returns a complex number from its polar form r e^{iθ}.
  ///
  /// @param magnitude The magnitude r.
  /// @param angle The angle θ in radians.
  /// @return The complex number.
  static Complex polar(const Real<Scalar>& magnitude,
                       const Real<Scalar>& angle) {
    return Complex{magnitude * cos(angle), magnitude * sin(angle)};
  }

  /// Returns the real part.
  ///
  /// @return The real part.
  Real<Scalar>& real() { return m_real; }

  /// Returns the real part.
  ///
  /// @return The real part.
  const Real<Scalar>& real() const { return m_real; }

  /// Returns the imaginary part.
  ///
  /// @return The imaginary part.
  Real<Scalar>& imag() { return m_imag; }

  /// Returns the imaginary part.
  ///
  /// @return The imaginary part.
  const Real<Scalar>& imag() const { return m_imag; }

  /// Returns the complex number's value.
  ///
  /// @return The complex number's value.
  std::complex<Scalar> value() {
    return std::complex<Scalar>{m_real.value(), m_imag.value()};
  }

  /// Sets the complex number's value.
  ///
  /// This is only meaningful for complex decision variables and parameters.
  ///
  /// @param value The value.
  void set_value(const std::complex<Scalar>& value) {
    m_real.set_value(value.real());
    m_imag.set_value(value.imag());
  }

  /// Returns the complex conjugate a − bi.
  ///
  /// @return The complex conjugate.
  Complex conj() const { return Complex{m_real, -m_imag}; }

  /// Returns the squared magnitude a² + b².
  ///
  /// @return The squared magnitude.
  Real<Scalar> abs2() const { return m_real * m_real + m_imag * m_imag; }

  /// Returns the magnitude √(a² + b²).
  ///
  /// The magnitude isn't differentiable at zero. For constraints and costs
  /// that can reach zero, use abs2().
  ///
  /// @return The magnitude.
  Real<Scalar> abs() const { return hypot(m_real, m_imag); }

  /// Returns the angle atan2(b, a) in radians.
  ///
  /// The angle isn't differentiable at zero or continuous across the negative
  /// real axis.
  ///
  /// @return The angle.
  Real<Scalar> arg() const { return atan2(m_imag, m_real); }

 private:
  Real<Scalar> m_real;
  Real<Scalar> m_imag;
};

template <typename Scalar>
Complex(Real<Scalar>, Real<Scalar>) -> Complex<Scalar>;

template <typename Scalar>
Complex(std::complex<Scalar>) -> Complex<Scalar>;

namespace detail {

/// Whether T is Complex<Scalar>.
template <typename T, typename Scalar>
concept SleipnirComplex = std::same_as<std::decay_t<T>, Complex<Scalar>>;

/// Whether T converts to Complex<Scalar> as an operand of complex arithmetic:
/// a Complex, a Real, a std::complex, or a real constant.
template <typename T, typename Scalar>
concept ComplexOperand =
    SleipnirComplex<T, Scalar> ||
    std::same_as<std::decay_t<T>, std::complex<Scalar>> ||
    (ScalarLike<T> && (!SleipnirType<T> || SleipnirScalarLike<T, Scalar>));

/// Whether T is a std::complex<Scalar>.
template <typename T, typename Scalar>
concept StdComplex = std::same_as<std::decay_t<T>, std::complex<Scalar>>;

}  // namespace detail

// The binary operators take a Complex on one side and anything that converts
// to Complex on the other. The second overload of each operator excludes a
// Complex left-hand side so Complex-Complex operands select the first.

/// Adds two complex numbers.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar, detail::ComplexOperand<Scalar> RHS>
Complex<Scalar> operator+(const Complex<Scalar>& lhs, const RHS& rhs) {
  Complex<Scalar> r{rhs};
  return Complex<Scalar>{lhs.real() + r.real(), lhs.imag() + r.imag()};
}

/// Adds two complex numbers.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar, detail::ComplexOperand<Scalar> LHS>
  requires(!detail::SleipnirComplex<LHS, Scalar>)
Complex<Scalar> operator+(const LHS& lhs, const Complex<Scalar>& rhs) {
  return rhs + lhs;
}

/// Adds a Real and a constant complex number.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar>
Complex<Scalar> operator+(const Real<Scalar>& lhs,
                          const std::complex<Scalar>& rhs) {
  return Complex<Scalar>{lhs} + rhs;
}

/// Adds a constant complex number and a Real.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The sum.
template <typename Scalar>
Complex<Scalar> operator+(const std::complex<Scalar>& lhs,
                          const Real<Scalar>& rhs) {
  return Complex<Scalar>{rhs} + lhs;
}

/// Subtracts two complex numbers.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar, detail::ComplexOperand<Scalar> RHS>
Complex<Scalar> operator-(const Complex<Scalar>& lhs, const RHS& rhs) {
  Complex<Scalar> r{rhs};
  return Complex<Scalar>{lhs.real() - r.real(), lhs.imag() - r.imag()};
}

/// Subtracts two complex numbers.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar, detail::ComplexOperand<Scalar> LHS>
  requires(!detail::SleipnirComplex<LHS, Scalar>)
Complex<Scalar> operator-(const LHS& lhs, const Complex<Scalar>& rhs) {
  return Complex<Scalar>{lhs} - rhs;
}

/// Subtracts a constant complex number from a Real.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar>
Complex<Scalar> operator-(const Real<Scalar>& lhs,
                          const std::complex<Scalar>& rhs) {
  return Complex<Scalar>{lhs} - rhs;
}

/// Subtracts a Real from a constant complex number.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The difference.
template <typename Scalar>
Complex<Scalar> operator-(const std::complex<Scalar>& lhs,
                          const Real<Scalar>& rhs) {
  return Complex<Scalar>{lhs} - rhs;
}

/// Multiplies two complex numbers.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product.
template <typename Scalar, detail::ComplexOperand<Scalar> RHS>
Complex<Scalar> operator*(const Complex<Scalar>& lhs, const RHS& rhs) {
  if constexpr (ScalarLike<RHS>) {
    // A real factor scales both parts
    Real<Scalar> r{rhs};
    return Complex<Scalar>{lhs.real() * r, lhs.imag() * r};
  } else {
    Complex<Scalar> r{rhs};
    return Complex<Scalar>{lhs.real() * r.real() - lhs.imag() * r.imag(),
                           lhs.real() * r.imag() + lhs.imag() * r.real()};
  }
}

/// Multiplies two complex numbers.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product.
template <typename Scalar, detail::ComplexOperand<Scalar> LHS>
  requires(!detail::SleipnirComplex<LHS, Scalar>)
Complex<Scalar> operator*(const LHS& lhs, const Complex<Scalar>& rhs) {
  return rhs * lhs;
}

/// Multiplies a Real by a constant complex number.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product.
template <typename Scalar>
Complex<Scalar> operator*(const Real<Scalar>& lhs,
                          const std::complex<Scalar>& rhs) {
  return Complex<Scalar>{lhs * rhs.real(), lhs * rhs.imag()};
}

/// Multiplies a constant complex number by a Real.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The product.
template <typename Scalar>
Complex<Scalar> operator*(const std::complex<Scalar>& lhs,
                          const Real<Scalar>& rhs) {
  return rhs * lhs;
}

/// Divides two complex numbers.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The quotient.
template <typename Scalar, detail::ComplexOperand<Scalar> RHS>
Complex<Scalar> operator/(const Complex<Scalar>& lhs, const RHS& rhs) {
  if constexpr (ScalarLike<RHS>) {
    // A real divisor divides both parts
    Real<Scalar> r{rhs};
    return Complex<Scalar>{lhs.real() / r, lhs.imag() / r};
  } else if constexpr (detail::StdComplex<RHS, Scalar>) {
    return lhs * (Scalar{1} / rhs);
  } else {
    // (a + bi)/(c + di) = ((ac + bd) + (bc − ad)i)/(c² + d²)
    Real<Scalar> denom = rhs.abs2();
    return Complex<Scalar>{
        (lhs.real() * rhs.real() + lhs.imag() * rhs.imag()) / denom,
        (lhs.imag() * rhs.real() - lhs.real() * rhs.imag()) / denom};
  }
}

/// Divides two complex numbers.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The quotient.
template <typename Scalar, detail::ComplexOperand<Scalar> LHS>
  requires(!detail::SleipnirComplex<LHS, Scalar>)
Complex<Scalar> operator/(const LHS& lhs, const Complex<Scalar>& rhs) {
  return Complex<Scalar>{lhs} / rhs;
}

/// Divides a Real by a constant complex number.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The quotient.
template <typename Scalar>
Complex<Scalar> operator/(const Real<Scalar>& lhs,
                          const std::complex<Scalar>& rhs) {
  return lhs * (Scalar{1} / rhs);
}

/// Divides a constant complex number by a Real.
///
/// @param lhs Operator left-hand side.
/// @param rhs Operator right-hand side.
/// @return The quotient.
template <typename Scalar>
Complex<Scalar> operator/(const std::complex<Scalar>& lhs,
                          const Real<Scalar>& rhs) {
  return Complex<Scalar>{lhs.real() / rhs, lhs.imag() / rhs};
}

/// Negates a complex number.
///
/// @param z The complex number.
/// @return The negated complex number.
template <typename Scalar>
Complex<Scalar> operator-(const Complex<Scalar>& z) {
  return Complex<Scalar>{-z.real(), -z.imag()};
}

/// Returns a complex number unchanged.
///
/// @param z The complex number.
/// @return The complex number.
template <typename Scalar>
Complex<Scalar> operator+(const Complex<Scalar>& z) {
  return z;
}

/// Equality operator that returns equality constraints on the real and
/// imaginary parts of two complex numbers.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side.
/// @return The equality constraints.
template <typename Scalar>
EqualityConstraints<Scalar> operator==(const Complex<Scalar>& lhs,
                                       const Complex<Scalar>& rhs) {
  return {lhs.real() == rhs.real(), lhs.imag() == rhs.imag()};
}

// The mixed-operand overloads below have identical constraints, so for
// `a == b`, overload resolution picks the non-reversed candidate.

/// Equality operator that returns equality constraints on the real and
/// imaginary parts of a complex number and another operand.
///
/// @param lhs Left-hand side.
/// @param rhs Right-hand side. It's a Real, a std::complex, or a real
///     constant.
/// @return The equality constraints.
template <typename Scalar, detail::ComplexOperand<Scalar> RHS>
  requires(!detail::SleipnirComplex<RHS, Scalar>)
EqualityConstraints<Scalar> operator==(const Complex<Scalar>& lhs,
                                       const RHS& rhs) {
  return lhs == Complex<Scalar>{rhs};
}

/// Equality operator that returns equality constraints on the real and
/// imaginary parts of another operand and a complex number.
///
/// @param lhs Left-hand side. It's a Real, a std::complex, or a real
///     constant.
/// @param rhs Right-hand side.
/// @return The equality constraints.
template <typename Scalar, detail::ComplexOperand<Scalar> LHS>
  requires(!detail::SleipnirComplex<LHS, Scalar>)
EqualityConstraints<Scalar> operator==(const LHS& lhs,
                                       const Complex<Scalar>& rhs) {
  return Complex<Scalar>{lhs} == rhs;
}

}  // namespace slp
