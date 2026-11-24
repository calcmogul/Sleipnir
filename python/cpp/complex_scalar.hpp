// Copyright (c) Sleipnir contributors

#pragma once

#include <nanobind/nanobind.h>

#include <stdint.h>

#include <complex>

namespace nb = nanobind;

namespace slp {

/// A Python complex number, including subclasses like numpy.complex128.
///
/// Unlike std::complex<double>, whose caster also converts floats and anything
/// with __complex__() or __float__(), this only accepts complex instances. It
/// matches during nanobind's first overload pass, which doesn't apply implicit
/// conversions, so a numpy.complex128 selects its complex overload before a
/// float overload converts it with __float__() and drops its imaginary part.
struct ComplexScalar {
  /// The value.
  std::complex<double> value;
};

}  // namespace slp

namespace nanobind::detail {

/// Type caster for ComplexScalar.
template <>
struct type_caster<slp::ComplexScalar> {
  NB_TYPE_CASTER(slp::ComplexScalar, const_name("complex"))

  /// Converts a Python complex number to a ComplexScalar.
  ///
  /// @param src The Python object.
  /// @return True if src is a complex number.
  bool from_python(handle src, uint32_t, cleanup_list*) noexcept {
    if (!PyComplex_Check(src.ptr())) {
      return false;
    }
    value.value = {PyComplex_RealAsDouble(src.ptr()),
                   PyComplex_ImagAsDouble(src.ptr())};
    return true;
  }

  /// Converts a ComplexScalar to a Python complex number.
  ///
  /// @param value The ComplexScalar.
  /// @return The Python complex number.
  static handle from_cpp(const slp::ComplexScalar& value, rv_policy,
                         cleanup_list*) noexcept {
    return PyComplex_FromDoubles(value.value.real(), value.value.imag());
  }
};

}  // namespace nanobind::detail
