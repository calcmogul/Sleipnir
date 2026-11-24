// Copyright (c) Sleipnir contributors

#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>

#include <stdint.h>

#include <complex>
#include <optional>

#include <sleipnir/autodiff/complex.hpp>
#include <sleipnir/autodiff/complex_matrix.hpp>
#include <sleipnir/autodiff/complex_submatrix.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/real_submatrix.hpp>

#include "try_cast.hpp"

namespace nb = nanobind;

namespace slp {

/// Assigns a Python value to a ComplexSubmatrix.
///
/// Matrices must match the submatrix's shape, and scalars require a 1x1
/// submatrix. Values that aren't a matrix or scalar raise TypeError.
///
/// @param lhs The ComplexSubmatrix.
/// @param value The value.
inline void assign(ComplexSubmatrix<ComplexMatrix<double>> lhs,
                   const nb::object& value) {
  using M = ComplexMatrix<double>;

  if (auto rhs = try_cast<M>(value)) {
    lhs = rhs.value();
  } else if (auto rhs = try_cast<ComplexSubmatrix<M>>(value)) {
    lhs = M{rhs.value()};
  } else if (auto rhs = try_cast<RealMatrix<double>>(value)) {
    lhs = M{rhs.value()};
  } else if (auto rhs = try_cast<RealSubmatrix<RealMatrix<double>>>(value)) {
    lhs = M{rhs.value()};
  } else if (auto rhs = try_cast_to_eigen<std::complex<double>>(value)) {
    lhs = rhs.value();
  } else if (auto rhs = try_cast_to_eigen<double>(value)) {
    lhs = rhs.value();
  } else if (auto rhs = try_cast_to_eigen<int64_t>(value)) {
    lhs = rhs.value().cast<double>();
  } else if (auto rhs = try_cast_to_eigen<int32_t>(value)) {
    lhs = rhs.value().cast<double>();
  } else {
    std::optional<Complex<double>> scalar;
    if (auto rhs = try_cast<Complex<double>>(value)) {
      scalar = rhs.value();
    } else if (auto rhs = try_cast<Real<double>>(value)) {
      scalar = Complex<double>{rhs.value()};
    } else if (auto rhs = try_cast<int>(value)) {
      scalar = Complex<double>{static_cast<double>(rhs.value())};
    } else if (auto rhs = try_cast<double>(value)) {
      scalar = Complex<double>{rhs.value()};
    } else if (auto rhs = try_cast<std::complex<double>>(value)) {
      scalar = Complex<double>{rhs.value()};
    } else {
      throw nb::type_error(
          "ComplexMatrix.__setitem__ not implemented for value");
    }

    if (lhs.rows() != 1 || lhs.cols() != 1) {
      throw nb::value_error("Scalars can only be assigned to 1x1 slices");
    }
    lhs = scalar.value();
  }
}

}  // namespace slp
