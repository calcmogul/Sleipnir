// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>

#include <complex>

#include <sleipnir/autodiff/complex.hpp>
#include <sleipnir/autodiff/complex_matrix.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>

#include "docstrings.hpp"
#include "for_each_type.hpp"

namespace nb = nanobind;

namespace slp {

void bind_complex(nb::class_<Complex<double>>& cls) {
  using namespace nb::literals;

  using C = Complex<double>;
  using V = Real<double>;

  cls.def(nb::init<>(), DOC(slp, Complex, Complex));
  cls.def(nb::init<V, V>(), "real"_a, "imag"_a, DOC(slp, Complex, Complex, 3));
  // The complex constructor comes before the real one so that NumPy complex
  // scalars, which convert to float with a warning, keep their imaginary part
  cls.def(nb::init<std::complex<double>>(), "value"_a,
          DOC(slp, Complex, Complex, 5));
  cls.def(nb::init<const V&>(), "real"_a, DOC(slp, Complex, Complex, 4));
  cls.def_static("polar", &C::polar, "magnitude"_a, "angle"_a,
                 DOC(slp, Complex, polar));

  // Python can't enforce const references, so the parts are returned by value.
  // They're handles to the same expressions.
  cls.def(
      "real", [](const C& self) { return V{self.real()}; },
      DOC(slp, Complex, real));
  cls.def(
      "imag", [](const C& self) { return V{self.imag()}; },
      DOC(slp, Complex, imag));

  cls.def("value", &C::value, DOC(slp, Complex, value));
  cls.def("set_value", &C::set_value, "value"_a, DOC(slp, Complex, set_value));

  cls.def("conj", &C::conj, DOC(slp, Complex, conj));
  cls.def("abs2", &C::abs2, DOC(slp, Complex, abs2));
  cls.def("abs", &C::abs, DOC(slp, Complex, abs));
  cls.def("__abs__", &C::abs, DOC(slp, Complex, abs));
  cls.def("arg", &C::arg, DOC(slp, Complex, arg));

  for_each_type<const C&, std::complex<double>, const V&, double>(
      [&]<typename T> {
        cls.def(
            "__add__", [](const C& lhs, T rhs) { return lhs + rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, add));
        cls.def(
            "__radd__", [](const C& rhs, T lhs) { return lhs + rhs; },
            nb::is_operator(), "lhs"_a, DOC(slp, operator, add));
        cls.def(
            "__sub__", [](const C& lhs, T rhs) { return lhs - rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, sub));
        cls.def(
            "__rsub__", [](const C& rhs, T lhs) { return lhs - rhs; },
            nb::is_operator(), "lhs"_a, DOC(slp, operator, sub));
        cls.def(
            "__mul__", [](const C& lhs, T rhs) { return lhs * rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, mul));
        cls.def(
            "__rmul__", [](const C& rhs, T lhs) { return lhs * rhs; },
            nb::is_operator(), "lhs"_a, DOC(slp, operator, mul));
        cls.def(
            "__truediv__", [](const C& lhs, T rhs) { return lhs / rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, div));
        cls.def(
            "__rtruediv__", [](const C& rhs, T lhs) { return lhs / rhs; },
            nb::is_operator(), "lhs"_a, DOC(slp, operator, div));
        cls.def(
            "__eq__", [](const C& lhs, T rhs) { return lhs == rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, eq));
      });

  // Products with real matrices
  cls.def(
      "__mul__",
      [](const C& lhs, const RealMatrix<double>& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 7));
  cls.def(
      "__rmul__",
      [](const C& rhs, const RealMatrix<double>& lhs) { return lhs * rhs; },
      nb::is_operator(), "lhs"_a, DOC(slp, operator, mul, 6));

  cls.def(
      "__neg__", [](const C& self) { return -self; },
      DOC(slp, operator, sub, 5));
  cls.def(
      "__pos__", [](const C& self) { return +self; },
      DOC(slp, operator, add, 5));

  // NumPy scalars on the left defer to the reflected operators above
  cls.attr("__array_ufunc__") = nb::none();
}

}  // namespace slp
