// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>

#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/rotation2.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_rotation2(nb::class_<Rotation2<double>>& cls) {
  using namespace nb::literals;

  using UC = Rotation2<double>;

  cls.def(nb::init<double>(), "angle"_a, DOC(slp, Rotation2, Rotation2));
  cls.def(nb::init<const Real<double>&>(), "angle"_a,
          DOC(slp, Rotation2, Rotation2, 2));

  // Python can't enforce const references, so the components are returned by
  // value. They're handles to the same expressions.
  cls.def(
      "cos", [](const UC& self) { return Real<double>{self.cos()}; },
      DOC(slp, Rotation2, cos));
  cls.def(
      "sin", [](const UC& self) { return Real<double>{self.sin()}; },
      DOC(slp, Rotation2, sin));
  cls.def("angle", &UC::angle, DOC(slp, Rotation2, angle));

  cls.def("value", &UC::value, DOC(slp, Rotation2, value));
  cls.def("set_value", &UC::set_value, "angle"_a,
          DOC(slp, Rotation2, set_value));

  cls.def(
      "__mul__", [](const UC& lhs, const UC& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 12));
  cls.def(
      "__eq__", [](const UC& lhs, const UC& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 14));
  cls.def("relative_to", &UC::relative_to, "other"_a,
          DOC(slp, Rotation2, relative_to));
  cls.def("inverse", &UC::inverse, DOC(slp, Rotation2, inverse));
  cls.def("to_matrix", &UC::to_matrix, DOC(slp, Rotation2, to_matrix));
}

}  // namespace slp
