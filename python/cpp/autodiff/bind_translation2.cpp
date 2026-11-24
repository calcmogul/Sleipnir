// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/translation2.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_translation2(nb::class_<Translation2<double>>& cls) {
  using namespace nb::literals;

  using T = Translation2<double>;
  using V = Real<double>;

  cls.def(nb::init<V, V>(), "x"_a, "y"_a, DOC(slp, Translation2, Translation2));
  cls.def(nb::init<const RealMatrix<double>&>(), "vector"_a,
          DOC(slp, Translation2, Translation2, 2));
  cls.def(
      "__init__",
      [](T* self, const Eigen::Vector2d& vector) { new (self) T{vector}; },
      "vector"_a, DOC(slp, Translation2, Translation2, 3));

  // Python can't enforce const references, so the components are returned by
  // value. They're handles to the same expressions.
  cls.def(
      "x", [](const T& self) { return V{self.x()}; },
      DOC(slp, Translation2, x));
  cls.def(
      "y", [](const T& self) { return V{self.y()}; },
      DOC(slp, Translation2, y));

  cls.def("value", &T::value, DOC(slp, Translation2, value));
  cls.def("set_value", &T::set_value, "value"_a,
          DOC(slp, Translation2, set_value));

  cls.def("distance", &T::distance, "other"_a,
          DOC(slp, Translation2, distance));
  cls.def("squared_distance", &T::squared_distance, "other"_a,
          DOC(slp, Translation2, squared_distance));

  cls.def(
      "__add__", [](const T& lhs, const T& rhs) { return lhs + rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, add, 8));
  cls.def(
      "__sub__", [](const T& lhs, const T& rhs) { return lhs - rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, sub, 8));
  cls.def(
      "__neg__", [](const T& self) { return -self; },
      DOC(slp, operator, sub, 9));
  cls.def(
      "__mul__", [](const T& lhs, const V& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 10));
  cls.def(
      "__rmul__", [](const T& rhs, const V& lhs) { return lhs * rhs; },
      nb::is_operator(), "lhs"_a, DOC(slp, operator, mul, 11));
  cls.def(
      "__truediv__", [](const T& lhs, const V& rhs) { return lhs / rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, div, 7));
  cls.def(
      "__eq__", [](const T& lhs, const T& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 10));
}

}  // namespace slp
