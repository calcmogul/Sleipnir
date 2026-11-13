// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/translation3.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_translation3(nb::class_<Translation3<double>>& cls) {
  using namespace nb::literals;

  using T = Translation3<double>;
  using V = Variable<double>;

  cls.def(nb::init<V, V, V>(), "x"_a, "y"_a, "z"_a,
          DOC(slp, Translation3, Translation3));
  cls.def(nb::init<const VariableMatrix<double>&>(), "vector"_a,
          DOC(slp, Translation3, Translation3, 2));
  cls.def(
      "__init__",
      [](T* self, const Eigen::Vector3d& vector) { new (self) T{vector}; },
      "vector"_a, DOC(slp, Translation3, Translation3, 3));

  // Python can't enforce const references, so the components are returned by
  // value. They're handles to the same expressions.
  cls.def(
      "x", [](const T& self) { return V{self.x()}; },
      DOC(slp, Translation3, x));
  cls.def(
      "y", [](const T& self) { return V{self.y()}; },
      DOC(slp, Translation3, y));
  cls.def(
      "z", [](const T& self) { return V{self.z()}; },
      DOC(slp, Translation3, z));

  cls.def("value", &T::value, DOC(slp, Translation3, value));
  cls.def("set_value", &T::set_value, "value"_a,
          DOC(slp, Translation3, set_value));

  cls.def("distance", &T::distance, "other"_a,
          DOC(slp, Translation3, distance));
  cls.def("squared_distance", &T::squared_distance, "other"_a,
          DOC(slp, Translation3, squared_distance));
  cls.def("rotate_by", &T::rotate_by, "rotation"_a,
          DOC(slp, Translation3, rotate_by));
  cls.def("transform_by", &T::transform_by, "pose"_a,
          DOC(slp, Translation3, transform_by));

  cls.def(
      "__add__", [](const T& lhs, const T& rhs) { return lhs + rhs; },
      nb::is_operator(), "rhs"_a, "Adds two translations.");
  cls.def(
      "__sub__", [](const T& lhs, const T& rhs) { return lhs - rhs; },
      nb::is_operator(), "rhs"_a, "Subtracts two translations.");
  cls.def(
      "__neg__", [](const T& self) { return -self; }, "Negates a translation.");
  cls.def(
      "__mul__", [](const T& lhs, const V& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a, "Scales a translation.");
  cls.def(
      "__rmul__", [](const T& rhs, const V& lhs) { return lhs * rhs; },
      nb::is_operator(), "lhs"_a, "Scales a translation.");
  cls.def(
      "__truediv__", [](const T& lhs, const V& rhs) { return lhs / rhs; },
      nb::is_operator(), "rhs"_a, "Divides a translation by a scalar.");
  cls.def(
      "__eq__", [](const T& lhs, const T& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a,
      "Equality operator that returns an equality constraint for two "
      "translations.");
}

}  // namespace slp
