// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/similarity3.hpp>
#include <sleipnir/autodiff/translation3.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_translation3(nb::class_<Translation3<double>>& cls) {
  using namespace nb::literals;

  using T = Translation3<double>;
  using V = Real<double>;

  cls.def(nb::init<V, V, V>(), "x"_a, "y"_a, "z"_a,
          DOC(slp, Translation3, Translation3));
  cls.def(nb::init<const RealMatrix<double>&>(), "vector"_a,
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
  cls.def("transform_by",
          nb::overload_cast<const Pose3<double>&>(&T::transform_by, nb::const_),
          "pose"_a, DOC(slp, Translation3, transform_by));
  cls.def("transform_by",
          nb::overload_cast<const Similarity3<double>&>(&T::transform_by,
                                                        nb::const_),
          "similarity"_a, DOC(slp, Translation3, transform_by, 2));

  cls.def(
      "__add__", [](const T& lhs, const T& rhs) { return lhs + rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, add, 14));
  cls.def(
      "__sub__", [](const T& lhs, const T& rhs) { return lhs - rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, sub, 16));
  cls.def(
      "__neg__", [](const T& self) { return -self; },
      DOC(slp, operator, sub, 17));
  cls.def(
      "__mul__", [](const T& lhs, const V& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 25));
  cls.def(
      "__rmul__", [](const T& rhs, const V& lhs) { return lhs * rhs; },
      nb::is_operator(), "lhs"_a, DOC(slp, operator, mul, 26));
  cls.def(
      "__truediv__", [](const T& lhs, const V& rhs) { return lhs / rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, div, 8));
  cls.def(
      "__eq__", [](const T& lhs, const T& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 22));
}

}  // namespace slp
