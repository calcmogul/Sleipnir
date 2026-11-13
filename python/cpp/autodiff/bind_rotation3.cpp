// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_rotation3(nb::class_<Rotation3<double>>& cls) {
  using namespace nb::literals;

  using UQ = Rotation3<double>;

  cls.def(
      "__init__",
      [](UQ* self, double w, double x, double y, double z) {
        new (self) UQ{Eigen::Quaterniond{w, x, y, z}};
      },
      "w"_a, "x"_a, "y"_a, "z"_a, DOC(slp, Rotation3, Rotation3));
  cls.def(nb::init<const VariableMatrix<double>&>(), "rvec"_a,
          DOC(slp, Rotation3, Rotation3, 3));
  cls.def(
      "__init__",
      [](UQ* self, const Eigen::Vector3d& rvec) {
        new (self) UQ{VariableMatrix<double>{rvec}};
      },
      "rvec"_a, DOC(slp, Rotation3, Rotation3, 3));

  // Python can't enforce const references, so the components are returned by
  // value. They're handles to the same expressions.
  cls.def(
      "w", [](const UQ& self) { return Variable<double>{self.w()}; },
      DOC(slp, Rotation3, w));
  cls.def(
      "x", [](const UQ& self) { return Variable<double>{self.x()}; },
      DOC(slp, Rotation3, x));
  cls.def(
      "y", [](const UQ& self) { return Variable<double>{self.y()}; },
      DOC(slp, Rotation3, y));
  cls.def(
      "z", [](const UQ& self) { return Variable<double>{self.z()}; },
      DOC(slp, Rotation3, z));

  cls.def(
      "value",
      [](UQ& self) {
        auto q = self.value();
        return Eigen::Vector4d{q.w(), q.x(), q.y(), q.z()};
      },
      "Returns the rotation's value as the unit quaternion [w, x, y, z].");
  cls.def(
      "set_value",
      [](UQ& self, double w, double x, double y, double z) {
        self.set_value(Eigen::Quaterniond{w, x, y, z});
      },
      "w"_a, "x"_a, "y"_a, "z"_a, DOC(slp, Rotation3, set_value));

  cls.def(
      "__mul__", [](const UQ& lhs, const UQ& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a,
      "Composes two rotations with the Hamilton product.\n\n"
      "lhs * rhs rotates by rhs, then by lhs.");
  cls.def(
      "__eq__", [](const UQ& lhs, const UQ& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a,
      "Equality operator that returns an equality constraint for two "
      "rotations.\n\n"
      "The constraint is that the rotation vector of lhs relative to rhs is "
      "zero. Its only solution is lhs = rhs, and it's well-conditioned for "
      "rotations up to π apart.");
  cls.def("relative_to", &UQ::relative_to, "other"_a,
          DOC(slp, Rotation3, relative_to));
  cls.def("inverse", &UQ::inverse, DOC(slp, Rotation3, inverse));
  cls.def("to_matrix", &UQ::to_matrix, DOC(slp, Rotation3, to_matrix));
  cls.def("to_vector", &UQ::to_vector, DOC(slp, Rotation3, to_vector));
}

}  // namespace slp
