// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/translation3.hpp>
#include <sleipnir/autodiff/twist3.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_pose3(nb::class_<Pose3<double>>& cls) {
  using namespace nb::literals;

  using P = Pose3<double>;

  cls.def(nb::init<Translation3<double>, Rotation3<double>>(), "translation"_a,
          "rotation"_a, DOC(slp, Pose3, Pose3, 2));
  cls.def(
      "__init__",
      [](P* self, const Eigen::Vector3d& translation,
         const Rotation3<double>& rotation) {
        new (self) P{Translation3<double>{translation}, rotation};
      },
      "translation"_a, "rotation"_a, DOC(slp, Pose3, Pose3, 2));

  // Python can't enforce const references, so the parts are returned by value.
  // They're handles to the same expressions, so setting their values sets the
  // pose's value.
  cls.def(
      "translation",
      [](const P& self) { return Translation3<double>{self.translation()}; },
      DOC(slp, Pose3, translation));
  cls.def(
      "rotation",
      [](const P& self) { return Rotation3<double>{self.rotation()}; },
      DOC(slp, Pose3, rotation));

  cls.def(
      "__mul__", [](const P& lhs, const P& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a,
      "Composes two poses.\n\n"
      "lhs * rhs transforms by rhs, then by lhs. If rhs is a pose in lhs's "
      "frame, the result is that pose in lhs's parent frame.");
  cls.def(
      "__eq__", [](const P& lhs, const P& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a,
      "Equality operator that returns an equality constraint for two poses.\n\n"
      "The constraint is that the translations are equal and the rotations are "
      "equal.");
  cls.def("inverse", &P::inverse, DOC(slp, Pose3, inverse));
  cls.def("log", &P::log, DOC(slp, Pose3, log));
  cls.def("relative_to", &P::relative_to, "other"_a,
          DOC(slp, Pose3, relative_to));
  cls.def("to_matrix", &P::to_matrix, DOC(slp, Pose3, to_matrix));
}

}  // namespace slp
