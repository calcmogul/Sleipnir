// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/autodiff/twist2.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_pose2(nb::class_<Pose2<double>>& cls) {
  using namespace nb::literals;

  using P = Pose2<double>;

  cls.def(nb::init<const Eigen::Vector2d&, double>(), "translation"_a,
          "angle"_a, DOC(slp, Pose2, Pose2));
  cls.def(nb::init<Translation2<double>, Rotation2<double>>(), "translation"_a,
          "rotation"_a, DOC(slp, Pose2, Pose2, 2));
  cls.def(
      "__init__",
      [](P* self, const Eigen::Vector2d& translation,
         const Rotation2<double>& rotation) {
        new (self) P{Translation2<double>{translation}, rotation};
      },
      "translation"_a, "rotation"_a, DOC(slp, Pose2, Pose2, 2));

  // Python can't enforce const references, so the parts are returned by value.
  // They're handles to the same expressions, so setting their values sets the
  // pose's value.
  cls.def(
      "translation",
      [](const P& self) { return Translation2<double>{self.translation()}; },
      DOC(slp, Pose2, translation));
  cls.def(
      "rotation",
      [](const P& self) { return Rotation2<double>{self.rotation()}; },
      DOC(slp, Pose2, rotation));

  cls.def(
      "__mul__", [](const P& lhs, const P& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 8));
  cls.def(
      "__eq__", [](const P& lhs, const P& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 7));
  cls.def("inverse", &P::inverse, DOC(slp, Pose2, inverse));
  cls.def("log", &P::log, DOC(slp, Pose2, log));
  cls.def("relative_to", &P::relative_to, "other"_a,
          DOC(slp, Pose2, relative_to));
  cls.def("to_matrix", &P::to_matrix, DOC(slp, Pose2, to_matrix));
}

}  // namespace slp
