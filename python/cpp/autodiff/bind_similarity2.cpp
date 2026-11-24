// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/similarity2.hpp>
#include <sleipnir/autodiff/similarity_twist2.hpp>
#include <sleipnir/autodiff/translation2.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_similarity2(nb::class_<Similarity2<double>>& cls) {
  using namespace nb::literals;

  using S = Similarity2<double>;

  cls.def(nb::init<const Eigen::Vector2d&, double, double>(), "translation"_a,
          "angle"_a, "scale"_a, DOC(slp, Similarity2, Similarity2));
  cls.def(nb::init<Translation2<double>, Rotation2<double>, Real<double>>(),
          "translation"_a, "rotation"_a, "log_scale"_a,
          DOC(slp, Similarity2, Similarity2, 2));
  cls.def(
      "__init__",
      [](S* self, const Eigen::Vector2d& translation,
         const Rotation2<double>& rotation, const Real<double>& log_scale) {
        new (self) S{Translation2<double>{translation}, rotation, log_scale};
      },
      "translation"_a, "rotation"_a, "log_scale"_a,
      DOC(slp, Similarity2, Similarity2, 2));

  // Python can't enforce const references, so the parts are returned by value.
  // They're handles to the same expressions, so setting their values sets the
  // similarity transform's value.
  cls.def(
      "translation",
      [](const S& self) { return Translation2<double>{self.translation()}; },
      DOC(slp, Similarity2, translation));
  cls.def(
      "rotation",
      [](const S& self) { return Rotation2<double>{self.rotation()}; },
      DOC(slp, Similarity2, rotation));
  cls.def(
      "log_scale", [](const S& self) { return Real<double>{self.log_scale()}; },
      DOC(slp, Similarity2, log_scale));
  cls.def("scale", &S::scale, DOC(slp, Similarity2, scale));

  cls.def(
      "__mul__", [](const S& lhs, const S& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 14));
  cls.def(
      "__eq__", [](const S& lhs, const S& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 14));
  cls.def("inverse", &S::inverse, DOC(slp, Similarity2, inverse));
  cls.def("log", &S::log, DOC(slp, Similarity2, log));
  cls.def("relative_to", &S::relative_to, "other"_a,
          DOC(slp, Similarity2, relative_to));
  cls.def("to_matrix", &S::to_matrix, DOC(slp, Similarity2, to_matrix));
}

}  // namespace slp
