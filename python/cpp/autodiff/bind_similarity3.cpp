// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/similarity3.hpp>
#include <sleipnir/autodiff/similarity_twist3.hpp>
#include <sleipnir/autodiff/translation3.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_similarity3(nb::class_<Similarity3<double>>& cls) {
  using namespace nb::literals;

  using S = Similarity3<double>;

  cls.def(nb::init<Translation3<double>, Rotation3<double>, Real<double>>(),
          "translation"_a, "rotation"_a, "log_scale"_a,
          DOC(slp, Similarity3, Similarity3, 2));
  cls.def(
      "__init__",
      [](S* self, const Eigen::Vector3d& translation,
         const Rotation3<double>& rotation, const Real<double>& log_scale) {
        new (self) S{Translation3<double>{translation}, rotation, log_scale};
      },
      "translation"_a, "rotation"_a, "log_scale"_a,
      DOC(slp, Similarity3, Similarity3, 2));

  // Python can't enforce const references, so the parts are returned by value.
  // They're handles to the same expressions, so setting their values sets the
  // similarity transform's value.
  cls.def(
      "translation",
      [](const S& self) { return Translation3<double>{self.translation()}; },
      DOC(slp, Similarity3, translation));
  cls.def(
      "rotation",
      [](const S& self) { return Rotation3<double>{self.rotation()}; },
      DOC(slp, Similarity3, rotation));
  cls.def(
      "log_scale", [](const S& self) { return Real<double>{self.log_scale()}; },
      DOC(slp, Similarity3, log_scale));
  cls.def("scale", &S::scale, DOC(slp, Similarity3, scale));

  cls.def(
      "__mul__", [](const S& lhs, const S& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 14));
  cls.def(
      "__eq__", [](const S& lhs, const S& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 14));
  cls.def("inverse", &S::inverse, DOC(slp, Similarity3, inverse));
  cls.def("log", &S::log, DOC(slp, Similarity3, log));
  cls.def("relative_to", &S::relative_to, "other"_a,
          DOC(slp, Similarity3, relative_to));
  cls.def("to_matrix", &S::to_matrix, DOC(slp, Similarity3, to_matrix));
}

}  // namespace slp
