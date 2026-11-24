// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/similarity2.hpp>
#include <sleipnir/autodiff/similarity_twist2.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_similarity_twist2(nb::class_<SimilarityTwist2<double>>& cls) {
  using namespace nb::literals;

  using T = SimilarityTwist2<double>;
  using V = Real<double>;

  cls.def(nb::init<RealMatrix<double>, V, V>(), "linear"_a, "angular"_a,
          "log_scale"_a, DOC(slp, SimilarityTwist2, SimilarityTwist2));
  cls.def(
      "__init__",
      [](T* self, const Eigen::Vector2d& linear, const V& angular,
         const V& log_scale) {
        new (self) T{RealMatrix<double>{linear}, angular, log_scale};
      },
      "linear"_a, "angular"_a, "log_scale"_a,
      DOC(slp, SimilarityTwist2, SimilarityTwist2));

  // Python can't enforce const references, so the parts are returned by value.
  // They're handles to the same expressions.
  cls.def(
      "linear", [](const T& self) { return RealMatrix<double>{self.linear()}; },
      DOC(slp, SimilarityTwist2, linear));
  cls.def(
      "angular", [](const T& self) { return V{self.angular()}; },
      DOC(slp, SimilarityTwist2, angular));

  cls.def(
      "log_scale", [](const T& self) { return V{self.log_scale()}; },
      DOC(slp, SimilarityTwist2, log_scale));

  cls.def("exp", &T::exp, DOC(slp, SimilarityTwist2, exp));

  cls.def(
      "__mul__", [](const T& lhs, const V& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 14));
  cls.def(
      "__rmul__", [](const T& rhs, const V& lhs) { return lhs * rhs; },
      nb::is_operator(), "lhs"_a, DOC(slp, operator, mul, 15));
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
      "__eq__", [](const T& lhs, const T& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 18));
}

}  // namespace slp
