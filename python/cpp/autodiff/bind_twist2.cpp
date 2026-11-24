// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/twist2.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_twist2(nb::class_<Twist2<double>>& cls) {
  using namespace nb::literals;

  using T = Twist2<double>;
  using V = Real<double>;

  cls.def(nb::init<RealMatrix<double>, V>(), "linear"_a, "angular"_a,
          DOC(slp, Twist2, Twist2));
  cls.def(
      "__init__",
      [](T* self, const Eigen::Vector2d& linear, const V& angular) {
        new (self) T{RealMatrix<double>{linear}, angular};
      },
      "linear"_a, "angular"_a, DOC(slp, Twist2, Twist2));

  // Python can't enforce const references, so the parts are returned by value.
  // They're handles to the same expressions.
  cls.def(
      "linear", [](const T& self) { return RealMatrix<double>{self.linear()}; },
      DOC(slp, Twist2, linear));
  cls.def(
      "angular", [](const T& self) { return V{self.angular()}; },
      DOC(slp, Twist2, angular));

  cls.def("exp", &T::exp, DOC(slp, Twist2, exp));

  cls.def(
      "__mul__", [](const T& lhs, const V& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 24));
  cls.def(
      "__rmul__", [](const T& rhs, const V& lhs) { return lhs * rhs; },
      nb::is_operator(), "lhs"_a, DOC(slp, operator, mul, 25));
  cls.def(
      "__add__", [](const T& lhs, const T& rhs) { return lhs + rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, add, 12));
  cls.def(
      "__sub__", [](const T& lhs, const T& rhs) { return lhs - rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, sub, 16));
  cls.def(
      "__neg__", [](const T& self) { return -self; },
      DOC(slp, operator, sub, 17));
  cls.def(
      "__eq__", [](const T& lhs, const T& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 20));
}

}  // namespace slp
