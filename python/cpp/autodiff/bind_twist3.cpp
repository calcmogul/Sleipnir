// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/twist3.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_twist3(nb::class_<Twist3<double>>& cls) {
  using namespace nb::literals;

  using T = Twist3<double>;
  using V = Variable<double>;

  cls.def(nb::init<VariableMatrix<double>, VariableMatrix<double>>(),
          "linear"_a, "angular"_a, DOC(slp, Twist3, Twist3));
  cls.def(
      "__init__",
      [](T* self, const Eigen::Vector3d& linear,
         const Eigen::Vector3d& angular) {
        new (self)
            T{VariableMatrix<double>{linear}, VariableMatrix<double>{angular}};
      },
      "linear"_a, "angular"_a, DOC(slp, Twist3, Twist3));

  // Python can't enforce const references, so the parts are returned by value.
  // They're handles to the same expressions.
  cls.def(
      "linear",
      [](const T& self) { return VariableMatrix<double>{self.linear()}; },
      DOC(slp, Twist3, linear));
  cls.def(
      "angular",
      [](const T& self) { return VariableMatrix<double>{self.angular()}; },
      DOC(slp, Twist3, angular));

  cls.def("exp", &T::exp, DOC(slp, Twist3, exp));

  cls.def(
      "__mul__", [](const T& lhs, const V& rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a, "Scales a twist.");
  cls.def(
      "__rmul__", [](const T& rhs, const V& lhs) { return lhs * rhs; },
      nb::is_operator(), "lhs"_a, "Scales a twist.");
  cls.def(
      "__add__", [](const T& lhs, const T& rhs) { return lhs + rhs; },
      nb::is_operator(), "rhs"_a, "Adds two twists.");
  cls.def(
      "__sub__", [](const T& lhs, const T& rhs) { return lhs - rhs; },
      nb::is_operator(), "rhs"_a, "Subtracts two twists.");
  cls.def("__neg__", [](const T& self) { return -self; }, "Negates a twist.");
  cls.def(
      "__eq__", [](const T& lhs, const T& rhs) { return lhs == rhs; },
      nb::is_operator(), "rhs"_a,
      "Equality operator that returns an equality constraint for two "
      "twists.\n\n"
      "The constraint is that the linear parts are equal and the angular parts "
      "are equal.");
}

}  // namespace slp
