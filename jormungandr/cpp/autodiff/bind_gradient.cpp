// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/gradient.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_gradient(nb::class_<Gradient<double>>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<Variable<double>, Variable<double>>(), "variable"_a, "wrt"_a,
          DOC(slp, Gradient, Gradient));
  cls.def(nb::init<Variable<double>, VariableMatrix<double>>(), "variable"_a,
          "wrt"_a, DOC(slp, Gradient, Gradient, 2));
  cls.def("get", &Gradient<double>::get, DOC(slp, Gradient, get));
  cls.def(
      "value",
      [](Gradient<double>& self) {
        return Eigen::SparseMatrix<double>{self.value()};
      },
      DOC(slp, Gradient, value));
}

}  // namespace slp
