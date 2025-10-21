// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/jacobian.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_jacobian(nb::class_<Jacobian<double>>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<Variable<double>, Variable<double>>(), "variable"_a, "wrt"_a,
          DOC(slp, Jacobian, Jacobian));
  cls.def(nb::init<Variable<double>, VariableMatrix<double>>(), "variable"_a,
          "wrt"_a, DOC(slp, Jacobian, Jacobian));
  cls.def(nb::init<VariableMatrix<double>, VariableMatrix<double>>(),
          "variables"_a, "wrt"_a, DOC(slp, Jacobian, Jacobian));
  cls.def("get", &Jacobian<double>::get, DOC(slp, Jacobian, get));
  cls.def("value", &Jacobian<double>::value, DOC(slp, Jacobian, value));
}

}  // namespace slp
