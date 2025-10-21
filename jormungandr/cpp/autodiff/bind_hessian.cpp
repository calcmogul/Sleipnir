// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/hessian.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_hessian(nb::class_<Hessian<double>>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<Variable<double>, Variable<double>>(), "variable"_a, "wrt"_a,
          DOC(slp, Hessian, Hessian));
  cls.def(nb::init<Variable<double>, VariableMatrix<double>>(), "variable"_a,
          "wrt"_a, DOC(slp, Hessian, Hessian));
  cls.def("get", &Hessian<double>::get, DOC(slp, Hessian, get));
  cls.def("value", &Hessian<double>::value, DOC(slp, Hessian, value));
}

}  // namespace slp
