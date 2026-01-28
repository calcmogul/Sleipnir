// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/autodiff/jacobian.hpp>

namespace em = emscripten;

namespace slp {

void bind_jacobian(em::class_<Jacobian<double>>& cls) {
  using namespace em::literals;

  cls.function(em::init<Variable<double>, Variable<double>>(), "variable"_a,
               "wrt"_a);
  cls.function(em::init<Variable<double>, VariableMatrix<double>>(),
               "variable"_a, "wrt"_a);
  cls.function(em::init<VariableMatrix<double>, VariableMatrix<double>>(),
               "variables"_a, "wrt"_a);
  cls.function("get", &Jacobian<double>::get);
  cls.function("value", &Jacobian<double>::value);
}

}  // namespace slp
