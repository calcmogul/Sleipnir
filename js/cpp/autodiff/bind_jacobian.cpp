// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/autodiff/jacobian.hpp>

namespace em = emscripten;

namespace slp {

void bind_jacobian(em::class_<Jacobian<double>>& cls) {
  cls.constructor<Variable<double>, Variable<double>>();
  cls.constructor<Variable<double>, VariableMatrix<double>>();
  cls.constructor<VariableMatrix<double>, VariableMatrix<double>>();
  cls.function("get", &Jacobian<double>::get);
  cls.function("value", &Jacobian<double>::value);
}

}  // namespace slp
