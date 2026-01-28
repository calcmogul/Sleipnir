// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/autodiff/hessian.hpp>

namespace em = emscripten;

namespace slp {

void bind_hessian(em::class_<Hessian<double>>& cls) {
  cls.constructor<Variable<double>, Variable<double>>();
  cls.constructor<Variable<double>, VariableMatrix<double>>();
  cls.function("get", &Hessian<double>::get);
  cls.function("value", &Hessian<double>::value);
}

}  // namespace slp
