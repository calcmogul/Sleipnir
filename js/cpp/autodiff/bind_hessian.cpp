// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/autodiff/hessian.hpp>

namespace em = emscripten;

namespace slp {

void bind_hessian(em::class_<Hessian<double>>& cls) {
  using namespace em::literals;

  cls.function(em::init<Variable<double>, Variable<double>>(), "variable"_a,
               "wrt"_a);
  cls.function(em::init<Variable<double>, VariableMatrix<double>>(),
               "variable"_a, "wrt"_a);
  cls.function("get", &Hessian<double>::get);
  cls.function("value", &Hessian<double>::value);
}

}  // namespace slp
