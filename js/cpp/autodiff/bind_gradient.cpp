// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/autodiff/gradient.hpp>

namespace em = emscripten;

namespace slp {

void bind_gradient(em::class_<Gradient<double>>& cls) {
  cls.constructor<Variable<double>, Variable<double>>();
  cls.constructor<Variable<double>, VariableMatrix<double>>();
  cls.function("get", &Gradient<double>::get);
  cls.function("value", [](Gradient<double>& self) {
    return Eigen::SparseMatrix<double>{self.value()};
  });
}

}  // namespace slp
