// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/autodiff/gradient.hpp>

namespace em = emscripten;

namespace slp {

void bind_gradient(em::class_<Gradient<double>>& cls) {
  using namespace em::literals;

  cls.function(em::init<Variable<double>, Variable<double>>(), "variable"_a,
               "wrt"_a);
  cls.function(em::init<Variable<double>, VariableMatrix<double>>(),
               "variable"_a, "wrt"_a);
  cls.function("get", &Gradient<double>::get);
  cls.function("value", [](Gradient<double>& self) {
    return Eigen::SparseMatrix<double>{self.value()};
  });
}

}  // namespace slp
