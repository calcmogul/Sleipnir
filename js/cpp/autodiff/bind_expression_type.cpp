// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/autodiff/expression_type.hpp>

namespace em = emscripten;

namespace slp {

void bind_expression_type(em::enum_<ExpressionType>& e) {
  e.value("NONE", ExpressionType::NONE);
  e.value("CONSTANT", ExpressionType::CONSTANT);
  e.value("LINEAR", ExpressionType::LINEAR);
  e.value("QUADRATIC", ExpressionType::QUADRATIC);
  e.value("NONLINEAR", ExpressionType::NONLINEAR);
}

}  // namespace slp
