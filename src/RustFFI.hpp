// Copyright (c) Sleipnir contributors

#pragma once

#include <memory>

#include <rust/cxx.h>

#include "sleipnir/autodiff/Variable.hpp"

namespace sleipnir {

using sleipnir::Variable;

std::unique_ptr<Variable> variable_new();
std::unique_ptr<Variable> variable_new_with_value(double value);

}  // namespace sleipnir
