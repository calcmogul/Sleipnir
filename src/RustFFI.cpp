// Copyright (c) Sleipnir contributors

#include "RustFFI.hpp"

#include "sleipnir/src/lib.rs.h"

namespace sleipnir {

std::unique_ptr<Variable> variable_new() {
  return std::make_unique<Variable>();
}

std::unique_ptr<Variable> variable_new_with_value(double value) {
  return std::make_unique<Variable>(value);
}

}  // namespace sleipnir
