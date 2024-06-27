// Copyright (c) Sleipnir contributors

#include "sleipnirrust.hpp"

#include "sleipnir/src/lib.rs.h"

namespace sleipnirrust {

std::unique_ptr<Variable> new_variable() {
  return std::make_unique<Variable>(Variable());
}

}  // namespace sleipnirrust
