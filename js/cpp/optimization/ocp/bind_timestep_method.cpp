// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/optimization/ocp/timestep_method.hpp>

namespace em = emscripten;

namespace slp {

void bind_timestep_method(em::enum_<TimestepMethod>& e) {
  e.value("FIXED", TimestepMethod::FIXED);
  e.value("VARIABLE", TimestepMethod::VARIABLE);
  e.value("VARIABLE_SINGLE", TimestepMethod::VARIABLE_SINGLE);
}

}  // namespace slp
