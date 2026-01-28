// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/optimization/ocp/dynamics_type.hpp>

namespace em = emscripten;

namespace slp {

void bind_dynamics_type(em::enum_<DynamicsType>& e) {
  e.value("EXPLICIT_ODE", DynamicsType::EXPLICIT_ODE);
  e.value("DISCRETE", DynamicsType::DISCRETE);
}

}  // namespace slp
