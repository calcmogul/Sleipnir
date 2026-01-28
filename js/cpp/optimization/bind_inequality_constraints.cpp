// Copyright (c) Sleipnir contributors

#include <vector>

#include <emscripten/bind.h>
#include <sleipnir/autodiff/variable.hpp>

namespace em = emscripten;

namespace slp {

void bind_inequality_constraints(
    em::class_<InequalityConstraints<double>>& cls) {
  cls.constructor<const std::vector<InequalityConstraints<double>>&>();
  cls.function(
      "__bool__",
      [](InequalityConstraints<double>& self) -> bool { return self; },
      em::is_operator());
}

}  // namespace slp
