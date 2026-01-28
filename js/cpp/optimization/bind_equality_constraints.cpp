// Copyright (c) Sleipnir contributors

#include <vector>

#include <emscripten/bind.h>
#include <sleipnir/autodiff/variable.hpp>

namespace em = emscripten;

namespace slp {

void bind_equality_constraints(em::class_<EqualityConstraints<double>>& cls) {
  cls.constructor<const std::vector<EqualityConstraints<double>>&>();
  cls.function(
      "__bool__",
      [](EqualityConstraints<double>& self) -> bool { return self; },
      em::is_operator());
}

}  // namespace slp
