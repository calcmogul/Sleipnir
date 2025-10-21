// Copyright (c) Sleipnir contributors

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <sleipnir/autodiff/variable.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_inequality_constraints(
    nb::class_<InequalityConstraints<double>>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<const std::vector<InequalityConstraints<double>>&>(),
          "inequality_constraints"_a,
          DOC(slp, InequalityConstraints, InequalityConstraints, 2));
  cls.def(
      "__bool__",
      [](InequalityConstraints<double>& self) -> bool { return self; },
      nb::is_operator(), DOC(slp, InequalityConstraints, operator, bool));
}

}  // namespace slp
