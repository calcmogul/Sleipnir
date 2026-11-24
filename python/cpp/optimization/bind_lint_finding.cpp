// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sleipnir/optimization/lint.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_lint_finding(nb::class_<LintFinding>& cls) {
  cls.def_ro("kind", &LintFinding::kind, DOC(slp, LintFinding, kind));
  cls.def_ro("decision_variables", &LintFinding::decision_variables,
             DOC(slp, LintFinding, decision_variables));
  cls.def_ro("cost", &LintFinding::cost, DOC(slp, LintFinding, cost));
  cls.def_ro("equality_constraints", &LintFinding::equality_constraints,
             DOC(slp, LintFinding, equality_constraints));
  cls.def_ro("inequality_constraints", &LintFinding::inequality_constraints,
             DOC(slp, LintFinding, inequality_constraints));
  cls.def_ro("message", &LintFinding::message, DOC(slp, LintFinding, message));
  cls.def("__str__", [](const LintFinding& self) { return self.message; });
  cls.def("__repr__", [](const LintFinding& self) {
    return nb::str("LintFinding(kind=LintKind.{}, message={!r})")
        .format(nb::cast(self.kind).attr("name"), self.message);
  });
}

}  // namespace slp
