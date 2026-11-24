// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>

#include <sleipnir/optimization/lint.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_lint_kind(nb::enum_<LintKind>& e) {
  e.value("UNUSED_DECISION_VARIABLE", LintKind::UNUSED_DECISION_VARIABLE,
          DOC(slp, LintKind, UNUSED_DECISION_VARIABLE));
  e.value("CONSTANT_CONSTRAINT", LintKind::CONSTANT_CONSTRAINT,
          DOC(slp, LintKind, CONSTANT_CONSTRAINT));
  e.value("SPHERE_CONSTRAINT", LintKind::SPHERE_CONSTRAINT,
          DOC(slp, LintKind, SPHERE_CONSTRAINT));
  e.value("REDUNDANT_CONSTRAINTS", LintKind::REDUNDANT_CONSTRAINTS,
          DOC(slp, LintKind, REDUNDANT_CONSTRAINTS));
  e.value("CONFLICTING_CONSTRAINTS", LintKind::CONFLICTING_CONSTRAINTS,
          DOC(slp, LintKind, CONFLICTING_CONSTRAINTS));
  e.value("EQUALITY_AS_INEQUALITIES", LintKind::EQUALITY_AS_INEQUALITIES,
          DOC(slp, LintKind, EQUALITY_AS_INEQUALITIES));
  e.value("NONSMOOTH_FUNCTION", LintKind::NONSMOOTH_FUNCTION,
          DOC(slp, LintKind, NONSMOOTH_FUNCTION));
  e.value("BADLY_SCALED", LintKind::BADLY_SCALED,
          DOC(slp, LintKind, BADLY_SCALED));
}

}  // namespace slp
