// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>

#include <sleipnir/optimization/solver/exit_status.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_exit_status(nb::enum_<ExitStatus>& e) {
  e.value("SUCCESS", ExitStatus::SUCCESS, DOC(slp, ExitStatus, SUCCESS));
  e.value("CALLBACK_REQUESTED_STOP", ExitStatus::CALLBACK_REQUESTED_STOP,
          DOC(slp, ExitStatus, CALLBACK_REQUESTED_STOP));
  e.value("TOO_FEW_DOFS", ExitStatus::TOO_FEW_DOFS,
          DOC(slp, ExitStatus, TOO_FEW_DOFS));
  e.value("LOCALLY_INFEASIBLE", ExitStatus::LOCALLY_INFEASIBLE,
          DOC(slp, ExitStatus, LOCALLY_INFEASIBLE));
  e.value("GLOBALLY_INFEASIBLE", ExitStatus::GLOBALLY_INFEASIBLE,
          DOC(slp, ExitStatus, GLOBALLY_INFEASIBLE));
  e.value("FACTORIZATION_FAILED", ExitStatus::FACTORIZATION_FAILED,
          DOC(slp, ExitStatus, FACTORIZATION_FAILED));
  e.value("FEASIBILITY_RESTORATION_FAILED",
          ExitStatus::FEASIBILITY_RESTORATION_FAILED,
          DOC(slp, ExitStatus, FEASIBILITY_RESTORATION_FAILED));
  e.value("NONFINITE_INITIAL_COST_OR_CONSTRAINTS",
          ExitStatus::NONFINITE_INITIAL_COST_OR_CONSTRAINTS,
          DOC(slp, ExitStatus, NONFINITE_INITIAL_COST_OR_CONSTRAINTS));
  e.value("DIVERGING_ITERATES", ExitStatus::DIVERGING_ITERATES,
          DOC(slp, ExitStatus, DIVERGING_ITERATES));
  e.value("MAX_ITERATIONS_EXCEEDED", ExitStatus::MAX_ITERATIONS_EXCEEDED,
          DOC(slp, ExitStatus, MAX_ITERATIONS_EXCEEDED));
  e.value("TIMEOUT", ExitStatus::TIMEOUT, DOC(slp, ExitStatus, TIMEOUT));
}

}  // namespace slp
