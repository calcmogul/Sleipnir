// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/optimization/solver/exit_status.hpp>

namespace em = emscripten;

namespace slp {

void bind_exit_status(em::enum_<ExitStatus>& e) {
  e.value("SUCCESS", ExitStatus::SUCCESS);
  e.value("CALLBACK_REQUESTED_STOP", ExitStatus::CALLBACK_REQUESTED_STOP);
  e.value("TOO_FEW_DOFS", ExitStatus::TOO_FEW_DOFS);
  e.value("LOCALLY_INFEASIBLE", ExitStatus::LOCALLY_INFEASIBLE);
  e.value("GLOBALLY_INFEASIBLE", ExitStatus::GLOBALLY_INFEASIBLE);
  e.value("FACTORIZATION_FAILED", ExitStatus::FACTORIZATION_FAILED);
  e.value("LINE_SEARCH_FAILED", ExitStatus::LINE_SEARCH_FAILED);
  e.value("NONFINITE_INITIAL_COST_OR_CONSTRAINTS",
          ExitStatus::NONFINITE_INITIAL_COST_OR_CONSTRAINTS);
  e.value("DIVERGING_ITERATES", ExitStatus::DIVERGING_ITERATES);
  e.value("MAX_ITERATIONS_EXCEEDED", ExitStatus::MAX_ITERATIONS_EXCEEDED);
  e.value("TIMEOUT", ExitStatus::TIMEOUT);
}

}  // namespace slp
