// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <sleipnir/optimization/SolverIterationInfo.hpp>

#include "Docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindSolverIterationInfo(nb::class_<SolverIterationInfo>& cls) {
  cls.def_ro("iteration", &SolverIterationInfo::iteration,
             DOC(sleipnir, SolverIterationInfo, iteration));
  cls.def_prop_ro(
      "x", [](const SolverIterationInfo& self) { return self.x; },
      DOC(sleipnir, SolverIterationInfo, x));
  cls.def_prop_ro(
      "g",
      [](const SolverIterationInfo& self) {
        return Eigen::SparseMatrix<double>{self.g};
      },
      DOC(sleipnir, SolverIterationInfo, g));
  cls.def_prop_ro(
      "H", [](const SolverIterationInfo& self) { return self.H; },
      DOC(sleipnir, SolverIterationInfo, H));
  cls.def_prop_ro(
      "A_i", [](const SolverIterationInfo& self) { return self.A_i; },
      DOC(sleipnir, SolverIterationInfo, A_i));
}

}  // namespace sleipnir
