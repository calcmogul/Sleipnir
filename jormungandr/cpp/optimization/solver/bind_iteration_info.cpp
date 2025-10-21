// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <sleipnir/optimization/solver/iteration_info.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_iteration_info(nb::class_<IterationInfo<double>>& cls) {
  cls.def_ro("iteration", &IterationInfo<double>::iteration,
             DOC(slp, IterationInfo, iteration));
  cls.def_prop_ro(
      "x", [](const IterationInfo<double>& self) { return self.x; },
      DOC(slp, IterationInfo, x));
  cls.def_prop_ro(
      "g",
      [](const IterationInfo<double>& self) {
        return Eigen::SparseMatrix<double>{self.g};
      },
      DOC(slp, IterationInfo, g));
  cls.def_prop_ro(
      "H", [](const IterationInfo<double>& self) { return self.H; },
      DOC(slp, IterationInfo, H));
  cls.def_prop_ro(
      "A_e", [](const IterationInfo<double>& self) { return self.A_e; },
      DOC(slp, IterationInfo, A_e));
  cls.def_prop_ro(
      "A_i", [](const IterationInfo<double>& self) { return self.A_i; },
      DOC(slp, IterationInfo, A_i));
}

}  // namespace slp
