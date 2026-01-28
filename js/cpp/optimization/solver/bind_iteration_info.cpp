// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/optimization/solver/iteration_info.hpp>

namespace em = emscripten;

namespace slp {

void bind_iteration_info(em::class_<IterationInfo<double>>& cls) {
  cls.function("iteration", &IterationInfo<double>::iteration);
  cls.property("x", [](const IterationInfo<double>& self) { return self.x; });
  cls.property("g", [](const IterationInfo<double>& self) {
    return Eigen::SparseMatrix<double>{self.g};
  });
  cls.property("H", [](const IterationInfo<double>& self) { return self.H; });
  cls.property("A_e",
               [](const IterationInfo<double>& self) { return self.A_e; });
  cls.property("A_i",
               [](const IterationInfo<double>& self) { return self.A_i; });
}

}  // namespace slp
