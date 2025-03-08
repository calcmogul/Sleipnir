// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/augmented_lagrangian.hpp"

namespace slp {

template SLEIPNIR_DLLEXPORT ExitStatus augmented_lagrangian(
    const AugmentedLagrangianMatrixCallbacks<double>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options& options, Eigen::Vector<double, Eigen::Dynamic>& x);

}  // namespace slp
