// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/sqp.hpp"

#include <functional>
#include <span>

#include <Eigen/Core>

#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/sqp_matrix_callbacks.hpp"
#include "sleipnir/util/symbol_exports.hpp"

template SLEIPNIR_DLLEXPORT slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<double>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const slp::Options<double>& options, Eigen::VectorX<double>& x);
