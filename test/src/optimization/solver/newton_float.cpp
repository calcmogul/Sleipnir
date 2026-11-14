// Copyright (c) Sleipnir contributors

#include <functional>
#include <span>

#include <Eigen/Core>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>
#include <sleipnir/optimization/solver/newton.hpp>  // IWYU pragma: keep
#include <sleipnir/optimization/solver/newton_matrix_callbacks.hpp>
#include <sleipnir/optimization/solver/options.hpp>

template slp::ExitStatus slp::newton(
    const NewtonMatrixCallbacks<float>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<float>& info)>>
        iteration_callbacks,
    const Options<float>& options, Eigen::Vector<float, Eigen::Dynamic>& x);
