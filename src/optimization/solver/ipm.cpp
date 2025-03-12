// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/ipm.hpp"

#include <functional>
#include <span>

#include <Eigen/Core>

#include "sleipnir/optimization/solver/control_flow.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/ipm_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/util/iterates.hpp"
#include "sleipnir/util/symbol_exports.hpp"

template SLEIPNIR_DLLEXPORT slp::ExitStatus slp::ipm(
    const IPMMatrixCallbacks<double>& matrix_callbacks, bool is_nlp,
    std::span<std::function<ControlFlow(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options<double>& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorX<double>& x);

template SLEIPNIR_DLLEXPORT slp::ExitStatus slp::ipm(
    const IPMMatrixCallbacks<double>& matrix_callbacks, bool is_nlp,
    std::span<std::function<ControlFlow(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options<double>& options, bool in_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    IPMIterates<double>& iterates, double& sqrt_μ, int& iterations);
