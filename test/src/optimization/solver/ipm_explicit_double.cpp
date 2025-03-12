// Copyright (c) Sleipnir contributors

#include <functional>
#include <span>

#include <Eigen/Core>
#include <sleipnir/optimization/solver/control_flow.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/ipm.hpp>  // IWYU pragma: keep
#include <sleipnir/optimization/solver/ipm_matrix_callbacks.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>
#include <sleipnir/optimization/solver/options.hpp>
#include <sleipnir/optimization/solver/util/iterates.hpp>

#include "explicit_double.hpp"

template slp::ExitStatus slp::ipm(
    const IPMMatrixCallbacks<ExplicitDouble>& matrix_callbacks, bool is_nlp,
    std::span<
        std::function<ControlFlow(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const Options<ExplicitDouble>& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorX<ExplicitDouble>& x);

template slp::ExitStatus slp::ipm(
    const IPMMatrixCallbacks<ExplicitDouble>& matrix_callbacks, bool is_nlp,
    std::span<
        std::function<ControlFlow(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const Options<ExplicitDouble>& options, bool in_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    IPMIterates<ExplicitDouble>& iterates, ExplicitDouble& sqrt_μ,
    int& iterations);
