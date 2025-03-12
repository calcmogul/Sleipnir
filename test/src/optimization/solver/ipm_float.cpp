// Copyright (c) Sleipnir contributors

#include <sleipnir/optimization/solver/ipm.hpp>

template slp::ExitStatus slp::ipm(
    const IPMMatrixCallbacks<float>& matrix_callbacks, bool is_nlp,
    std::span<std::function<bool(const IterationInfo<float>& info)>>
        iteration_callbacks,
    const Options<float>& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::Vector<float, Eigen::Dynamic>& x);
