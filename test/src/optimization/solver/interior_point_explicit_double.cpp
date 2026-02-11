// Copyright (c) Sleipnir contributors

#include <sleipnir/optimization/solver/interior_point.hpp>

#include "explicit_double.hpp"

template slp::ExitStatus slp::interior_point(
    const InteriorPointMatrixCallbacks<ExplicitDouble>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const Options& options, bool use_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::Vector<ExplicitDouble, Eigen::Dynamic>& x,
    Eigen::Vector<ExplicitDouble, Eigen::Dynamic>& s);
