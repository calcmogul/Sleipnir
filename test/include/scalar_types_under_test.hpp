// Copyright (c) Sleipnir contributors

#pragma once

#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/interior_point.hpp>
#include <sleipnir/optimization/solver/newton.hpp>
#include <sleipnir/optimization/solver/sqp.hpp>

#include "explicit_double.hpp"

extern template class slp::Gradient<ExplicitDouble>;

extern template class slp::Hessian<ExplicitDouble, Eigen::Lower | Eigen::Upper>;

extern template class slp::Jacobian<ExplicitDouble>;

extern template slp::VariableMatrix<ExplicitDouble> slp::solve(
    const slp::VariableMatrix<ExplicitDouble>& A,
    const slp::VariableMatrix<ExplicitDouble>& B);

extern template class slp::OCP<ExplicitDouble>;

extern template slp::ExitStatus slp::interior_point(
    const InteriorPointMatrixCallbacks<ExplicitDouble>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const Options& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    int num_equality_constraints, int num_inequality_constraints,
    Eigen::Vector<ExplicitDouble, Eigen::Dynamic>& x);

extern template slp::ExitStatus slp::newton(
    const NewtonMatrixCallbacks<ExplicitDouble>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const Options& options, Eigen::Vector<ExplicitDouble, Eigen::Dynamic>& x);

extern template slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<ExplicitDouble>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const Options& options, Eigen::Vector<ExplicitDouble, Eigen::Dynamic>& x);

extern template class slp::Problem<ExplicitDouble>;

#define SCALAR_TYPES_UNDER_TEST double, ExplicitDouble
