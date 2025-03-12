// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/ipm.hpp>
#include <sleipnir/optimization/solver/newton.hpp>
#include <sleipnir/optimization/solver/sqp.hpp>

#include "explicit_double.hpp"

// SCALAR_TYPES is defined by CMake from SLEIPNIR_TEST_SCALAR_TYPES

extern template class slp::Gradient<float>;

extern template class slp::Hessian<float, Eigen::Lower | Eigen::Upper>;

extern template class slp::Jacobian<float>;

extern template class slp::VariableMatrix<float>;

extern template slp::VariableMatrix<float> slp::solve(
    const slp::VariableMatrix<float>& A, const slp::VariableMatrix<float>& B);

extern template class slp::OCP<float>;

extern template slp::ExitStatus slp::ipm(
    const IPMMatrixCallbacks<float>& matrix_callbacks, bool is_nlp,
    std::span<std::function<bool(const IterationInfo<float>& info)>>
        iteration_callbacks,
    const Options<float>& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::Vector<float, Eigen::Dynamic>& x);

extern template slp::ExitStatus slp::newton(
    const NewtonMatrixCallbacks<float>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<float>& info)>>
        iteration_callbacks,
    const Options<float>& options, Eigen::Vector<float, Eigen::Dynamic>& x);

extern template slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<float>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<float>& info)>>
        iteration_callbacks,
    const Options<float>& options, Eigen::Vector<float, Eigen::Dynamic>& x);

extern template class slp::Problem<float>;

extern template class slp::Gradient<ExplicitDouble>;

extern template class slp::Hessian<ExplicitDouble, Eigen::Lower | Eigen::Upper>;

extern template class slp::Jacobian<ExplicitDouble>;

extern template class slp::VariableMatrix<ExplicitDouble>;

extern template slp::VariableMatrix<ExplicitDouble> slp::solve(
    const slp::VariableMatrix<ExplicitDouble>& A,
    const slp::VariableMatrix<ExplicitDouble>& B);

extern template class slp::OCP<ExplicitDouble>;

extern template slp::ExitStatus slp::ipm(
    const IPMMatrixCallbacks<ExplicitDouble>& matrix_callbacks, bool is_nlp,
    std::span<std::function<bool(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const Options<ExplicitDouble>& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorX<ExplicitDouble>& x);

extern template slp::ExitStatus slp::newton(
    const NewtonMatrixCallbacks<ExplicitDouble>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const Options<ExplicitDouble>& options, Eigen::VectorX<ExplicitDouble>& x);

extern template slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<ExplicitDouble>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const Options<ExplicitDouble>& options, Eigen::VectorX<ExplicitDouble>& x);

extern template class slp::Problem<ExplicitDouble>;
