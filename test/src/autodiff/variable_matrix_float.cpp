// Copyright (c) Sleipnir contributors

#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

template class slp::VariableMatrix<float>;

template slp::VariableMatrix<float> slp::solve(
    const slp::VariableMatrix<float>& A, const slp::VariableMatrix<float>& B);
