// Copyright (c) Sleipnir contributors

#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "explicit_double.hpp"

template class slp::VariableMatrix<ExplicitDouble>;

template slp::VariableMatrix<ExplicitDouble> slp::solve(
    const slp::VariableMatrix<ExplicitDouble>& A,
    const slp::VariableMatrix<ExplicitDouble>& B);
