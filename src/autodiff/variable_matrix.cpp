// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/variable_matrix.hpp"

#include "sleipnir/autodiff/pose2.hpp"
#include "sleipnir/autodiff/pose3.hpp"
#include "sleipnir/autodiff/rotation2.hpp"
#include "sleipnir/autodiff/rotation3.hpp"
#include "sleipnir/util/symbol_exports.hpp"

template class EXPORT_TEMPLATE_DEFINE(SLEIPNIR_DLLEXPORT)
slp::VariableMatrix<double>;

template slp::VariableMatrix<double> slp::solve(
    const slp::VariableMatrix<double>& A, const slp::VariableMatrix<double>& B);
