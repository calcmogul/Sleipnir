// Copyright (c) Sleipnir contributors

#pragma once

#include <emscripten/bind.h>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_block.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/optimization/ocp/dynamics_type.hpp>
#include <sleipnir/optimization/ocp/timestep_method.hpp>
#include <sleipnir/optimization/ocp/transcription_method.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>

namespace em = emscripten;

namespace slp {

void bind_expression_type(em::enum_<ExpressionType>& e);

void bind_variable(em::module_& autodiff, em::class_<Variable<double>>& cls);
void bind_variable_matrix(em::module_& autodiff,
                          em::class_<VariableMatrix<double>>& cls);
void bind_variable_block(
    em::class_<VariableBlock<VariableMatrix<double>>>& cls);

void bind_gradient(em::class_<Gradient<double>>& cls);
void bind_hessian(em::class_<Hessian<double>>& cls);
void bind_jacobian(em::class_<Jacobian<double>>& cls);

void bind_equality_constraints(em::class_<EqualityConstraints<double>>& cls);
void bind_inequality_constraints(
    em::class_<InequalityConstraints<double>>& cls);

void bind_exit_status(em::enum_<ExitStatus>& e);
void bind_iteration_info(em::class_<IterationInfo<double>>& cls);

void bind_problem(em::class_<Problem<double>>& cls);

void bind_dynamics_type(em::enum_<DynamicsType>& e);
void bind_timestep_method(em::enum_<TimestepMethod>& e);
void bind_transcription_method(em::enum_<TranscriptionMethod>& e);

void bind_ocp(em::class_<OCP<double>, Problem<double>>& cls);

}  // namespace slp
