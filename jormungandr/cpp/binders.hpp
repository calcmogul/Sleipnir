// Copyright (c) Sleipnir contributors

#pragma once

#include <nanobind/nanobind.h>
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

namespace nb = nanobind;

namespace slp {

void bind_expression_type(nb::enum_<ExpressionType>& e);

void bind_variable(nb::module_& autodiff, nb::class_<Variable<double>>& cls);
void bind_variable_matrix(nb::module_& autodiff,
                          nb::class_<VariableMatrix<double>>& cls);
void bind_variable_block(
    nb::class_<VariableBlock<VariableMatrix<double>>>& cls);

void bind_gradient(nb::class_<Gradient<double>>& cls);
void bind_hessian(nb::class_<Hessian<double>>& cls);
void bind_jacobian(nb::class_<Jacobian<double>>& cls);

void bind_equality_constraints(nb::class_<EqualityConstraints<double>>& cls);
void bind_inequality_constraints(
    nb::class_<InequalityConstraints<double>>& cls);

void bind_exit_status(nb::enum_<ExitStatus>& e);
void bind_iteration_info(nb::class_<IterationInfo<double>>& cls);

void bind_problem(nb::class_<Problem<double>>& cls);

void bind_dynamics_type(nb::enum_<DynamicsType>& e);
void bind_timestep_method(nb::enum_<TimestepMethod>& e);
void bind_transcription_method(nb::enum_<TranscriptionMethod>& e);

void bind_ocp(nb::class_<OCP<double>, Problem<double>>& cls);

}  // namespace slp
