// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
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

#include "binders.hpp"
#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

NB_MODULE(_jormungandr, m) {
  m.doc() =
      "A linearity-exploiting sparse nonlinear constrained optimization "
      "problem solver that uses the interior-point method.";

  nb::module_ autodiff = m.def_submodule("autodiff");
  nb::module_ optimization = m.def_submodule("optimization");

  nb::enum_<ExpressionType> expression_type{autodiff, "ExpressionType",
                                            DOC(slp, ExpressionType)};

  nb::class_<Variable<double>> variable{autodiff, "Variable",
                                        DOC(slp, Variable)};
  nb::class_<VariableMatrix<double>> variable_matrix{autodiff, "VariableMatrix",
                                                     DOC(slp, VariableMatrix)};
  nb::class_<VariableBlock<VariableMatrix<double>>> variable_block{
      autodiff, "VariableBlock", DOC(slp, VariableBlock)};

  nb::class_<Gradient<double>> gradient{autodiff, "Gradient",
                                        DOC(slp, Gradient)};
  nb::class_<Hessian<double>> hessian{autodiff, "Hessian", DOC(slp, Hessian)};
  nb::class_<Jacobian<double>> jacobian{autodiff, "Jacobian",
                                        DOC(slp, Jacobian)};

  nb::class_<EqualityConstraints<double>> equality_constraints{
      optimization, "EqualityConstraints", DOC(slp, EqualityConstraints)};
  nb::class_<InequalityConstraints<double>> inequality_constraints{
      optimization, "InequalityConstraints", DOC(slp_InequalityConstraints)};

  nb::enum_<ExitStatus> exit_status{optimization, "ExitStatus",
                                    DOC(slp, ExitStatus)};
  nb::class_<IterationInfo<double>> iteration_info{
      optimization, "IterationInfo", DOC(slp, IterationInfo)};

  nb::class_<Problem<double>> problem{optimization, "Problem",
                                      DOC(slp, Problem)};

  nb::enum_<DynamicsType> dynamics_type{optimization, "DynamicsType",
                                        DOC(slp, DynamicsType)};
  nb::enum_<TimestepMethod> timestep_method{optimization, "TimestepMethod",
                                            DOC(slp, TimestepMethod)};
  nb::enum_<TranscriptionMethod> transcription_method{
      optimization, "TranscriptionMethod", DOC(slp, TranscriptionMethod)};

  nb::class_<OCP<double>, Problem<double>> ocp{optimization, "OCP",
                                               DOC(slp, OCP)};

  bind_expression_type(expression_type);

  bind_variable(autodiff, variable);
  bind_variable_matrix(autodiff, variable_matrix);
  bind_variable_block(variable_block);

  // Implicit conversions
  variable.def(nb::init_implicit<VariableMatrix<double>>());
  variable_matrix.def(
      nb::init_implicit<VariableBlock<VariableMatrix<double>>>());

  bind_gradient(gradient);
  bind_hessian(hessian);
  bind_jacobian(jacobian);

  bind_equality_constraints(equality_constraints);
  bind_inequality_constraints(inequality_constraints);

  bind_exit_status(exit_status);
  bind_iteration_info(iteration_info);

  bind_problem(problem);

  bind_dynamics_type(dynamics_type);
  bind_timestep_method(timestep_method);
  bind_transcription_method(transcription_method);

  bind_ocp(ocp);
}

}  // namespace slp
