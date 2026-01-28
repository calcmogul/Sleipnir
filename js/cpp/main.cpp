// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
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
#include "for_each_type.hpp"

namespace em = emscripten;

namespace slp {

EMSCRIPTEN_BINDINGS(sleipnir) {
  em::module_ autodiff = sleipnir.def_submodule("autodiff");
  em::module_ optimization = sleipnir.def_submodule("optimization");

  em::enum_<ExpressionType> expression_type{autodiff, "ExpressionType"};

  em::class_<Variable<double>> variable{autodiff, "Variable"};
  em::class_<VariableMatrix<double>> variable_matrix{autodiff,
                                                     "VariableMatrix"};
  em::class_<VariableBlock<VariableMatrix<double>>> variable_block{
      autodiff, "VariableBlock"};

  em::class_<Gradient<double>> gradient{autodiff, "Gradient"};
  em::class_<Hessian<double>> hessian{autodiff, "Hessian"};
  em::class_<Jacobian<double>> jacobian{autodiff, "Jacobian"};

  em::class_<EqualityConstraints<double>> equality_constraints{
      optimization, "EqualityConstraints"};
  em::class_<InequalityConstraints<double>> inequality_constraints{
      optimization, "InequalityConstraints"};

  // Bounds function
  for_each_type<
      double, int, const Variable<double>&, const VariableMatrix<double>&,
      const VariableBlock<VariableMatrix<double>>&, em::DRef<Eigen::MatrixXd>>(
      [&]<typename L> {
        for_each_type<const Variable<double>&, const VariableMatrix<double>&,
                      const VariableBlock<VariableMatrix<double>>&>(
            [&]<typename X> {
              for_each_type<double, int, const Variable<double>&,
                            const VariableMatrix<double>&,
                            const VariableBlock<VariableMatrix<double>>&,
                            em::DRef<Eigen::MatrixXd>>([&]<typename U> {
                optimization.function("bounds", &bounds<L&&, X&&, U&&>);
              });
            });
      });

  em::enum_<ExitStatus> exit_status{optimization, "ExitStatus"};
  em::class_<IterationInfo<double>> iteration_info{optimization,
                                                   "IterationInfo"};

  em::class_<Problem<double>> problem{optimization, "Problem"};

  em::enum_<DynamicsType> dynamics_type{optimization, "DynamicsType"};
  em::enum_<TimestepMethod> timestep_method{optimization, "TimestepMethod"};
  em::enum_<TranscriptionMethod> transcription_method{optimization,
                                                      "TranscriptionMethod"};

  em::class_<OCP<double>, Problem<double>> ocp{optimization, "OCP"};

  bind_expression_type(expression_type);

  bind_variable(autodiff, variable);
  bind_variable_matrix(autodiff, variable_matrix);
  bind_variable_block(variable_block);

  // Implicit conversions
  variable_matrix.function(
      em::init_implicit<VariableBlock<VariableMatrix<double>>>());

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
