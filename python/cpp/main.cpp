// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/autodiff/translation3.hpp>
#include <sleipnir/autodiff/twist2.hpp>
#include <sleipnir/autodiff/twist3.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_block.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/manifold.hpp>
#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/optimization/ocp/dynamics_type.hpp>
#include <sleipnir/optimization/ocp/timestep_method.hpp>
#include <sleipnir/optimization/ocp/transcription_method.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>

#include "binders.hpp"
#include "docstrings.hpp"
#include "for_each_type.hpp"

namespace nb = nanobind;

namespace slp {

NB_MODULE(_sleipnir, m) {
  using namespace nb::literals;

  m.doc() =
      "A linearity-exploiting reverse mode autodiff library and nonlinear "
      "program solver DSL.";

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
  nb::class_<UnitVector<double>> unit_vector{autodiff, "UnitVector",
                                             DOC(slp, UnitVector)};
  nb::class_<Rotation2<double>> rotation2{autodiff, "Rotation2",
                                          DOC(slp, Rotation2)};
  nb::class_<Rotation3<double>> rotation3{autodiff, "Rotation3",
                                          DOC(slp, Rotation3)};
  nb::class_<Pose2<double>> pose2{autodiff, "Pose2", DOC(slp, Pose2)};
  nb::class_<Pose3<double>> pose3{autodiff, "Pose3", DOC(slp, Pose3)};
  nb::class_<Twist2<double>> twist2{autodiff, "Twist2", DOC(slp, Twist2)};
  nb::class_<Twist3<double>> twist3{autodiff, "Twist3", DOC(slp, Twist3)};
  nb::class_<Translation2<double>> translation2{autodiff, "Translation2",
                                                DOC(slp, Translation2)};
  nb::class_<Translation3<double>> translation3{autodiff, "Translation3",
                                                DOC(slp, Translation3)};

  nb::class_<Gradient<double>> gradient{autodiff, "Gradient",
                                        DOC(slp, Gradient)};
  nb::class_<Hessian<double>> hessian{autodiff, "Hessian", DOC(slp, Hessian)};
  nb::class_<Jacobian<double>> jacobian{autodiff, "Jacobian",
                                        DOC(slp, Jacobian)};

  nb::class_<EqualityConstraints<double>> equality_constraints{
      optimization, "EqualityConstraints", DOC(slp, EqualityConstraints)};
  nb::class_<InequalityConstraints<double>> inequality_constraints{
      optimization, "InequalityConstraints", DOC(slp_InequalityConstraints)};

  // Bounds function
  for_each_type<
      double, int, const Variable<double>&, const VariableMatrix<double>&,
      const VariableBlock<VariableMatrix<double>>&, nb::DRef<Eigen::MatrixXd>>(
      [&]<typename L> {
        for_each_type<const Variable<double>&, const VariableMatrix<double>&,
                      const VariableBlock<VariableMatrix<double>>&>(
            [&]<typename X> {
              for_each_type<double, int, const Variable<double>&,
                            const VariableMatrix<double>&,
                            const VariableBlock<VariableMatrix<double>>&,
                            nb::DRef<Eigen::MatrixXd>>([&]<typename U> {
                optimization.def("bounds", &bounds<L&&, X&&, U&&>, "l"_a, "x"_a,
                                 "u"_a, DOC(slp, bounds));
              });
            });
      });

  nb::enum_<ExitStatus> exit_status{optimization, "ExitStatus",
                                    DOC(slp, ExitStatus)};
  nb::class_<IterationInfo<double>> iteration_info{
      optimization, "IterationInfo", DOC(slp, IterationInfo)};

  nb::class_<sphere_t> sphere_cls{optimization, "sphere_t", DOC(slp, sphere_t)};
  nb::class_<so2_t> so2_cls{optimization, "so2_t", DOC(slp, so2_t)};
  nb::class_<so3_t> so3_cls{optimization, "so3_t", DOC(slp, so3_t)};
  nb::class_<se2_t> se2_cls{optimization, "se2_t", DOC(slp, se2_t)};
  nb::class_<se3_t> se3_cls{optimization, "se3_t", DOC(slp, se3_t)};

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
  bind_unit_vector(unit_vector);
  bind_rotation2(rotation2);
  bind_rotation3(rotation3);
  bind_pose2(pose2);
  bind_pose3(pose3);
  bind_twist2(twist2);
  bind_twist3(twist3);
  bind_translation2(translation2);
  bind_translation3(translation3);

  // Implicit conversions
  variable_matrix.def(
      nb::init_implicit<VariableBlock<VariableMatrix<double>>>());
  variable_matrix.def(nb::init_implicit<UnitVector<double>>());
  variable_matrix.def(nb::init_implicit<Translation2<double>>());
  variable_matrix.def(nb::init_implicit<Translation3<double>>());

  variable_matrix.def("rotate_by",
                      nb::overload_cast<const Rotation3<double>&>(
                          &VariableMatrix<double>::rotate_by, nb::const_),
                      nb::arg("q"), DOC(slp, VariableMatrix, rotate_by));
  variable_matrix.def("rotate_by",
                      nb::overload_cast<const Rotation2<double>&>(
                          &VariableMatrix<double>::rotate_by, nb::const_),
                      nb::arg("z"), DOC(slp, VariableMatrix, rotate_by, 2));

  bind_gradient(gradient);
  bind_hessian(hessian);
  bind_jacobian(jacobian);

  bind_equality_constraints(equality_constraints);
  bind_inequality_constraints(inequality_constraints);

  bind_exit_status(exit_status);
  bind_iteration_info(iteration_info);

  bind_manifold_tags(optimization);
  bind_problem(problem);

  bind_dynamics_type(dynamics_type);
  bind_timestep_method(timestep_method);
  bind_transcription_method(transcription_method);

  bind_ocp(ocp);
}

}  // namespace slp
