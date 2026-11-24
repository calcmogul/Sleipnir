// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>

#include <complex>

#include <Eigen/Core>
#include <sleipnir/autodiff/complex.hpp>
#include <sleipnir/autodiff/complex_matrix.hpp>
#include <sleipnir/autodiff/complex_submatrix.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/real_submatrix.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/similarity2.hpp>
#include <sleipnir/autodiff/similarity3.hpp>
#include <sleipnir/autodiff/similarity_twist2.hpp>
#include <sleipnir/autodiff/similarity_twist3.hpp>
#include <sleipnir/autodiff/spd_matrix.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/autodiff/translation3.hpp>
#include <sleipnir/autodiff/twist2.hpp>
#include <sleipnir/autodiff/twist3.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>
#include <sleipnir/optimization/lint.hpp>
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

  nb::class_<Real<double>> real{autodiff, "Real", DOC(slp, Real)};
  nb::class_<RealMatrix<double>> real_matrix{autodiff, "RealMatrix",
                                             DOC(slp, RealMatrix, 2)};
  nb::class_<Complex<double>> complex{autodiff, "Complex", DOC(slp, Complex)};
  nb::class_<ComplexMatrix<double>> complex_matrix{autodiff, "ComplexMatrix",
                                                   DOC(slp, ComplexMatrix)};
  nb::class_<ComplexSubmatrix<ComplexMatrix<double>>> complex_submatrix{
      autodiff, "ComplexSubmatrix", DOC(slp, ComplexSubmatrix)};
  nb::class_<RealSubmatrix<RealMatrix<double>>> real_submatrix{
      autodiff, "RealSubmatrix", DOC(slp, RealSubmatrix)};
  nb::class_<UnitVector<double>> unit_vector{autodiff, "UnitVector",
                                             DOC(slp, UnitVector)};
  nb::class_<SPDMatrix<double>> spd_matrix{autodiff, "SPDMatrix",
                                           DOC(slp, SPDMatrix)};
  nb::class_<SymmetricView<double>> symmetric_view{autodiff, "SymmetricView",
                                                   DOC(slp, SymmetricView)};
  nb::class_<Rotation2<double>> rotation2{autodiff, "Rotation2",
                                          DOC(slp, Rotation2)};
  nb::class_<Rotation3<double>> rotation3{autodiff, "Rotation3",
                                          DOC(slp, Rotation3)};
  nb::class_<Pose2<double>> pose2{autodiff, "Pose2", DOC(slp, Pose2)};
  nb::class_<Pose3<double>> pose3{autodiff, "Pose3", DOC(slp, Pose3)};
  nb::class_<Similarity2<double>> similarity2{autodiff, "Similarity2",
                                              DOC(slp, Similarity2)};
  nb::class_<Similarity3<double>> similarity3{autodiff, "Similarity3",
                                              DOC(slp, Similarity3)};
  nb::class_<Twist2<double>> twist2{autodiff, "Twist2", DOC(slp, Twist2)};
  nb::class_<Twist3<double>> twist3{autodiff, "Twist3", DOC(slp, Twist3)};
  nb::class_<SimilarityTwist2<double>> similarity_twist2{
      autodiff, "SimilarityTwist2", DOC(slp, SimilarityTwist2)};
  nb::class_<SimilarityTwist3<double>> similarity_twist3{
      autodiff, "SimilarityTwist3", DOC(slp, SimilarityTwist3)};
  nb::class_<Translation2<double>> translation2{autodiff, "Translation2",
                                                DOC(slp, Translation2)};
  nb::class_<Translation3<double>> translation3{autodiff, "Translation3",
                                                DOC(slp, Translation3, 2)};

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
  for_each_type<double, int, const Real<double>&, const RealMatrix<double>&,
                const RealSubmatrix<RealMatrix<double>>&,
                nb::DRef<Eigen::MatrixXd>>([&]<typename L> {
    for_each_type<const Real<double>&, const RealMatrix<double>&,
                  const RealSubmatrix<RealMatrix<double>>&>([&]<typename X> {
      for_each_type<double, int, const Real<double>&, const RealMatrix<double>&,
                    const RealSubmatrix<RealMatrix<double>>&,
                    nb::DRef<Eigen::MatrixXd>>([&]<typename U> {
        optimization.def("bounds", &bounds<L&&, X&&, U&&>, "l"_a, "x"_a, "u"_a,
                         DOC(slp, bounds));
      });
    });
  });

  nb::enum_<ExitStatus> exit_status{optimization, "ExitStatus",
                                    DOC(slp, ExitStatus)};
  nb::class_<IterationInfo<double>> iteration_info{
      optimization, "IterationInfo", DOC(slp, IterationInfo)};

  nb::enum_<LintKind> lint_kind{optimization, "LintKind", DOC(slp, LintKind)};
  nb::class_<LintFinding> lint_finding{optimization, "LintFinding",
                                       DOC(slp, LintFinding)};

  // The forward declarations in rotation2.hpp, rotation3.hpp, spd_matrix.hpp,
  // and unit_vector.hpp sort first, so they take the first four Problem
  // docstrings
  nb::class_<Problem<double>> problem{optimization, "Problem",
                                      DOC(slp, Problem, 5)};

  nb::enum_<DynamicsType> dynamics_type{optimization, "DynamicsType",
                                        DOC(slp, DynamicsType)};
  nb::enum_<TimestepMethod> timestep_method{optimization, "TimestepMethod",
                                            DOC(slp, TimestepMethod)};
  nb::enum_<TranscriptionMethod> transcription_method{
      optimization, "TranscriptionMethod", DOC(slp, TranscriptionMethod)};

  nb::class_<OCP<double>, Problem<double>> ocp{optimization, "OCP",
                                               DOC(slp, OCP)};

  bind_expression_type(expression_type);

  bind_real(autodiff, real);
  bind_real_matrix(autodiff, real_matrix);
  bind_complex(complex);
  bind_complex_matrix(complex_matrix);
  bind_complex_submatrix(complex_submatrix);
  bind_real_submatrix(real_submatrix);
  bind_unit_vector(unit_vector);
  bind_spd_matrix(spd_matrix);
  bind_symmetric_view(symmetric_view);
  bind_rotation2(rotation2);
  bind_rotation3(rotation3);
  bind_pose2(pose2);
  bind_pose3(pose3);
  bind_similarity2(similarity2);
  bind_similarity3(similarity3);
  bind_twist2(twist2);
  bind_twist3(twist3);
  bind_similarity_twist2(similarity_twist2);
  bind_similarity_twist3(similarity_twist3);
  bind_translation2(translation2);
  bind_translation3(translation3);

  // Implicit conversions
  real_matrix.def(nb::init_implicit<RealSubmatrix<RealMatrix<double>>>());
  real_matrix.def(nb::init_implicit<UnitVector<double>>());
  real_matrix.def(nb::init_implicit<Translation2<double>>());
  real_matrix.def(nb::init_implicit<Translation3<double>>());
  complex.def(nb::init_implicit<std::complex<double>>());
  complex.def(nb::init_implicit<Real<double>>());
  complex_matrix.def(nb::init_implicit<RealMatrix<double>>());
  complex_matrix.def(
      nb::init_implicit<ComplexSubmatrix<ComplexMatrix<double>>>());

  // An SPDMatrix converts explicitly, as in C++
  real_matrix.def(nb::init<const SPDMatrix<double>&>(), "matrix"_a,
                  DOC(slp, SPDMatrix, operator, slp, RealMatrix));

  bind_gradient(gradient);
  bind_hessian(hessian);
  bind_jacobian(jacobian);

  bind_equality_constraints(equality_constraints);
  bind_inequality_constraints(inequality_constraints);

  bind_exit_status(exit_status);
  bind_iteration_info(iteration_info);

  bind_lint_kind(lint_kind);
  bind_lint_finding(lint_finding);

  bind_problem(problem);

  bind_dynamics_type(dynamics_type);
  bind_timestep_method(timestep_method);
  bind_transcription_method(transcription_method);

  bind_ocp(ocp);
}

}  // namespace slp
