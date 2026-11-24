// Copyright (c) Sleipnir contributors

// GCC reports the fixed-size 3x3 SelfAdjointEigenSolver's eigenvalues as
// maybe-uninitialized after construction, which computes them
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

namespace {

template <typename T>
Eigen::Matrix3<T> rayleigh_matrix() {
  // Symmetric with distinct eigenvalues
  return Eigen::Matrix3<T>{
      {T(2), T(-1), T(0.5)}, {T(-1), T(3), T(0.25)}, {T(0.5), T(0.25), T(1)}};
}

/// Adds an iteration callback that checks the manifold blocks of the iterate
/// stay unit-length.
template <typename T>
void check_iterates_on_manifold(slp::Problem<T>& problem, int offset,
                                int size) {
  problem.add_callback([=](const slp::IterationInfo<T>& info) {
    CHECK_THAT(info.x.segment(offset, size).norm(),
               WithinAbs(T(1), rounding_tolerance<T>(10)));
  });
}

}  // namespace

TEMPLATE_TEST_CASE("Problem - Sphere Rayleigh quotient", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Matrix3<T> A = rayleigh_matrix<T>();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<T>> eigensolver{A};
  const T λ_min = eigensolver.eigenvalues()[0];
  const Eigen::Vector3<T> v_min = eigensolver.eigenvectors().col(0);

  auto x0 =
      GENERATE(Eigen::Vector3d{1, 0, 0}, Eigen::Vector3d{-1, 0.2, 0.1},
               Eigen::Vector3d{0.1, 1, -0.3}, Eigen::Vector3d{0.5, -0.5, 0.7});

  slp::Problem<T> problem;
  auto n = problem.unit_vector_decision_variable(3);
  n.set_value(x0.template cast<T>().eval());

  problem.minimize(n.T() * A * n);
  check_iterates_on_manifold(problem, 0, 3);

  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  Eigen::Vector3<T> n_value = n.value();
  CHECK_THAT(n_value.norm(), WithinAbs(T(1), rounding_tolerance<T>(10)));
  CHECK_THAT(n_value.dot(A * n_value),
             WithinAbs(λ_min, solution_tolerance<T>(1)));
  using std::abs;
  CHECK_THAT(abs(n_value.dot(v_min)),
             WithinAbs(T(1), solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE(
    "Problem - Sphere with constraints matches unit-norm "
    "equality constraint",
    "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Matrix3<T> A = rayleigh_matrix<T>();
  const Eigen::Vector3<T> x0 = Eigen::Vector3<T>{T(1), T(1), T(1)}.normalized();

  // Equality constraint (SQP) or inequality constraint (IPM)
  auto use_inequality = GENERATE(false, true);

  auto add_constraint = [&](slp::Problem<T>& problem, const auto& n) {
    if (use_inequality) {
      problem.subject_to(n[2] >= T(0.6));
    } else {
      problem.subject_to(n[0] == T(0.5));
    }
  };

  // Manifold formulation
  slp::Problem<T> manifold_problem;
  auto n = manifold_problem.unit_vector_decision_variable(3);
  n.set_value(x0);
  manifold_problem.minimize(n.T() * A * n);
  add_constraint(manifold_problem, n);
  check_iterates_on_manifold(manifold_problem, 0, 3);

  CHECK(manifold_problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::SUCCESS);

  // Unit-norm equality constraint formulation
  slp::Problem<T> constrained_problem;
  auto m = constrained_problem.real_decision_variable(3);
  m.set_value(x0);
  constrained_problem.minimize(m.T() * A * m);
  constrained_problem.subject_to(m.T() * m == T(1));
  add_constraint(constrained_problem, m);

  CHECK(constrained_problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::SUCCESS);

  CHECK_THAT(n.value(), WithinAbs(m.value(), solution_tolerance<T>(10)));
}
