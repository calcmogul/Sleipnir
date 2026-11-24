// Copyright (c) Sleipnir contributors

// GCC reports the fixed-size 3x3 SelfAdjointEigenSolver's eigenvalues as
// maybe-uninitialized after construction, which computes them
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Problem - UnitVector equality: constant direction",
                   "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Matrix3<T> A{
      {T(2), T(-1), T(0.5)}, {T(-1), T(3), T(0.25)}, {T(0.5), T(0.25), T(1)}};

  // The direction is nearly opposite the initial value e₁
  const Eigen::Vector3<T> d{T(-1), T(0.2), T(-0.1)};

  slp::Problem<T> problem;
  auto n = problem.unit_vector_decision_variable(3);
  problem.minimize(n.T() * A * n);
  problem.subject_to(n == d);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  Eigen::Vector3<T> n_value = n.value();
  CHECK_THAT(n_value, WithinAbs(Eigen::Vector3<T>{d.normalized()},
                                solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem - UnitVector equality: two decision variables",
                   "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Matrix3<T> A{
      {T(2), T(-1), T(0.5)}, {T(-1), T(3), T(0.25)}, {T(0.5), T(0.25), T(1)}};

  slp::Problem<T> problem;
  auto a = problem.unit_vector_decision_variable(3);
  auto b = problem.unit_vector_decision_variable(3);
  a.set_value(Eigen::Vector3<T>{T(1), T(0.3), T(0)});
  b.set_value(Eigen::Vector3<T>{T(0.2), T(1), T(0.4)});
  problem.minimize(a.T() * A * a + b.T() * A * b);
  problem.subject_to(a == b);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<T>> eigensolver{A};
  Eigen::Vector3<T> a_value = a.value();
  Eigen::Vector3<T> b_value = b.value();
  CHECK_THAT(a_value.dot(b_value), WithinAbs(T(1), solution_tolerance<T>(1)));
  CHECK_THAT(a_value.dot(A * a_value),
             WithinAbs(eigensolver.eigenvalues()[0], solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem - Distance tolerances: at least",
                   "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto p = problem.translation2_decision_variable();
  p.set_value(Eigen::Vector2<T>{T(0.5), T(0.1)});
  problem.minimize(p.squared_distance(slp::Translation2<T>{T(1), T(0)}));
  problem.subject_to(p.squared_distance(slp::Translation2<T>{T(0), T(0)}) >=
                     T(4));

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(p.x().value(), WithinAbs(T(2), solution_tolerance<T>(1)));
  CHECK_THAT(p.y().value(), WithinAbs(T(0), solution_tolerance<T>(1)));
}
