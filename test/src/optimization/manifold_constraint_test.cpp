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
#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/manifold.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Problem - Heading equality from the opposite heading",
                   "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto pose = problem.decision_variable(slp::se2);

  // The initial heading is 0, opposite the constrained heading
  problem.minimize(
      pose.translation().squared_distance(slp::Translation2<T>{T(1), T(2)}));
  problem.subject_to(pose.rotation() == slp::Rotation2<T>{T(std::numbers::pi)});

  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  using std::abs;
  CHECK_THAT(abs(pose.rotation().value()),
             WithinAbs(T(std::numbers::pi), solution_tolerance<T>(1)));
  CHECK_THAT(pose.translation().x().value(),
             WithinAbs(T(1), solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem - Pose3 equality", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Quaternion<T> q_target{Eigen::AngleAxis<T>{
      T(2.9), Eigen::Vector3<T>{T(1), T(-2), T(0.5)}.normalized()}};
  const slp::Pose3<T> target{Eigen::Vector3<T>{T(1), T(-0.5), T(2)}, q_target};

  slp::Problem<T> problem;
  auto pose = problem.decision_variable(slp::se3);
  problem.subject_to(pose == target);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  Eigen::Vector3<T> t = pose.translation().value();
  CHECK_THAT(t, WithinAbs(Eigen::Vector3<T>{T(1), T(-0.5), T(2)},
                          solution_tolerance<T>(1)));
  using std::abs;
  CHECK_THAT(abs(pose.rotation().value().dot(q_target)),
             WithinAbs(T(1), solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem - UnitVector equality", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Matrix3<T> A{
      {T(2), T(-1), T(0.5)}, {T(-1), T(3), T(0.25)}, {T(0.5), T(0.25), T(1)}};

  SECTION("Constant direction") {
    // The direction is nearly opposite the initial value e₁
    const Eigen::Vector3<T> d{T(-1), T(0.2), T(-0.1)};

    slp::Problem<T> problem;
    auto n = problem.decision_variable(slp::sphere, 3);
    problem.minimize(n.T() * A * n);
    problem.subject_to(n == d);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    Eigen::Vector3<T> n_value = n.value();
    CHECK_THAT(n_value, WithinAbs(Eigen::Vector3<T>{d.normalized()},
                                  solution_tolerance<T>(1)));
  }

  SECTION("Two decision variables") {
    slp::Problem<T> problem;
    auto a = problem.decision_variable(slp::sphere, 3);
    auto b = problem.decision_variable(slp::sphere, 3);
    a.set_value(Eigen::Vector3<T>{T(1), T(0.3), T(0)});
    b.set_value(Eigen::Vector3<T>{T(0.2), T(1), T(0.4)});
    problem.minimize(a.T() * A * a + b.T() * A * b);
    problem.subject_to(a == b);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3<T>> eigensolver{A};
    Eigen::Vector3<T> a_value = a.value();
    Eigen::Vector3<T> b_value = b.value();
    CHECK_THAT(a_value.dot(b_value), WithinAbs(T(1), solution_tolerance<T>(1)));
    CHECK_THAT(a_value.dot(A * a_value), WithinAbs(eigensolver.eigenvalues()[0],
                                                   solution_tolerance<T>(1)));
  }
}

TEMPLATE_TEST_CASE("Problem - Heading tolerance wrapping past pi",
                   "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  using std::cos;
  using std::sin;

  // The allowed headings are 3 ± 0.3 rad, which wraps past π. The cost pulls
  // the heading toward −2.5 rad, past the upper end at 3.3 rad.
  slp::Problem<T> problem;
  auto z = problem.decision_variable(slp::so2);
  z.set_value(T(3));
  problem.minimize(-(z.cos() * cos(T(-2.5)) + z.sin() * sin(T(-2.5))));
  auto error = z.relative_to(slp::Rotation2<T>{T(3)}).angle();
  problem.subject_to(slp::bounds(T(-0.3), error, T(0.3)));

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  // The active bound's multiplier is |sin(3.3 + 2.5)| ≈ 0.46, so the
  // interior-point slack at convergence is about tolerance/0.46
  CHECK_THAT(z.value(), WithinAbs(T(3.3 - 2 * std::numbers::pi),
                                  solution_tolerance<T>(5)));
}

TEMPLATE_TEST_CASE("Problem - Rotation3 and UnitVector tolerances",
                   "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  using std::abs;
  using std::acos;
  using std::cos;

  // The cost pulls the rotation toward 2 rad about an axis, and the
  // constraint keeps it within 0.4 rad of the identity
  const Eigen::Quaternion<T> q_target{Eigen::AngleAxis<T>{
      T(2), Eigen::Vector3<T>{T(1), T(1), T(0)}.normalized()}};
  const slp::Rotation3<T> target{q_target};

  slp::Problem<T> problem;
  auto q = problem.decision_variable(slp::so3);
  auto n = problem.decision_variable(slp::sphere, 3);

  slp::Variable<T> q_dot = q.w() * target.w() + q.x() * target.x() +
                           q.y() * target.y() + q.z() * target.z();
  problem.minimize(-q_dot * q_dot - n[2]);
  auto error =
      q.relative_to(slp::Rotation3<T>{Eigen::Quaternion<T>::Identity()})
          .to_vector();
  problem.subject_to(error.T() * error <= T(0.4 * 0.4));

  // The cost pulls the vector toward e₃, and the constraint keeps it within
  // 0.3 rad of e₁
  problem.subject_to(n.dot(Eigen::Vector3<T>::UnitX()) >= cos(T(0.3)));

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(T(2) * acos(abs(q.value().w())),
             WithinAbs(T(0.4), solution_tolerance<T>(1)));
  CHECK_THAT(n.value(0), WithinAbs(cos(T(0.3)), solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem - Distance tolerances", "[Problem][Manifold]",
                   SCALAR_TYPES) {
  using T = TestType;

  SECTION("At most") {
    slp::Problem<T> problem;
    auto pose = problem.decision_variable(slp::se2);
    problem.minimize(
        pose.translation().squared_distance(slp::Translation2<T>{T(3), T(4)}));
    problem.subject_to(pose.translation().squared_distance(
                           slp::Translation2<T>{T(0), T(0)}) <= T(1));

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    CHECK_THAT(pose.translation().x().value(),
               WithinAbs(T(0.6), solution_tolerance<T>(1)));
    CHECK_THAT(pose.translation().y().value(),
               WithinAbs(T(0.8), solution_tolerance<T>(1)));
  }

  SECTION("At least") {
    slp::Problem<T> problem;
    slp::Translation2<T> p{problem.decision_variable(2)};
    p.set_value(Eigen::Vector2<T>{T(0.5), T(0.1)});
    problem.minimize(p.squared_distance(slp::Translation2<T>{T(1), T(0)}));
    problem.subject_to(p.squared_distance(slp::Translation2<T>{T(0), T(0)}) >=
                       T(4));

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    CHECK_THAT(p.x().value(), WithinAbs(T(2), solution_tolerance<T>(1)));
    CHECK_THAT(p.y().value(), WithinAbs(T(0), solution_tolerance<T>(1)));
  }
}
