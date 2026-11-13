// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
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

#include "catch_matchers.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("Manifold comparisons - UnitVector equality",
                   "[ManifoldComparison]", SCALAR_TYPES) {
  using T = TestType;
  using std::acos;
  using std::hypot;

  slp::Problem<T> problem;
  auto a = problem.decision_variable(slp::sphere, 3);
  auto b = problem.decision_variable(slp::sphere, 3);
  a.set_value(Eigen::Vector3<T>{T(0.6), T(0.8), T(0)});
  b.set_value(Eigen::Vector3<T>{T(0.6), T(0.8), T(0)});

  constexpr T tolerance(1e-5);

  // Unit vectors in ℝ³ have two degrees of freedom
  auto c = a == b;
  CHECK(c.constraints.size() == 2);
  for (auto& constraint : c.constraints) {
    CHECK_THAT(constraint.value(), WithinAbs(T(0), tolerance));
  }

  // The components are the logarithm map, so their norm is the angle
  b.set_value(Eigen::Vector3<T>{T(0), T(0.8), T(0.6)});
  T expected_angle = acos(T(0.64));
  for (auto&& constraints : {a == b, a == Eigen::Vector3<T>{T(0), T(4), T(3)},
                             Eigen::Vector3<T>{T(0), T(4), T(3)} == a}) {
    REQUIRE(constraints.constraints.size() == 2);
    T norm = hypot(slp::Variable<T>{constraints.constraints[0]}.value(),
                   slp::Variable<T>{constraints.constraints[1]}.value());
    CHECK_THAT(norm, WithinAbs(expected_angle, tolerance));
  }

  // The opposite vector isn't a solution
  b.set_value(Eigen::Vector3<T>{T(-0.6), T(-0.8), T(0.01)});
  auto opposite = a == b;
  T norm = hypot(slp::Variable<T>{opposite.constraints[0]}.value(),
                 slp::Variable<T>{opposite.constraints[1]}.value());
  CHECK(norm > T(3));

  // Comparing with a VariableMatrix is still elementwise
  CHECK((a == slp::VariableMatrix<T>{b}).constraints.size() == 3);
}

TEMPLATE_TEST_CASE("Manifold comparisons - Rotation and pose equality",
                   "[ManifoldComparison]", SCALAR_TYPES) {
  using T = TestType;

  constexpr T tolerance(1e-5);

  // The constraint is the relative angle, even across ±π
  slp::Rotation2<T> z1{T(3)};
  slp::Rotation2<T> z2{T(-3)};
  auto c2 = z1 == z2;
  REQUIRE(c2.constraints.size() == 1);
  CHECK_THAT(c2.constraints[0].value(),
             WithinAbs(T(6) - T(2 * std::numbers::pi), tolerance));

  // The constraint is the relative rotation vector, and q and −q are equal
  const Eigen::Quaternion<T> q{
      Eigen::AngleAxis<T>{T(0.5), Eigen::Vector3<T>::UnitZ()}};
  slp::Rotation3<T> q1{q};
  slp::Rotation3<T> q2{Eigen::Quaternion<T>{-q.coeffs()}};
  slp::Rotation3<T> q3{Eigen::Quaternion<T>::Identity()};
  auto c3 = q1 == q2;
  REQUIRE(c3.constraints.size() == 3);
  for (auto& constraint : c3.constraints) {
    CHECK_THAT(constraint.value(), WithinAbs(T(0), tolerance));
  }
  auto c3_angle = q1 == q3;
  CHECK_THAT(c3_angle.constraints[2].value(), WithinAbs(T(0.5), tolerance));

  slp::Pose2<T> pose2{Eigen::Vector2<T>{T(1), T(2)}, T(0.5)};
  CHECK((pose2 == pose2).constraints.size() == 3);
  slp::Pose3<T> pose3{Eigen::Vector3<T>{T(1), T(2), T(3)}, q};
  CHECK((pose3 == pose3).constraints.size() == 6);
}

TEMPLATE_TEST_CASE("Manifold comparisons - relative_to() and distances",
                   "[ManifoldComparison]", SCALAR_TYPES) {
  using T = TestType;

  constexpr T tolerance(1e-5);

  // The relative angle wraps only when the rotations are opposite
  slp::Rotation2<T> z1{T(3)};
  slp::Rotation2<T> z2{T(-3)};
  CHECK_THAT(z1.relative_to(z2).value(),
             WithinAbs(T(6) - T(2 * std::numbers::pi), tolerance));

  // other * this.relative_to(other) = this
  const Eigen::Quaternion<T> qa{Eigen::AngleAxis<T>{
      T(0.5), Eigen::Vector3<T>{T(1), T(2), T(0)}.normalized()}};
  const Eigen::Quaternion<T> qb{Eigen::AngleAxis<T>{
      T(-1.2), Eigen::Vector3<T>{T(0), T(-1), T(3)}.normalized()}};
  slp::Rotation3<T> q1{qa};
  slp::Rotation3<T> q2{qb};
  CHECK_THAT((q2 * q1.relative_to(q2)).value().coeffs(),
             WithinAbs(qa.coeffs(), tolerance));
  CHECK_THAT(q1.relative_to(slp::Rotation3<T>{Eigen::Quaternion<T>::Identity()})
                 .value()
                 .coeffs(),
             WithinAbs(qa.coeffs(), tolerance));

  slp::Pose2<T> pose2a{Eigen::Vector2<T>{T(1), T(2)}, T(0.5)};
  slp::Pose2<T> pose2b{Eigen::Vector2<T>{T(-3), T(1)}, T(2.5)};
  CHECK_THAT((pose2b * pose2a.relative_to(pose2b)).to_matrix().value(),
             WithinAbs(pose2a.to_matrix().value(), tolerance));

  slp::Pose3<T> pose3a{Eigen::Vector3<T>{T(1), T(2), T(3)}, qa};
  slp::Pose3<T> pose3b{Eigen::Vector3<T>{T(-3), T(1), T(0.5)}, qb};
  CHECK_THAT((pose3b * pose3a.relative_to(pose3b)).to_matrix().value(),
             WithinAbs(pose3a.to_matrix().value(), tolerance));

  slp::Translation2<T> t{T(3), T(4)};
  slp::Translation2<T> origin{T(0), T(0)};
  CHECK_THAT(t.distance(origin).value(), WithinAbs(T(5), tolerance));
  CHECK_THAT(t.squared_distance(origin).value(), WithinAbs(T(25), tolerance));
}
