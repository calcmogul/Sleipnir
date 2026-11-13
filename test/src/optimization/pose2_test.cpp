// Copyright (c) Sleipnir contributors

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/manifold.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("Pose2 - Default value and set_value", "[Pose2]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto pose = problem.decision_variable(slp::se2);

  CHECK(pose.translation().value() == Eigen::Vector2<T>::Zero());
  CHECK(pose.rotation().value() == T(0));

  pose.translation().set_value(Eigen::Vector2<T>{T(1), T(-2)});
  pose.rotation().set_value(T(0.5));

  constexpr T tolerance(1e-6);
  CHECK(pose.translation().x().value() == T(1));
  CHECK(pose.translation().y().value() == T(-2));
  CHECK_THAT(pose.rotation().value(), WithinAbs(T(0.5), tolerance));
}

TEMPLATE_TEST_CASE("Pose2 - Operations", "[Pose2]", SCALAR_TYPES) {
  using T = TestType;

  using std::cos;
  using std::sin;

  slp::Problem<T> problem;
  auto pose1 = problem.decision_variable(slp::se2);
  pose1.translation().set_value(Eigen::Vector2<T>{T(1), T(2)});
  pose1.rotation().set_value(T(0.7));
  const slp::Pose2<T> pose2{Eigen::Vector2<T>{T(-0.5), T(0.25)}, T(-2.1)};

  auto to_eigen = [](T x, T y, T θ) {
    Eigen::Matrix3<T> result{
        {cos(θ), -sin(θ), x}, {sin(θ), cos(θ), y}, {T(0), T(0), T(1)}};
    return result;
  };
  const Eigen::Matrix3<T> T1 = to_eigen(T(1), T(2), T(0.7));
  const Eigen::Matrix3<T> T2 = to_eigen(T(-0.5), T(0.25), T(-2.1));

  constexpr T tolerance(1e-5);

  CHECK_THAT(pose1.to_matrix().value(), WithinAbs(T1, tolerance));
  CHECK_THAT((pose1 * pose2).to_matrix().value(),
             WithinAbs(Eigen::Matrix3<T>{T1 * T2}, tolerance));
  CHECK_THAT(pose1.inverse().to_matrix().value(),
             WithinAbs(Eigen::Matrix3<T>{T1.inverse()}, tolerance));
  CHECK_THAT((pose1 * pose1.inverse()).to_matrix().value(),
             WithinAbs(Eigen::Matrix3<T>::Identity(), tolerance));

  // Poses can also be made from a Translation2 and a Rotation2
  const slp::Pose2<T> pose2_from_parts{slp::Translation2<T>{T(-0.5), T(0.25)},
                                       slp::Rotation2<T>{T(-2.1)}};
  CHECK_THAT(pose2_from_parts.to_matrix().value(), WithinAbs(T2, tolerance));

  const Eigen::Vector2<T> p{T(0.3), T(-1.2)};
  Eigen::Vector2<T> expected = (T1 * p.homogeneous()).head(2);
  CHECK_THAT(slp::Translation2<T>{p}.transform_by(pose1).value(),
             WithinAbs(expected, tolerance));
}
