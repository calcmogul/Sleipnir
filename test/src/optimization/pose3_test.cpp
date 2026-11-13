// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/translation3.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/manifold.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("Pose3 - Default value and set_value", "[Pose3]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto pose = problem.decision_variable(slp::se3);

  CHECK(pose.translation().value() == Eigen::Vector3<T>::Zero());
  CHECK(pose.rotation().value().coeffs() ==
        Eigen::Quaternion<T>::Identity().coeffs());

  pose.translation().set_value(Eigen::Vector3<T>{T(1), T(-2), T(3)});
  pose.rotation().set_value(Eigen::Quaternion<T>{T(2), T(0), T(0), T(0)});

  CHECK(pose.translation().z().value() == T(3));
  CHECK(pose.rotation().value().w() == T(1));
}

TEMPLATE_TEST_CASE("Pose3 - Operations", "[Pose3]", SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Quaternion<T> q1{Eigen::AngleAxis<T>{
      T(0.7), Eigen::Vector3<T>{T(1), T(2), T(-1)}.normalized()}};
  const Eigen::Quaternion<T> q2{Eigen::AngleAxis<T>{
      T(-2.1), Eigen::Vector3<T>{T(0), T(-1), T(3)}.normalized()}};
  const Eigen::Vector3<T> t1{T(1), T(2), T(3)};
  const Eigen::Vector3<T> t2{T(-0.5), T(0.25), T(1.5)};

  slp::Problem<T> problem;
  auto pose1 = problem.decision_variable(slp::se3);
  pose1.translation().set_value(t1);
  pose1.rotation().set_value(q1);
  const slp::Pose3<T> pose2{t2, q2};

  auto to_eigen = [](const Eigen::Vector3<T>& t,
                     const Eigen::Quaternion<T>& q) {
    Eigen::Matrix4<T> result = Eigen::Matrix4<T>::Identity();
    result.template topLeftCorner<3, 3>() = q.toRotationMatrix();
    result.template topRightCorner<3, 1>() = t;
    return result;
  };
  const Eigen::Matrix4<T> T1 = to_eigen(t1, q1);
  const Eigen::Matrix4<T> T2 = to_eigen(t2, q2);

  constexpr T tolerance(1e-5);

  CHECK_THAT(pose1.to_matrix().value(), WithinAbs(T1, tolerance));
  CHECK_THAT((pose1 * pose2).to_matrix().value(),
             WithinAbs(Eigen::Matrix4<T>{T1 * T2}, tolerance));
  CHECK_THAT(pose1.inverse().to_matrix().value(),
             WithinAbs(Eigen::Matrix4<T>{T1.inverse()}, tolerance));
  CHECK_THAT((pose1 * pose1.inverse()).to_matrix().value(),
             WithinAbs(Eigen::Matrix4<T>::Identity(), tolerance));

  const Eigen::Vector3<T> p{T(0.3), T(-1.2), T(0.8)};
  Eigen::Vector3<T> expected = (T1 * p.homogeneous()).head(3);
  CHECK_THAT(slp::Translation3<T>{p}.transform_by(pose1).value(),
             WithinAbs(expected, tolerance));
}
