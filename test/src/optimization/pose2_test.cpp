// Copyright (c) Sleipnir contributors

#include <cmath>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Pose2 - Default value and set_value", "[Pose2]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto pose = problem.pose2_decision_variable();

  CHECK(pose.translation().value() == Eigen::Vector2<T>::Zero());
  CHECK(pose.rotation().value() == T(0));

  pose.translation().set_value(Eigen::Vector2<T>{T(1), T(-2)});
  pose.rotation().set_value(T(0.5));

  constexpr T tolerance(1e-6);
  CHECK(pose.translation().x().value() == T(1));
  CHECK(pose.translation().y().value() == T(-2));
  CHECK_THAT(pose.rotation().value(), WithinAbs(T(0.5), tolerance));
}

TEMPLATE_TEST_CASE("Pose2 - Parameter", "[Pose2]", SCALAR_TYPES) {
  using T = TestType;
  using enum slp::ExpressionType;

  std::vector<slp::Translation2<T>> points{
      {T(1), T(0)}, {T(0), T(2)}, {T(-1), T(-1)}};

  slp::Problem<T> problem;
  auto pose = problem.pose2_decision_variable();
  auto target = problem.pose2_parameter();

  CHECK(target.translation().value() == Eigen::Vector2<T>::Zero());
  CHECK(target.rotation().value() == T(0));
  CHECK(target.translation().x().type() == PARAMETER);
  CHECK(target.rotation().cos().type() == PARAMETER);

  auto translation = problem.translation2_parameter();
  CHECK(translation.value() == Eigen::Vector2<T>::Zero());
  CHECK(translation.y().type() == PARAMETER);

  slp::Real<T> cost{T(0)};
  for (auto& p : points) {
    cost += p.transform_by(pose).squared_distance(p.transform_by(target));
  }
  problem.minimize(cost);

  for (auto [x, y, angle] :
       {std::tuple{T(1), T(-0.5), T(0.4)}, std::tuple{T(-2), T(3), T(2.8)}}) {
    target.translation().set_value(Eigen::Vector2<T>{x, y});
    target.rotation().set_value(angle);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    CHECK_THAT(pose.translation().x().value(),
               WithinAbs(x, solution_tolerance<T>(1)));
    CHECK_THAT(pose.translation().y().value(),
               WithinAbs(y, solution_tolerance<T>(1)));
    CHECK_THAT(pose.rotation().value(),
               WithinAbs(angle, solution_tolerance<T>(1)));
  }
}

TEMPLATE_TEST_CASE("Pose2 - Operations", "[Pose2]", SCALAR_TYPES) {
  using T = TestType;

  using std::cos;
  using std::sin;

  slp::Problem<T> problem;
  auto pose1 = problem.pose2_decision_variable();
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
