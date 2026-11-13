// Copyright (c) Sleipnir contributors

#include <cmath>

#include <Eigen/Core>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/pose2.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/autodiff/twist2.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_block.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Rotation2 - Constructor from a Variable angle",
                   "[Rotation2]", SCALAR_TYPES) {
  using T = TestType;

  using std::cos;
  using std::sin;

  slp::Problem<T> problem;
  auto θ = problem.decision_variable();
  θ.set_value(T(2.5));
  slp::Rotation2<T> z{θ};

  constexpr T tolerance(1e-6);
  CHECK_THAT(z.value(), WithinAbs(T(2.5), tolerance));
  CHECK_THAT(z.angle().value(), WithinAbs(T(2.5), tolerance));

  // The rotation depends on θ
  θ.set_value(T(-1));
  CHECK_THAT(z.value(), WithinAbs(T(-1), tolerance));
  slp::Variable<T> c = z.cos();
  CHECK_THAT(c.value(), WithinAbs(cos(T(-1)), tolerance));

  // Constants still use the Scalar constructor
  CHECK_THAT(slp::Rotation2<T>{T(0.5)}.value(), WithinAbs(T(0.5), tolerance));
}

TEMPLATE_TEST_CASE("Twist2 - exp() and log()", "[Twist2]", SCALAR_TYPES) {
  using T = TestType;

  using std::sin;

  constexpr T tolerance(1e-5);

  // Moving at speed v while turning at rate ω traces a circular arc of radius
  // v/ω
  for (T ω : {T(0), T(1e-9), T(0.5), T(1.5), T(3)}) {
    CAPTURE(ω);

    slp::Twist2<T> twist{Eigen::Vector2<T>{T(2), T(0)}, ω};
    auto pose = twist.exp();

    Eigen::Vector2<T> expected_translation =
        ω == T(0) ? Eigen::Vector2<T>{T(2), T(0)}
                  : Eigen::Vector2<T>{T(2) * sin(ω) / ω,
                                      T(4) * sin(ω / T(2)) * sin(ω / T(2)) / ω};
    CHECK_THAT(pose.translation().value(),
               WithinAbs(expected_translation, rounding_tolerance<T>(10)));
    CHECK_THAT(pose.rotation().value(),
               WithinAbs(ω, rounding_tolerance<T>(10)));
  }

  // exp() of a twist is a one-parameter subgroup, so exp(ξ) = exp(ξ/2)², and
  // log() inverts it for angular parts in (−π, π]
  slp::Twist2<T> ξ{Eigen::Vector2<T>{T(1), T(-2)}, T(2.5)};
  auto half = (ξ * T(0.5)).exp();
  CHECK_THAT((half * half).to_matrix().value(),
             WithinAbs(ξ.exp().to_matrix().value(), tolerance));

  auto log = ξ.exp().log();
  CHECK_THAT(
      log.linear().value(),
      WithinAbs(Eigen::MatrixX<T>{Eigen::Vector2<T>{T(1), T(-2)}}, tolerance));
  slp::Variable<T> log_angular = log.angular();
  CHECK_THAT(log_angular.value(), WithinAbs(T(2.5), tolerance));

  // Twist arithmetic
  slp::Twist2<T> sum = ξ + ξ - (-ξ);
  CHECK_THAT(
      sum.linear().value(),
      WithinAbs(Eigen::MatrixX<T>{Eigen::Vector2<T>{T(3), T(-6)}}, tolerance));
  CHECK((ξ == T(2) * ξ).constraints.size() == 3);
}
