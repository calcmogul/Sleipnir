// Copyright (c) Sleipnir contributors

#include <cmath>
#include <limits>
#include <numbers>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/pose3.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/translation3.hpp>
#include <sleipnir/autodiff/twist3.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_block.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Rotation3 - Constructor from a rotation vector",
                   "[Rotation3]", SCALAR_TYPES) {
  using T = TestType;

  constexpr T tolerance = rounding_tolerance<T>(10);

  const Eigen::Vector3<T> axis =
      Eigen::Vector3<T>{T(1), T(-2), T(0.5)}.normalized();

  // Angles on both sides of the series switch at θ = 2
  for (T angle : {T(0), T(1e-9), T(0.5), T(1.999), T(2.001), T(3), T(4)}) {
    slp::Rotation3<T> q{Eigen::Vector3<T>{angle * axis}};
    Eigen::Quaternion<T> expected{Eigen::AngleAxis<T>{angle, axis}};

    CAPTURE(angle);
    CHECK_THAT(q.value().coeffs(), WithinAbs(expected.coeffs(), tolerance));

    // to_vector() is the inverse for angles in [0, π]
    if (angle <= T(std::numbers::pi)) {
      CHECK_THAT(q.to_vector().value(),
                 WithinAbs(Eigen::MatrixX<T>{angle * axis}, tolerance));
    }
  }
}

TEMPLATE_TEST_CASE("Twist3 - exp() and log()", "[Twist3]", SCALAR_TYPES) {
  using T = TestType;

  using std::sin;

  constexpr T tolerance(1e-5);

  // Moving along x and z while turning about z at rate ω traces a helix
  for (T ω : {T(0), T(1e-9), T(0.5), T(1.5), T(3)}) {
    CAPTURE(ω);

    slp::Twist3<T> twist{Eigen::Vector3<T>{T(2), T(0), T(1)},
                         Eigen::Vector3<T>{T(0), T(0), ω}};
    auto pose = twist.exp();

    Eigen::Vector3<T> expected_translation =
        ω == T(0)
            ? Eigen::Vector3<T>{T(2), T(0), T(1)}
            : Eigen::Vector3<T>{T(2) * sin(ω) / ω,
                                T(4) * sin(ω / T(2)) * sin(ω / T(2)) / ω, T(1)};
    CHECK_THAT(pose.translation().value(),
               WithinAbs(expected_translation, rounding_tolerance<T>(10)));
  }

  slp::Twist3<T> ξ{Eigen::Vector3<T>{T(1), T(-2), T(0.5)},
                   Eigen::Vector3<T>{T(0.3), T(-1.2), T(2)}};
  auto half = (ξ * T(0.5)).exp();
  CHECK_THAT((half * half).to_matrix().value(),
             WithinAbs(ξ.exp().to_matrix().value(), tolerance));

  auto log = ξ.exp().log();
  CHECK_THAT(
      log.linear().value(),
      WithinAbs(Eigen::MatrixX<T>{Eigen::Vector3<T>{T(1), T(-2), T(0.5)}},
                tolerance));
  CHECK_THAT(
      log.angular().value(),
      WithinAbs(Eigen::MatrixX<T>{Eigen::Vector3<T>{T(0.3), T(-1.2), T(2)}},
                tolerance));

  // exp(log(T)) = T
  const slp::Pose3<T> pose{
      Eigen::Vector3<T>{T(3), T(1), T(-2)},
      Eigen::Quaternion<T>{Eigen::AngleAxis<T>{
          T(2.9), Eigen::Vector3<T>{T(1), T(1), T(1)}.normalized()}}};
  CHECK_THAT(pose.log().exp().to_matrix().value(),
             WithinAbs(pose.to_matrix().value(), tolerance));

  CHECK((ξ == T(2) * ξ).constraints.size() == 6);
}

TEMPLATE_TEST_CASE("Twist3 - Derivatives at zero and across the series switch",
                   "[Twist3]", SCALAR_TYPES) {
  using T = TestType;

  constexpr T ε = std::numeric_limits<T>::epsilon();

  slp::Problem<T> problem;
  auto linear = problem.decision_variable(3);
  auto angular = problem.decision_variable(3);
  slp::VariableMatrix<T> wrt{6, 1};
  for (int i = 0; i < 3; ++i) {
    wrt[i] = linear[i];
    wrt[i + 3] = angular[i];
  }

  slp::Twist3<T> ξ{linear, angular};
  auto pose = ξ.exp();
  auto twist = pose.log();

  // A function that depends on every series
  slp::VariableMatrix<T> out{9, 1};
  for (int i = 0; i < 3; ++i) {
    out[i] = slp::VariableMatrix<T>{pose.translation()}[i];
    out[i + 3] = twist.linear()[i];
    out[i + 6] = twist.angular()[i];
  }
  slp::Variable<T> f = pose.translation().squared_distance(
                           slp::Translation3<T>{T(0), T(0), T(0)}) +
                       twist.linear().T() * twist.linear();

  slp::Jacobian J{out, wrt};
  slp::Hessian H{f, wrt};

  // At zero, exp() is the identity to first order, so the translation's
  // Jacobian is [I 0], and log(exp(ξ)) = ξ has Jacobian I
  linear.set_value(Eigen::Vector3<T>::Zero());
  angular.set_value(Eigen::Vector3<T>::Zero());
  Eigen::MatrixX<T> J_zero = J.value();
  Eigen::MatrixX<T> J_expected{9, 6};
  J_expected.setZero();
  J_expected.topLeftCorner(3, 3).setIdentity();
  J_expected.bottomRows(6).setIdentity();
  CHECK_THAT(J_zero, WithinAbs(J_expected, rounding_tolerance<T>(1)));
  CHECK(Eigen::MatrixX<T>{H.value()}.allFinite());

  // The series and closed forms agree on either side of the switch at θ = 1
  // and θ = 2
  const Eigen::Vector3<T> axis =
      Eigen::Vector3<T>{T(1), T(-2), T(0.5)}.normalized();
  linear.set_value(Eigen::Vector3<T>{T(0.3), T(-1), T(2)});
  for (T θ : {T(1), T(2)}) {
    CAPTURE(θ);

    angular.set_value(Eigen::Vector3<T>{θ * (T(1) - T(4) * ε) * axis});
    Eigen::MatrixX<T> J_series = J.value();
    Eigen::MatrixX<T> H_series = H.value();

    angular.set_value(Eigen::Vector3<T>{θ * (T(1) + T(4) * ε) * axis});
    Eigen::MatrixX<T> J_closed = J.value();
    Eigen::MatrixX<T> H_closed = H.value();

    CHECK_THAT(J_series, WithinAbs(J_closed, rounding_tolerance<T>(100)));
    CHECK_THAT(H_series, WithinAbs(H_closed, rounding_tolerance<T>(500)));
  }
}
