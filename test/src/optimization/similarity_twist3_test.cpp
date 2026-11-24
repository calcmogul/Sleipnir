// Copyright (c) Sleipnir contributors

#include <limits>
#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/similarity3.hpp>
#include <sleipnir/autodiff/similarity_twist3.hpp>
#include <sleipnir/autodiff/translation3.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <unsupported/Eigen/MatrixFunctions>  // IWYU pragma: keep

#include "catch_matchers.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("SimilarityTwist3 - exp() and log()", "[SimilarityTwist3]",
                   SCALAR_TYPES) {
  using T = TestType;

  constexpr T tolerance(1e-5);

  // exp() is the matrix exponential of the generator
  //
  //   [σI + Ω  v]
  //   [  0     0]
  //
  // where Ω is the cross product matrix of ω
  const Eigen::Vector3<T> axis =
      Eigen::Vector3<T>{T(1), T(-2), T(0.5)}.normalized();
  for (auto [σ, θ] : {std::pair{T(0), T(0)}, std::pair{T(1e-9), T(1e-9)},
                      std::pair{T(0.3), T(0)}, std::pair{T(0), T(2)},
                      std::pair{T(0.6), T(0.8)}, std::pair{T(-1.5), T(0.2)},
                      std::pair{T(2), T(3)}}) {
    CAPTURE(σ, θ);

    const Eigen::Vector3<T> v{T(1), T(-2), T(0.5)};
    const Eigen::Vector3<T> ω = θ * axis;
    slp::SimilarityTwist3<T> twist{v, ω, σ};
    Eigen::Matrix4<T> G = Eigen::Matrix4<T>::Zero();
    G.template topLeftCorner<3, 3>() =
        σ * Eigen::Matrix3<T>::Identity() +
        Eigen::Matrix3<T>{
            {T(0), -ω[2], ω[1]}, {ω[2], T(0), -ω[0]}, {-ω[1], ω[0], T(0)}};
    G.template topRightCorner<3, 1>() = v;
    CHECK_THAT(twist.exp().to_matrix().value(),
               WithinAbs(
                   Eigen::Matrix4<T>{
                       G.template cast<double>().exp().template cast<T>()},
                   rounding_tolerance<T>(100)));

    auto log = twist.exp().log();
    CHECK_THAT(log.linear().value(),
               WithinAbs(Eigen::MatrixX<T>{v}, rounding_tolerance<T>(100)));
    CHECK_THAT(log.angular().value(),
               WithinAbs(Eigen::MatrixX<T>{ω}, rounding_tolerance<T>(100)));
    slp::Real<T> log_log_scale = log.log_scale();
    CHECK_THAT(log_log_scale.value(), WithinAbs(σ, rounding_tolerance<T>(10)));
  }

  // exp() of a twist is a one-parameter subgroup, so exp(ξ) = exp(ξ/2)²
  slp::SimilarityTwist3<T> ξ{Eigen::Vector3<T>{T(1), T(-2), T(0.5)},
                             Eigen::Vector3<T>{T(0.3), T(-1.2), T(2)}, T(0.7)};
  auto half = (ξ * T(0.5)).exp();
  CHECK_THAT((half * half).to_matrix().value(),
             WithinAbs(ξ.exp().to_matrix().value(), tolerance));

  // exp(log(S)) = S
  const slp::Similarity3<T> similarity{
      Eigen::Vector3<T>{T(3), T(1), T(-2)},
      Eigen::Quaternion<T>{Eigen::AngleAxis<T>{
          T(2.9), Eigen::Vector3<T>{T(1), T(1), T(1)}.normalized()}},
      T(0.2)};
  CHECK_THAT(similarity.log().exp().to_matrix().value(),
             WithinAbs(similarity.to_matrix().value(), tolerance));

  slp::SimilarityTwist3<T> sum = ξ + ξ - (-ξ);
  slp::Real<T> sum_log_scale = sum.log_scale();
  CHECK_THAT(sum_log_scale.value(), WithinAbs(T(2.1), tolerance));
  CHECK((ξ == T(2) * ξ).constraints.size() == 7);
}

TEMPLATE_TEST_CASE(
    "SimilarityTwist3 - Derivatives at zero and across the series switches",
    "[SimilarityTwist3]", SCALAR_TYPES) {
  using T = TestType;

  constexpr T ε = std::numeric_limits<T>::epsilon();

  slp::Problem<T> problem;
  auto linear = problem.real_decision_variable(3);
  auto angular = problem.real_decision_variable(3);
  auto log_scale = problem.real_decision_variable();
  slp::RealMatrix<T> wrt{7, 1};
  for (int i = 0; i < 3; ++i) {
    wrt[i] = linear[i];
    wrt[i + 3] = angular[i];
  }
  wrt[6] = log_scale;

  slp::SimilarityTwist3<T> ξ{linear, angular, log_scale};
  auto similarity = ξ.exp();
  auto twist = similarity.log();

  // A function that depends on every series
  slp::RealMatrix<T> out{10, 1};
  for (int i = 0; i < 3; ++i) {
    out[i] = slp::RealMatrix<T>{similarity.translation()}[i];
    out[i + 3] = twist.linear()[i];
    out[i + 6] = twist.angular()[i];
  }
  out[9] = twist.log_scale();
  slp::Real<T> f = similarity.translation().squared_distance(
                       slp::Translation3<T>{T(0), T(0), T(0)}) +
                   twist.linear().T() * twist.linear();

  slp::Jacobian J{out, wrt};
  slp::Hessian H{f, wrt};

  // At zero, exp() is the identity to first order, so the translation's
  // Jacobian is [I 0], and log(exp(ξ)) = ξ has Jacobian I
  linear.set_value(Eigen::Vector3<T>::Zero());
  angular.set_value(Eigen::Vector3<T>::Zero());
  log_scale.set_value(T(0));
  Eigen::MatrixX<T> J_expected{10, 7};
  J_expected.setZero();
  J_expected.topLeftCorner(3, 3).setIdentity();
  J_expected.bottomRows(7).setIdentity();
  CHECK_THAT(J.value().toDense(),
             WithinAbs(J_expected, rounding_tolerance<T>(1)));
  CHECK(Eigen::MatrixX<T>{H.value()}.allFinite());

  // The series and closed forms agree on either side of each switch: σ² + θ²
  // = 1, θ² = σ², and σ² = 1
  const Eigen::Vector3<T> axis =
      Eigen::Vector3<T>{T(1), T(-2), T(0.5)}.normalized();
  linear.set_value(Eigen::Vector3<T>{T(0.3), T(-1), T(2)});
  for (auto [σ, θ, vary_θ] :
       {std::tuple{T(0.6), T(0.8), true}, std::tuple{T(1), T(1), true},
        std::tuple{T(-1), T(0.5), false}, std::tuple{T(1), T(0.5), false}}) {
    CAPTURE(σ, θ, vary_θ);

    auto set = [&](T factor) {
      angular.set_value(Eigen::Vector3<T>{(vary_θ ? θ * factor : θ) * axis});
      log_scale.set_value(vary_θ ? σ : σ * factor);
    };

    set(T(1) - T(4) * ε);
    Eigen::MatrixX<T> J_below = J.value();
    Eigen::MatrixX<T> H_below = H.value();

    set(T(1) + T(4) * ε);
    Eigen::MatrixX<T> J_above = J.value();
    Eigen::MatrixX<T> H_above = H.value();

    CHECK_THAT(J_below, WithinAbs(J_above, rounding_tolerance<T>(100)));
    CHECK_THAT(H_below, WithinAbs(H_above, rounding_tolerance<T>(500)));
  }
}

TEMPLATE_TEST_CASE(
    "Problem - SimilarityTwist3 between two similarity transforms",
    "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Vector3<T> v_true{T(1), T(-2), T(0.5)};
  const Eigen::Vector3<T> ω_true{T(0.3), T(-1.2), T(2)};
  const T σ_true(-0.4);

  const slp::Similarity3<T> start{
      Eigen::Vector3<T>{T(-1), T(0.5), T(2)},
      Eigen::Quaternion<T>{Eigen::AngleAxis<T>{
          T(1), Eigen::Vector3<T>{T(0), T(1), T(1)}.normalized()}},
      T(1.5)};
  const auto end =
      start * slp::SimilarityTwist3<T>{v_true, ω_true, σ_true}.exp();

  slp::Problem<T> problem;
  slp::SimilarityTwist3<T> twist{problem.real_decision_variable(3),
                                 problem.real_decision_variable(3),
                                 problem.real_decision_variable()};
  problem.subject_to(start * twist.exp() == end);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(twist.linear().value(),
             WithinAbs(Eigen::MatrixX<T>{v_true}, solution_tolerance<T>(1)));
  CHECK_THAT(twist.angular().value(),
             WithinAbs(Eigen::MatrixX<T>{ω_true}, solution_tolerance<T>(1)));
  slp::Real<T> log_scale = twist.log_scale();
  CHECK_THAT(log_scale.value(), WithinAbs(σ_true, solution_tolerance<T>(1)));
}
