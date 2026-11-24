// Copyright (c) Sleipnir contributors

#include <limits>
#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/similarity2.hpp>
#include <sleipnir/autodiff/similarity_twist2.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <unsupported/Eigen/MatrixFunctions>  // IWYU pragma: keep

#include "catch_matchers.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("SimilarityTwist2 - exp() and log()", "[SimilarityTwist2]",
                   SCALAR_TYPES) {
  using T = TestType;

  constexpr T tolerance(1e-5);

  // exp() is the matrix exponential of the generator
  //
  //   [σ  −θ  v₀]
  //   [θ   σ  v₁]
  //   [0   0  0 ]
  for (auto [σ, θ] : {std::pair{T(0), T(0)}, std::pair{T(1e-9), T(-1e-9)},
                      std::pair{T(0.3), T(0)}, std::pair{T(0), T(2)},
                      std::pair{T(0.6), T(-0.8)}, std::pair{T(-1.5), T(0.2)},
                      std::pair{T(2), T(3)}}) {
    CAPTURE(σ, θ);

    const Eigen::Vector2<T> v{T(1), T(-2)};
    slp::SimilarityTwist2<T> twist{v, θ, σ};
    Eigen::Matrix3<T> G{{σ, -θ, v[0]}, {θ, σ, v[1]}, {T(0), T(0), T(0)}};
    CHECK_THAT(twist.exp().to_matrix().value(),
               WithinAbs(
                   Eigen::Matrix3<T>{
                       G.template cast<double>().exp().template cast<T>()},
                   rounding_tolerance<T>(100)));

    auto log = twist.exp().log();
    CHECK_THAT(log.linear().value(),
               WithinAbs(Eigen::MatrixX<T>{v}, rounding_tolerance<T>(100)));
    slp::Real<T> log_angular = log.angular();
    CHECK_THAT(log_angular.value(), WithinAbs(θ, rounding_tolerance<T>(10)));
    slp::Real<T> log_log_scale = log.log_scale();
    CHECK_THAT(log_log_scale.value(), WithinAbs(σ, rounding_tolerance<T>(10)));
  }

  // exp() of a twist is a one-parameter subgroup, so exp(ξ) = exp(ξ/2)²
  slp::SimilarityTwist2<T> ξ{Eigen::Vector2<T>{T(1), T(-2)}, T(2.5), T(0.7)};
  auto half = (ξ * T(0.5)).exp();
  CHECK_THAT((half * half).to_matrix().value(),
             WithinAbs(ξ.exp().to_matrix().value(), tolerance));

  // exp(log(S)) = S
  const slp::Similarity2<T> similarity{Eigen::Vector2<T>{T(3), T(1)}, T(-2.9),
                                       T(0.2)};
  CHECK_THAT(similarity.log().exp().to_matrix().value(),
             WithinAbs(similarity.to_matrix().value(), tolerance));

  // Twist arithmetic
  slp::SimilarityTwist2<T> sum = ξ + ξ - (-ξ);
  CHECK_THAT(
      sum.linear().value(),
      WithinAbs(Eigen::MatrixX<T>{Eigen::Vector2<T>{T(3), T(-6)}}, tolerance));
  slp::Real<T> sum_log_scale = sum.log_scale();
  CHECK_THAT(sum_log_scale.value(), WithinAbs(T(2.1), tolerance));
  CHECK((ξ == T(2) * ξ).constraints.size() == 4);
}

TEMPLATE_TEST_CASE(
    "SimilarityTwist2 - Derivatives at zero and across the series switches",
    "[SimilarityTwist2]", SCALAR_TYPES) {
  using T = TestType;

  constexpr T ε = std::numeric_limits<T>::epsilon();

  slp::Problem<T> problem;
  auto linear = problem.real_decision_variable(2);
  auto angular = problem.real_decision_variable();
  auto log_scale = problem.real_decision_variable();
  slp::RealMatrix<T> wrt{{linear[0]}, {linear[1]}, {angular}, {log_scale}};

  slp::SimilarityTwist2<T> ξ{linear, angular, log_scale};
  auto similarity = ξ.exp();
  auto twist = similarity.log();

  // A function that depends on every series
  slp::RealMatrix<T> out{6, 1};
  for (int i = 0; i < 2; ++i) {
    out[i] = slp::RealMatrix<T>{similarity.translation()}[i];
    out[i + 2] = twist.linear()[i];
  }
  out[4] = twist.angular();
  out[5] = twist.log_scale();
  slp::Real<T> f = similarity.translation().squared_distance(
                       slp::Translation2<T>{T(0), T(0)}) +
                   twist.linear().T() * twist.linear();

  slp::Jacobian J{out, wrt};
  slp::Hessian H{f, wrt};

  // At zero, exp() is the identity to first order, so the translation's
  // Jacobian is [I 0], and log(exp(ξ)) = ξ has Jacobian I
  linear.set_value(Eigen::Vector2<T>::Zero());
  angular.set_value(T(0));
  log_scale.set_value(T(0));
  Eigen::MatrixX<T> J_expected{6, 4};
  J_expected.setZero();
  J_expected.topLeftCorner(2, 2).setIdentity();
  J_expected.bottomRows(4).setIdentity();
  CHECK_THAT(J.value().toDense(),
             WithinAbs(J_expected, rounding_tolerance<T>(1)));
  CHECK(Eigen::MatrixX<T>{H.value()}.allFinite());

  // The series and closed forms agree on either side of each switch: σ² + θ²
  // = 1, θ² = σ², and σ² = 1
  linear.set_value(Eigen::Vector2<T>{T(0.3), T(-1)});
  for (auto [σ, θ, vary_θ] :
       {std::tuple{T(0.6), T(-0.8), true}, std::tuple{T(1), T(1), true},
        std::tuple{T(-1), T(0.5), false}, std::tuple{T(1), T(0.5), false}}) {
    CAPTURE(σ, θ, vary_θ);

    auto set = [&](T factor) {
      angular.set_value(vary_θ ? θ * factor : θ);
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
    "Problem - SimilarityTwist2 between two similarity transforms",
    "[Problem][Manifold]", SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Vector2<T> v_true{T(1), T(-2)};
  const T θ_true(2);
  const T σ_true(-0.4);

  const slp::Similarity2<T> start{Eigen::Vector2<T>{T(-1), T(0.5)}, T(1),
                                  T(1.5)};
  const auto end =
      start * slp::SimilarityTwist2<T>{v_true, θ_true, σ_true}.exp();

  slp::Problem<T> problem;
  slp::SimilarityTwist2<T> twist{problem.real_decision_variable(2),
                                 problem.real_decision_variable(),
                                 problem.real_decision_variable()};
  problem.subject_to(start * twist.exp() == end);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(twist.linear().value(),
             WithinAbs(Eigen::MatrixX<T>{v_true}, solution_tolerance<T>(1)));
  slp::Real<T> angular = twist.angular();
  CHECK_THAT(angular.value(), WithinAbs(θ_true, solution_tolerance<T>(1)));
  slp::Real<T> log_scale = twist.log_scale();
  CHECK_THAT(log_scale.value(), WithinAbs(σ_true, solution_tolerance<T>(1)));
}
