// Copyright (c) Sleipnir contributors

#include <cmath>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/similarity2.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("Similarity2 - Default value and set_value", "[Similarity2]",
                   SCALAR_TYPES) {
  using T = TestType;

  using std::log;

  slp::Problem<T> problem;
  auto similarity = problem.similarity2_decision_variable();

  CHECK(similarity.translation().value() == Eigen::Vector2<T>::Zero());
  CHECK(similarity.rotation().value() == T(0));
  CHECK(similarity.log_scale().value() == T(0));
  CHECK(similarity.scale().value() == T(1));

  similarity.translation().set_value(Eigen::Vector2<T>{T(1), T(-2)});
  similarity.rotation().set_value(T(0.5));
  similarity.log_scale().set_value(log(T(3)));

  constexpr T tolerance(1e-6);
  CHECK(similarity.translation().x().value() == T(1));
  CHECK(similarity.translation().y().value() == T(-2));
  CHECK_THAT(similarity.rotation().value(), WithinAbs(T(0.5), tolerance));
  CHECK_THAT(similarity.scale().value(), WithinAbs(T(3), tolerance));
}

TEMPLATE_TEST_CASE("Similarity2 - Parameter", "[Similarity2]", SCALAR_TYPES) {
  using T = TestType;
  using enum slp::ExpressionType;

  slp::Problem<T> problem;
  auto similarity = problem.similarity2_parameter();

  CHECK(similarity.translation().value() == Eigen::Vector2<T>::Zero());
  CHECK(similarity.rotation().value() == T(0));
  CHECK(similarity.log_scale().value() == T(0));
  CHECK(similarity.translation().x().type() == PARAMETER);
  CHECK(similarity.rotation().sin().type() == PARAMETER);
  CHECK(similarity.log_scale().type() == PARAMETER);
}

TEMPLATE_TEST_CASE("Similarity2 - Operations", "[Similarity2]", SCALAR_TYPES) {
  using T = TestType;

  using std::cos;
  using std::log;
  using std::sin;

  slp::Problem<T> problem;
  auto similarity1 = problem.similarity2_decision_variable();
  similarity1.translation().set_value(Eigen::Vector2<T>{T(1), T(2)});
  similarity1.rotation().set_value(T(0.7));
  similarity1.log_scale().set_value(log(T(1.5)));
  const slp::Similarity2<T> similarity2{Eigen::Vector2<T>{T(-0.5), T(0.25)},
                                        T(-2.1), T(0.4)};

  auto to_eigen = [](T x, T y, T θ, T s) {
    Eigen::Matrix3<T> result{{s * cos(θ), -s * sin(θ), x},
                             {s * sin(θ), s * cos(θ), y},
                             {T(0), T(0), T(1)}};
    return result;
  };
  const Eigen::Matrix3<T> T1 = to_eigen(T(1), T(2), T(0.7), T(1.5));
  const Eigen::Matrix3<T> T2 = to_eigen(T(-0.5), T(0.25), T(-2.1), T(0.4));

  constexpr T tolerance(1e-5);

  CHECK_THAT(similarity1.to_matrix().value(), WithinAbs(T1, tolerance));
  CHECK_THAT((similarity1 * similarity2).to_matrix().value(),
             WithinAbs(Eigen::Matrix3<T>{T1 * T2}, tolerance));
  CHECK_THAT(similarity1.inverse().to_matrix().value(),
             WithinAbs(Eigen::Matrix3<T>{T1.inverse()}, tolerance));
  CHECK_THAT((similarity1 * similarity1.inverse()).to_matrix().value(),
             WithinAbs(Eigen::Matrix3<T>::Identity(), tolerance));
  CHECK_THAT(similarity1.relative_to(similarity2).to_matrix().value(),
             WithinAbs(Eigen::Matrix3<T>{T2.inverse() * T1}, tolerance));

  // Similarity transforms can also be made from a Translation2, a Rotation2,
  // and a log scale
  const slp::Similarity2<T> similarity2_from_parts{
      slp::Translation2<T>{T(-0.5), T(0.25)}, slp::Rotation2<T>{T(-2.1)},
      slp::Real<T>{log(T(0.4))}};
  CHECK_THAT(similarity2_from_parts.to_matrix().value(),
             WithinAbs(T2, tolerance));

  const Eigen::Vector2<T> p{T(0.3), T(-1.2)};
  Eigen::Vector2<T> expected = (T1 * p.homogeneous()).head(2);
  CHECK_THAT(slp::Translation2<T>{p}.transform_by(similarity1).value(),
             WithinAbs(expected, tolerance));
}

TEMPLATE_TEST_CASE("Similarity2 - Equality constraint", "[Similarity2]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto similarity = problem.similarity2_decision_variable();
  const slp::Similarity2<T> target{Eigen::Vector2<T>{T(2), T(-1)}, T(3),
                                   T(0.5)};

  auto constraint = similarity == target;
  CHECK(constraint.constraints.size() == 4);

  problem.subject_to(constraint);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  constexpr T tolerance(1e-5);
  CHECK_THAT(similarity.translation().value(),
             WithinAbs(Eigen::Vector2<T>{T(2), T(-1)}, tolerance));
  CHECK_THAT(similarity.rotation().value(), WithinAbs(T(3), tolerance));
  CHECK_THAT(similarity.scale().value(), WithinAbs(T(0.5), tolerance));
}
