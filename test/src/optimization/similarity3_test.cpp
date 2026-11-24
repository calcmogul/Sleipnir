// Copyright (c) Sleipnir contributors

#include <cmath>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/rotation3.hpp>
#include <sleipnir/autodiff/similarity3.hpp>
#include <sleipnir/autodiff/translation3.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Similarity3 - Default value and set_value", "[Similarity3]",
                   SCALAR_TYPES) {
  using T = TestType;

  using std::log;

  slp::Problem<T> problem;
  auto similarity = problem.similarity3_decision_variable();

  CHECK(similarity.translation().value() == Eigen::Vector3<T>::Zero());
  CHECK(similarity.rotation().value().coeffs() ==
        Eigen::Quaternion<T>::Identity().coeffs());
  CHECK(similarity.log_scale().value() == T(0));
  CHECK(similarity.scale().value() == T(1));

  similarity.translation().set_value(Eigen::Vector3<T>{T(1), T(-2), T(3)});
  similarity.rotation().set_value(Eigen::Quaternion<T>{T(2), T(0), T(0), T(0)});
  similarity.log_scale().set_value(log(T(3)));

  constexpr T tolerance(1e-6);
  CHECK(similarity.translation().z().value() == T(3));
  CHECK(similarity.rotation().value().w() == T(1));
  CHECK_THAT(similarity.scale().value(), WithinAbs(T(3), tolerance));
}

TEMPLATE_TEST_CASE("Similarity3 - Parameter", "[Similarity3]", SCALAR_TYPES) {
  using T = TestType;
  using enum slp::ExpressionType;

  std::vector<slp::Translation3<T>> points{{T(1), T(0), T(0)},
                                           {T(0), T(2), T(0)},
                                           {T(0), T(0), T(-1)},
                                           {T(1), T(1), T(1)}};

  slp::Problem<T> problem;
  auto similarity = problem.similarity3_decision_variable();
  auto target = problem.similarity3_parameter();

  CHECK(target.translation().value() == Eigen::Vector3<T>::Zero());
  CHECK(target.rotation().value().coeffs() ==
        Eigen::Quaternion<T>::Identity().coeffs());
  CHECK(target.log_scale().value() == T(0));
  CHECK(target.translation().x().type() == PARAMETER);
  CHECK(target.rotation().w().type() == PARAMETER);
  CHECK(target.log_scale().type() == PARAMETER);

  slp::Real<T> cost{T(0)};
  for (auto& p : points) {
    cost += p.transform_by(similarity).squared_distance(p.transform_by(target));
  }
  problem.minimize(cost);

  for (auto [t, q, log_scale] :
       {std::tuple{
            Eigen::Vector3<T>{T(1), T(-0.5), T(2)},
            Eigen::Quaternion<T>{Eigen::AngleAxis<T>{
                T(0.7), Eigen::Vector3<T>{T(0.2), T(1), T(-0.4)}.normalized()}},
            T(0.3)},
        std::tuple{
            Eigen::Vector3<T>{T(-1), T(2), T(0)},
            Eigen::Quaternion<T>{Eigen::AngleAxis<T>{
                T(2.5), Eigen::Vector3<T>{T(1), T(0), T(0.5)}.normalized()}},
            T(-0.2)}}) {
    target.translation().set_value(t);
    target.rotation().set_value(q);
    target.log_scale().set_value(log_scale);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    Eigen::Vector3<T> t_value = similarity.translation().value();
    for (int i = 0; i < 3; ++i) {
      CHECK_THAT(t_value[i], WithinAbs(t[i], solution_tolerance<T>(1)));
    }
    using std::abs;
    CHECK_THAT(abs(similarity.rotation().value().dot(q)),
               WithinAbs(T(1), solution_tolerance<T>(1)));
    CHECK_THAT(similarity.log_scale().value(),
               WithinAbs(log_scale, solution_tolerance<T>(1)));
  }
}

TEMPLATE_TEST_CASE("Similarity3 - Operations", "[Similarity3]", SCALAR_TYPES) {
  using T = TestType;

  using std::log;

  const Eigen::Quaternion<T> q1{Eigen::AngleAxis<T>{
      T(0.7), Eigen::Vector3<T>{T(1), T(2), T(-1)}.normalized()}};
  const Eigen::Quaternion<T> q2{Eigen::AngleAxis<T>{
      T(-2.1), Eigen::Vector3<T>{T(0), T(-1), T(3)}.normalized()}};
  const Eigen::Vector3<T> t1{T(1), T(2), T(3)};
  const Eigen::Vector3<T> t2{T(-0.5), T(0.25), T(1.5)};
  const T s1(1.5);
  const T s2(0.4);

  slp::Problem<T> problem;
  auto similarity1 = problem.similarity3_decision_variable();
  similarity1.translation().set_value(t1);
  similarity1.rotation().set_value(q1);
  similarity1.log_scale().set_value(log(s1));
  const slp::Similarity3<T> similarity2{t2, q2, s2};

  auto to_eigen = [](const Eigen::Vector3<T>& t, const Eigen::Quaternion<T>& q,
                     T s) {
    Eigen::Matrix4<T> result = Eigen::Matrix4<T>::Identity();
    result.template topLeftCorner<3, 3>() = s * q.toRotationMatrix();
    result.template topRightCorner<3, 1>() = t;
    return result;
  };
  const Eigen::Matrix4<T> T1 = to_eigen(t1, q1, s1);
  const Eigen::Matrix4<T> T2 = to_eigen(t2, q2, s2);

  constexpr T tolerance(1e-5);

  CHECK_THAT(similarity1.to_matrix().value(), WithinAbs(T1, tolerance));
  CHECK_THAT((similarity1 * similarity2).to_matrix().value(),
             WithinAbs(Eigen::Matrix4<T>{T1 * T2}, tolerance));
  CHECK_THAT(similarity1.inverse().to_matrix().value(),
             WithinAbs(Eigen::Matrix4<T>{T1.inverse()}, tolerance));
  CHECK_THAT((similarity1 * similarity1.inverse()).to_matrix().value(),
             WithinAbs(Eigen::Matrix4<T>::Identity(), tolerance));
  CHECK_THAT(similarity1.relative_to(similarity2).to_matrix().value(),
             WithinAbs(Eigen::Matrix4<T>{T2.inverse() * T1}, tolerance));

  // Similarity transforms can also be made from a Translation3, a Rotation3,
  // and a log scale
  const slp::Similarity3<T> similarity2_from_parts{
      slp::Translation3<T>{t2}, slp::Rotation3<T>{q2}, slp::Real<T>{log(s2)}};
  CHECK_THAT(similarity2_from_parts.to_matrix().value(),
             WithinAbs(T2, tolerance));

  const Eigen::Vector3<T> p{T(0.3), T(-1.2), T(0.8)};
  Eigen::Vector3<T> expected = (T1 * p.homogeneous()).head(3);
  CHECK_THAT(slp::Translation3<T>{p}.transform_by(similarity1).value(),
             WithinAbs(expected, tolerance));
}

TEMPLATE_TEST_CASE("Similarity3 - Equality constraint", "[Similarity3]",
                   SCALAR_TYPES) {
  using T = TestType;

  using std::abs;

  const Eigen::Quaternion<T> q{Eigen::AngleAxis<T>{
      T(2.5), Eigen::Vector3<T>{T(0.2), T(1), T(-0.4)}.normalized()}};
  const Eigen::Vector3<T> t{T(1), T(-0.5), T(2)};

  slp::Problem<T> problem;
  auto similarity = problem.similarity3_decision_variable();
  const slp::Similarity3<T> target{t, q, T(2)};

  auto constraint = similarity == target;
  CHECK(constraint.constraints.size() == 7);

  problem.subject_to(constraint);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  constexpr T tolerance(1e-5);
  CHECK_THAT(similarity.translation().value(), WithinAbs(t, tolerance));
  CHECK_THAT(abs(similarity.rotation().value().dot(q)),
             WithinAbs(T(1), tolerance));
  CHECK_THAT(similarity.scale().value(), WithinAbs(T(2), tolerance));
}
