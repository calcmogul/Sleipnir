// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/rotation2.hpp>
#include <sleipnir/autodiff/translation2.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Rotation2 - Default value and set_value", "[Rotation2]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto z = problem.rotation2_decision_variable();

  CHECK(z.value() == T(0));
  CHECK(z.angle().value() == T(0));

  constexpr T tolerance(1e-6);

  z.set_value(T(0.5));
  CHECK_THAT(z.value(), WithinAbs(T(0.5), tolerance));
  CHECK_THAT(z.angle().value(), WithinAbs(T(0.5), tolerance));

  using std::cos;
  using std::sin;
  slp::Real<T> c = z.cos();
  slp::Real<T> s = z.sin();
  CHECK_THAT(c.value(), WithinAbs(cos(T(0.5)), tolerance));
  CHECK_THAT(s.value(), WithinAbs(sin(T(0.5)), tolerance));

  // Angles are reported in (−π, π]
  z.set_value(T(4));
  CHECK_THAT(z.value(), WithinAbs(T(4) - T(2 * std::numbers::pi), tolerance));
}

TEMPLATE_TEST_CASE("Rotation2 - Parameter", "[Rotation2]", SCALAR_TYPES) {
  using T = TestType;
  using enum slp::ExpressionType;

  slp::Problem<T> problem;
  auto z = problem.rotation2_parameter();

  CHECK(z.value() == T(0));
  CHECK(z.cos().type() == PARAMETER);
  CHECK(z.sin().type() == PARAMETER);

  z.set_value(T(0.5));
  CHECK_THAT(z.value(), WithinAbs(T(0.5), T(1e-6)));
}

TEMPLATE_TEST_CASE("Rotation2 - Operations", "[Rotation2]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto z1 = problem.rotation2_decision_variable();
  auto v = problem.unit_vector_decision_variable(2);
  z1.set_value(T(0.7));
  v.set_value(Eigen::Vector2<T>{T(0.6), T(0.8)});
  slp::Rotation2<T> z2{T(-2.1)};

  constexpr T tolerance(1e-5);

  // Composition adds angles, and inversion negates them
  CHECK_THAT((z1 * z2).value(), WithinAbs(T(0.7 - 2.1), tolerance));
  CHECK_THAT(z1.inverse().value(), WithinAbs(T(-0.7), tolerance));
  CHECK_THAT((z1 * z1.inverse()).value(), WithinAbs(T(0), tolerance));

  using std::cos;
  using std::sin;
  const Eigen::Matrix2<T> R{{cos(T(0.7)), -sin(T(0.7))},
                            {sin(T(0.7)), cos(T(0.7))}};
  CHECK_THAT(z1.to_matrix().value(), WithinAbs(R, tolerance));

  // Rotating a UnitVector returns a UnitVector
  slp::UnitVector<T> rotated_v = v.rotate_by(z1);
  Eigen::VectorX<T> rotated_v_expected = R * Eigen::Vector2<T>{T(0.6), T(0.8)};
  CHECK_THAT(rotated_v.value(), WithinAbs(rotated_v_expected, tolerance));

  // Rotating a Translation2 returns a Translation2
  const Eigen::Vector2<T> p{T(1.5), T(-2)};
  slp::Translation2<T> rotated_p = slp::Translation2<T>{p}.rotate_by(z1);
  Eigen::Vector2<T> rotated_p_expected = R * p;
  CHECK_THAT(rotated_p.value(), WithinAbs(rotated_p_expected, tolerance));
}

TEMPLATE_TEST_CASE("Rotation2 - angle() gradient", "[Rotation2]",
                   SCALAR_TYPES) {
  using T = TestType;
  using std::cos;
  using std::sin;

  slp::Problem<T> problem;
  auto z = problem.rotation2_decision_variable();
  slp::RealMatrix<T> wrt{{z.cos()}, {z.sin()}};
  slp::Gradient g{z.angle(), wrt};

  // ∂θ/∂(cos θ, sin θ) = (−sin θ, cos θ) on the unit circle
  for (T θ : {T(-3), T(-1), T(0), T(0.5), T(2)}) {
    z.set_value(θ);
    Eigen::VectorX<T> expected{{-sin(θ), cos(θ)}};
    CHECK_THAT(Eigen::VectorX<T>{g.value()},
               WithinAbs(expected, rounding_tolerance<T>(5)));
  }
}
