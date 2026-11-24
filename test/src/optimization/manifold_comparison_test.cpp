// Copyright (c) Sleipnir contributors

#include <cmath>
#include <numbers>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/unit_vector.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_matchers.hpp"
#include "scalar_types.hpp"

namespace {

template <typename L, typename R>
concept EqualityComparable = requires(L lhs, R rhs) { lhs == rhs; };

template <typename L, typename R>
concept LessEqualComparable = requires(L lhs, R rhs) { lhs <= rhs; };

using UV = slp::UnitVector<double>;
using VM = slp::RealMatrix<double>;

// A unit vector equals only another unit vector or a constant direction
static_assert(EqualityComparable<UV, UV>);
static_assert(EqualityComparable<UV, Eigen::Vector3d>);
static_assert(EqualityComparable<Eigen::Vector3d, UV>);
static_assert(!EqualityComparable<UV, VM>);
static_assert(!EqualityComparable<VM, UV>);
static_assert(!EqualityComparable<UV, slp::Real<double>>);
static_assert(!EqualityComparable<slp::Real<double>, UV>);

// Inequalities are componentwise bounds
static_assert(LessEqualComparable<UV, VM>);
static_assert(LessEqualComparable<VM, UV>);

}  // namespace

TEMPLATE_TEST_CASE("Manifold comparisons - UnitVector equality",
                   "[ManifoldComparison]", SCALAR_TYPES) {
  using T = TestType;
  using std::acos;
  using std::hypot;

  slp::Problem<T> problem;
  auto a = problem.unit_vector_decision_variable(3);
  auto b = problem.unit_vector_decision_variable(3);
  a.set_value(Eigen::Vector3<T>{T(0.6), T(0.8), T(0)});
  b.set_value(Eigen::Vector3<T>{T(0.6), T(0.8), T(0)});

  constexpr T tolerance(1e-5);

  // Unit vectors in ℝ³ have two degrees of freedom
  auto c = a == b;
  CHECK(c.constraints.size() == 2);
  for (auto& constraint : c.constraints) {
    CHECK_THAT(constraint.value(), WithinAbs(T(0), tolerance));
  }

  // The components are the logarithm map, so their norm is the angle
  b.set_value(Eigen::Vector3<T>{T(0), T(0.8), T(0.6)});
  T expected_angle = acos(T(0.64));
  for (auto&& constraints : {a == b, a == Eigen::Vector3<T>{T(0), T(4), T(3)},
                             Eigen::Vector3<T>{T(0), T(4), T(3)} == a}) {
    REQUIRE(constraints.constraints.size() == 2);
    T norm = hypot(slp::Real<T>{constraints.constraints[0]}.value(),
                   slp::Real<T>{constraints.constraints[1]}.value());
    CHECK_THAT(norm, WithinAbs(expected_angle, tolerance));
  }

  // The opposite vector isn't a solution
  b.set_value(Eigen::Vector3<T>{T(-0.6), T(-0.8), T(0.01)});
  auto opposite = a == b;
  T norm = hypot(slp::Real<T>{opposite.constraints[0]}.value(),
                 slp::Real<T>{opposite.constraints[1]}.value());
  CHECK(norm > T(3));
}
