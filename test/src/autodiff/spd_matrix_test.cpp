// Copyright (c) Sleipnir contributors

#include <concepts>
#include <utility>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/spd_matrix.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "scalar_types.hpp"

namespace {

// Comparisons that would be elementwise or ambiguous with the semidefinite
// ordering don't compile
template <typename T>
concept Comparable = requires(T lhs, T rhs) {
  lhs < rhs;
  lhs <= rhs;
  lhs > rhs;
  lhs >= rhs;
};

template <typename L, typename R>
concept LessEqualComparable = requires(L lhs, R rhs) { lhs <= rhs; };

template <typename L, typename R>
concept GreaterEqualComparable = requires(L lhs, R rhs) { lhs >= rhs; };

template <typename L, typename R>
concept EqualityComparable = requires(L lhs, R rhs) { lhs == rhs; };

using SPD = slp::SPDMatrix<double>;
using View = slp::SymmetricView<double>;
using VM = slp::RealMatrix<double>;

static_assert(!Comparable<SPD>);
static_assert(!Comparable<View>);
static_assert(!LessEqualComparable<SPD, VM>);
static_assert(!LessEqualComparable<VM, SPD>);
static_assert(!GreaterEqualComparable<SPD, VM>);
static_assert(!GreaterEqualComparable<VM, SPD>);
static_assert(!GreaterEqualComparable<SPD, double>);
static_assert(!LessEqualComparable<View, double>);
static_assert(!GreaterEqualComparable<View, VM>);

// An SPD matrix can't equal a scalar, but a symmetric view can
static_assert(!EqualityComparable<SPD, double>);
static_assert(EqualityComparable<View, double>);
static_assert(EqualityComparable<double, View>);
static_assert(EqualityComparable<View, View>);
static_assert(EqualityComparable<View, VM>);
static_assert(EqualityComparable<VM, View>);

// Closed operations keep the type
static_assert(
    std::same_as<decltype(std::declval<SPD>() + std::declval<SPD>()), SPD>);
static_assert(std::same_as<decltype(std::declval<SPD>().T()), SPD>);
static_assert(
    std::same_as<decltype(std::declval<SPD>() - std::declval<SPD>()), VM>);
static_assert(std::same_as<decltype(-std::declval<SPD>()), VM>);

}  // namespace

TEMPLATE_TEST_CASE("SPDMatrix - Elements alias across the diagonal",
                   "[SPDMatrix]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto P = problem.spd_decision_variable(3);

  CHECK(P.rows() == 3);
  CHECK(P.cols() == 3);
  CHECK(P.value() == Eigen::Matrix3<T>::Identity());

  const Eigen::Matrix3<T> value{
      {T(4), T(1), T(0.5)}, {T(1), T(2), T(-0.3)}, {T(0.5), T(-0.3), T(1)}};
  P.set_value(value);
  CHECK(P.value() == value);
  CHECK(P.value(0, 2) == T(0.5));
  CHECK(P.value(2, 0) == T(0.5));
}

TEMPLATE_TEST_CASE("SPDMatrix - Equality constrains the lower triangle",
                   "[SPDMatrix]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto P = problem.spd_decision_variable(3);
  auto Q = problem.spd_decision_variable(3);
  auto A = problem.real_decision_variable(3, 3);
  slp::RealMatrix<T> G = A.T() * A;
  const Eigen::Matrix3<T> S = Eigen::Matrix3<T>::Identity();

  CHECK((P == G).constraints.size() == 6);
  CHECK((G == P).constraints.size() == 6);
  CHECK((P == A.T() * A).constraints.size() == 6);
  CHECK((P == S).constraints.size() == 6);
  CHECK((S == P).constraints.size() == 6);
  CHECK((P == Q).constraints.size() == 6);
  CHECK((P == G.symmetric_view()).constraints.size() == 6);
  CHECK((G.symmetric_view() == P).constraints.size() == 6);

  // Element constraints are one per element
  CHECK((P[0, 1] == T(0)).constraints.size() == 1);
  CHECK((P[2, 2] <= T(3)).constraints.size() == 1);
}

TEMPLATE_TEST_CASE("SPDMatrix - Constant", "[SPDMatrix]", SCALAR_TYPES) {
  using T = TestType;

  const Eigen::Matrix2<T> value{{T(2), T(1)}, {T(1), T(3)}};
  slp::SPDMatrix<T> P{value};

  CHECK(P.value() == value);
  CHECK(P.trace().value() == T(5));
  CHECK((P + P).value() == T(2) * value);
  CHECK(slp::RealMatrix<T>{P * value}.value() == value * value);
}

TEMPLATE_TEST_CASE("SymmetricView - Reads the lower triangle",
                   "[SymmetricView]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;
  auto A = problem.real_decision_variable(2, 3);
  A.set_value(Eigen::Matrix<T, 2, 3>{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}});
  slp::RealMatrix<T> G = A.T() * A;

  auto view = G.symmetric_view();
  CHECK(view.rows() == 3);
  CHECK(view.cols() == 3);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      // (row, col) and (col, row) are the lower-triangle element
      auto lower = row >= col ? G[row, col] : G[col, row];
      CHECK(slp::Real<T>{view[row, col]}.value() == lower.value());
    }
  }

  CHECK((view == T(0)).constraints.size() == 6);
  CHECK((T(0) == view).constraints.size() == 6);
  CHECK((view == G).constraints.size() == 6);
  CHECK((G == view).constraints.size() == 6);
  CHECK((view == view).constraints.size() == 6);
  CHECK((view == Eigen::Matrix3<T>::Zero()).constraints.size() == 6);

  // A square block of a larger matrix
  auto M = problem.real_decision_variable(4, 4);
  CHECK((M.block(0, 0, 3, 3).symmetric_view() == T(0)).constraints.size() == 6);
}
