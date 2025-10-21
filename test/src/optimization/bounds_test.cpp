// Copyright (c) Sleipnir contributors

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#define CATCH_CONFIG_ENABLE_PAIR_STRINGMAKER
#include <catch2/catch_tostring.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/matchers/catch_matchers_quantifiers.hpp>

#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/optimization/bounds.hpp>
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Bounds - Detection", "[Bounds]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  constexpr auto inf = std::numeric_limits<T>::infinity();

  slp::Variable<T> x, y, z, w, v;
  auto decision_variables = std::to_array<slp::Variable<T>>({x, y, z, w, v});

  slp::Variable a = -z - T(1e-12);
  // We assume these imply constraints in the form c(x) ≥ 0
  auto inequality_constraint_variables = std::to_array<slp::Variable<T>>({
      x + y - T(3),
      x * y,
      T(3),
      x - T(3),
      x - T(4),
      (T(3) + T(4) * y - y) * T(4),
      slp::sin(w),
      a,
      -z,
      v - T(8),
      -v + T(8),
      -v + T(7),
      -v + T(6.5),
  });

  constexpr auto correct_bounds = std::to_array<std::pair<T, T>>({
      {T(4), inf},
      {T(-1), inf},
      {-inf, T(-1e-12)},
      {-inf, inf},
      {T(8), T(6.5)},
  });
  static_assert(correct_bounds.size() == decision_variables.size());
  const Eigen::Vector<bool, inequality_constraint_variables.size()>
      correct_bound_constraint_mask{
          false, false, false, true, true, true, false,
          true,  true,  true,  true, true, true,
      };
  constexpr auto correct_conflicting_bounds =
      std::to_array<std::pair<Eigen::Index, Eigen::Index>>({
          {9, 11},
          {9, 12},
      });

  slp::VariableMatrix<T> x_ad{decision_variables};
  slp::VariableMatrix<T> c_i_ad{inequality_constraint_variables};
  slp::Jacobian A_i{c_i_ad, x_ad};
  const auto [bound_constraint_mask, decision_var_indices_to_bounds,
              conflicting_bounds] =
      slp::get_bounds<T>(decision_variables, inequality_constraint_variables,
                         A_i.value());

  using Catch::Matchers::RangeEquals, Catch::Matchers::UnorderedRangeEquals;
  CHECK_THAT(decision_var_indices_to_bounds, RangeEquals(correct_bounds));
  CHECK_THAT(bound_constraint_mask, RangeEquals(correct_bound_constraint_mask));
  CHECK_THAT(conflicting_bounds,
             UnorderedRangeEquals(correct_conflicting_bounds));
}

TEMPLATE_TEST_CASE("Bounds - Projection", "[Bounds]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using std::abs;
  using std::isfinite;

  constexpr auto inf = std::numeric_limits<T>::infinity();

  auto bounds = std::to_array<std::pair<T, T>>({
      {-inf, inf},
      {-inf, T(3)},
      {T(2), T(2)},
      {T(12), T(12.1)},
      {T(-1), T(-1e-12)},
      {T(2), inf},
  });
  Eigen::Vector<T, bounds.size()> x, x_correct;

  // This tests that we exactly match section 3.6 in [2]
  SECTION("Initial value already mostly in bounds") {
    constexpr T κ_1{1e-2};
    constexpr T κ_2{1e-2};
    x.setZero();
    x_correct << T(0), T(0), T(2), T(12) + κ_2 * T(0.1),
        T(-1e-12) - std::min(κ_1, κ_2 * T(1 - 1e-12)), T(2) + κ_1 * T(2);
    slp::project_onto_bounds(x, bounds, κ_1, κ_2);
    CHECK(x == x_correct);
  }

  // This tests that we match the spirit of bound projection, without relying on
  // any details of the specific method
  const auto bounds_are_sane = [](auto x, auto bounds, bool stickToLower) {
    for (size_t i = 0; i < bounds.size(); i++) {
      const auto& [lower, upper] = bounds[i];
      if (stickToLower && isfinite(lower)) {            // NOLINT
        CHECK(abs(lower - x[i]) <= abs(upper - x[i]));  // NOLINT
      } else if (isfinite(upper)) {                     // NOLINT
        CHECK(abs(lower - x[i]) >= abs(upper - x[i]));  // NOLINT
      }
      if (lower == upper) {
        CHECK(lower == x[i]);
      } else {
        CHECK(lower < x[i]);
        CHECK(x[i] < upper);
      }
    }
  };
  SECTION("Initial value below all bounds") {
    constexpr T big_negative{-1000};
    x.setConstant(big_negative);
    slp::project_onto_bounds(x, bounds);
    bounds_are_sane(x, bounds, true);
  }
  SECTION("Initial value above all bounds") {
    constexpr T big_positive{1000};
    x.setConstant(big_positive);
    x_correct << big_positive, T(3), T(2), T(12.1), T(-1e-12), big_positive;
    slp::project_onto_bounds(x, bounds);
    bounds_are_sane(x, bounds, false);
  }
}
