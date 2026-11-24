// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

#include "sleipnir/autodiff/real.hpp"

// The exponential and logarithm maps of rotations and poses use functions of
// an angle θ, like sin θ/θ, that are 0/0 at θ = 0 and whose derivatives
// through √(θ²) are infinite there. These evaluate them as power series in
// s = θ² for s < 1 and in closed form otherwise, so their values and
// derivatives are accurate everywhere. With 11 terms, each series' truncation
// error is below ε for s < 1.

namespace slp::detail {

/// Number of series terms.
inline constexpr size_t trig_series_terms = 11;

/// Returns Σₖ cₖsᵏ for s < 1 and closed_form(√s) otherwise.
///
/// @param s The argument s = θ².
/// @param coeffs The series coefficients cₖ.
/// @param closed_form The closed form as a function of θ.
/// @return The function's value.
template <typename Scalar, typename ClosedForm>
Real<Scalar> series_or_closed_form(
    const Real<Scalar>& s, const std::array<Scalar, trig_series_terms>& coeffs,
    ClosedForm&& closed_form) {
  // Both branches are evaluated, and an unselected branch's infinite partial
  // derivatives would turn its zero adjoint into NaN, so each branch gets a
  // safe constant input when it's not selected
  const Real<Scalar> one{Scalar(1)};
  auto select = [&](const Real<Scalar>& series_value,
                    const Real<Scalar>& closed_form_value) {
    return if_else(s < one, series_value, closed_form_value);
  };

  Real<Scalar> series_s = select(s, Real<Scalar>{Scalar(0)});
  Real<Scalar> series{coeffs.back()};
  for (int k = static_cast<int>(trig_series_terms) - 2; k >= 0; --k) {
    series = coeffs[k] + series_s * series;
  }

  Real<Scalar> θ = sqrt(select(one, s));

  return select(series, std::forward<ClosedForm>(closed_form)(θ));
}

/// Returns the coefficients (−1)ᵏ/(2k + m)!.
///
/// @param m The factorial's offset.
/// @return The coefficients.
template <typename Scalar>
std::array<Scalar, trig_series_terms> alternating_factorial_coeffs(int m) {
  std::array<Scalar, trig_series_terms> coeffs;
  Scalar factorial(1);
  for (int i = 2; i <= m; ++i) {
    factorial *= Scalar(i);
  }
  for (int k = 0; k < static_cast<int>(trig_series_terms); ++k) {
    coeffs[k] = Scalar(k % 2 == 0 ? 1 : -1) / factorial;
    factorial *= Scalar(2 * k + m + 1) * Scalar(2 * k + m + 2);
  }
  return coeffs;
}

/// Returns cos θ.
///
/// @param s θ².
/// @return cos θ.
template <typename Scalar>
Real<Scalar> cos_of_sqrt(const Real<Scalar>& s) {
  static const auto coeffs = alternating_factorial_coeffs<Scalar>(0);
  return series_or_closed_form(s, coeffs,
                               [](const Real<Scalar>& θ) { return cos(θ); });
}

/// Returns sin θ/θ.
///
/// @param s θ².
/// @return sin θ/θ.
template <typename Scalar>
Real<Scalar> sin_over_theta(const Real<Scalar>& s) {
  static const auto coeffs = alternating_factorial_coeffs<Scalar>(1);
  return series_or_closed_form(
      s, coeffs, [](const Real<Scalar>& θ) { return sin(θ) / θ; });
}

}  // namespace slp::detail
