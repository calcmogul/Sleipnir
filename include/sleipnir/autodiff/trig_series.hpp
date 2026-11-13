// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

#include "sleipnir/autodiff/variable.hpp"

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
Variable<Scalar> series_or_closed_form(
    const Variable<Scalar>& s,
    const std::array<Scalar, trig_series_terms>& coeffs,
    ClosedForm&& closed_form) {
  // Both branches are evaluated, and an unselected branch's infinite partial
  // derivatives would turn its zero adjoint into NaN, so each branch gets a
  // safe constant input when it's not selected
  constexpr auto less = +[](Scalar a, Scalar b) { return a < b; };
  const Variable<Scalar> one{Scalar(1)};
  auto select = [&](const Variable<Scalar>& series_value,
                    const Variable<Scalar>& closed_form_value) {
    return if_else(less, s, one, series_value, closed_form_value);
  };

  Variable<Scalar> series_s = select(s, Variable<Scalar>{Scalar(0)});
  Variable<Scalar> series{coeffs.back()};
  for (int k = static_cast<int>(trig_series_terms) - 2; k >= 0; --k) {
    series = coeffs[k] + series_s * series;
  }

  Variable<Scalar> θ = sqrt(select(one, s));

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

/// Returns the coefficients (−1)ᵏB₂ₖ/(2k)! of (θ/2) cot(θ/2), where B₂ₖ are
/// the Bernoulli numbers.
///
/// @return The coefficients for k = 0, 1, …, 11.
template <typename Scalar>
std::array<Scalar, trig_series_terms + 1> half_cot_coeffs() {
  constexpr std::array<std::pair<double, double>, trig_series_terms + 1>
      bernoulli{{{1, 1},
                 {1, 6},
                 {-1, 30},
                 {1, 42},
                 {-1, 30},
                 {5, 66},
                 {-691, 2730},
                 {7, 6},
                 {-3617, 510},
                 {43867, 798},
                 {-174611, 330},
                 {854513, 138}}};

  std::array<Scalar, trig_series_terms + 1> coeffs;
  Scalar factorial(1);
  for (int k = 0; k < static_cast<int>(trig_series_terms) + 1; ++k) {
    if (k > 0) {
      factorial *= Scalar(2 * k - 1) * Scalar(2 * k);
    }
    coeffs[k] = Scalar(k % 2 == 0 ? 1 : -1) * Scalar(bernoulli[k].first) /
                Scalar(bernoulli[k].second) / factorial;
  }
  return coeffs;
}

/// Returns cos θ.
///
/// @param s θ².
/// @return cos θ.
template <typename Scalar>
Variable<Scalar> cos_of_sqrt(const Variable<Scalar>& s) {
  static const auto coeffs = alternating_factorial_coeffs<Scalar>(0);
  return series_or_closed_form(
      s, coeffs, [](const Variable<Scalar>& θ) { return cos(θ); });
}

/// Returns sin θ/θ.
///
/// @param s θ².
/// @return sin θ/θ.
template <typename Scalar>
Variable<Scalar> sin_over_theta(const Variable<Scalar>& s) {
  static const auto coeffs = alternating_factorial_coeffs<Scalar>(1);
  return series_or_closed_form(
      s, coeffs, [](const Variable<Scalar>& θ) { return sin(θ) / θ; });
}

/// Returns (1 − cos θ)/θ².
///
/// @param s θ².
/// @return (1 − cos θ)/θ².
template <typename Scalar>
Variable<Scalar> one_minus_cos_over_theta_sq(const Variable<Scalar>& s) {
  static const auto coeffs = alternating_factorial_coeffs<Scalar>(2);
  return series_or_closed_form(s, coeffs, [](const Variable<Scalar>& θ) {
    return (Scalar(1) - cos(θ)) / (θ * θ);
  });
}

/// Returns (θ − sin θ)/θ³.
///
/// @param s θ².
/// @return (θ − sin θ)/θ³.
template <typename Scalar>
Variable<Scalar> theta_minus_sin_over_theta_cubed(const Variable<Scalar>& s) {
  static const auto coeffs = alternating_factorial_coeffs<Scalar>(3);
  return series_or_closed_form(s, coeffs, [](const Variable<Scalar>& θ) {
    return (θ - sin(θ)) / (θ * θ * θ);
  });
}

/// Returns (θ/2) cot(θ/2) = θ sin θ/(2(1 − cos θ)) for |θ| < 2π.
///
/// @param s θ².
/// @return (θ/2) cot(θ/2).
template <typename Scalar>
Variable<Scalar> half_theta_cot_half_theta(const Variable<Scalar>& s) {
  static const auto coeffs = [] {
    auto all = half_cot_coeffs<Scalar>();
    std::array<Scalar, trig_series_terms> result;
    std::copy(all.begin(), all.end() - 1, result.begin());
    return result;
  }();
  return series_or_closed_form(s, coeffs, [](const Variable<Scalar>& θ) {
    return θ * sin(θ) / (Scalar(2) * (Scalar(1) - cos(θ)));
  });
}

/// Returns (1 − (θ/2) cot(θ/2))/θ² for |θ| < 2π.
///
/// @param s θ².
/// @return (1 − (θ/2) cot(θ/2))/θ².
template <typename Scalar>
Variable<Scalar> one_minus_half_theta_cot_half_theta_over_theta_sq(
    const Variable<Scalar>& s) {
  // The series is (1 − Σₖ cₖsᵏ)/s = −Σₖ cₖ₊₁sᵏ
  static const auto coeffs = [] {
    auto all = half_cot_coeffs<Scalar>();
    std::array<Scalar, trig_series_terms> result;
    for (size_t k = 0; k < trig_series_terms; ++k) {
      result[k] = -all[k + 1];
    }
    return result;
  }();
  return series_or_closed_form(s, coeffs, [](const Variable<Scalar>& θ) {
    return (Scalar(1) - θ * sin(θ) / (Scalar(2) * (Scalar(1) - cos(θ)))) /
           (θ * θ);
  });
}

}  // namespace slp::detail
