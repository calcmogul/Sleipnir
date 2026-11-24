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

/// Returns (1 − cos θ)/θ².
///
/// @param s θ².
/// @return (1 − cos θ)/θ².
template <typename Scalar>
Real<Scalar> one_minus_cos_over_theta_sq(const Real<Scalar>& s) {
  static const auto coeffs = alternating_factorial_coeffs<Scalar>(2);
  return series_or_closed_form(s, coeffs, [](const Real<Scalar>& θ) {
    return (Scalar(1) - cos(θ)) / (θ * θ);
  });
}

/// Returns (θ − sin θ)/θ³.
///
/// @param s θ².
/// @return (θ − sin θ)/θ³.
template <typename Scalar>
Real<Scalar> theta_minus_sin_over_theta_cubed(const Real<Scalar>& s) {
  static const auto coeffs = alternating_factorial_coeffs<Scalar>(3);
  return series_or_closed_form(s, coeffs, [](const Real<Scalar>& θ) {
    return (θ - sin(θ)) / (θ * θ * θ);
  });
}

/// Returns (θ/2) cot(θ/2) = θ sin θ/(2(1 − cos θ)) for |θ| < 2π.
///
/// @param s θ².
/// @return (θ/2) cot(θ/2).
template <typename Scalar>
Real<Scalar> half_theta_cot_half_theta(const Real<Scalar>& s) {
  static const auto coeffs = [] {
    auto all = half_cot_coeffs<Scalar>();
    std::array<Scalar, trig_series_terms> result;
    std::copy(all.begin(), all.end() - 1, result.begin());
    return result;
  }();
  return series_or_closed_form(s, coeffs, [](const Real<Scalar>& θ) {
    return θ * sin(θ) / (Scalar(2) * (Scalar(1) - cos(θ)));
  });
}

/// Returns (1 − (θ/2) cot(θ/2))/θ² for |θ| < 2π.
///
/// @param s θ².
/// @return (1 − (θ/2) cot(θ/2))/θ².
template <typename Scalar>
Real<Scalar> one_minus_half_theta_cot_half_theta_over_theta_sq(
    const Real<Scalar>& s) {
  // The series is (1 − Σₖ cₖsᵏ)/s = −Σₖ cₖ₊₁sᵏ
  static const auto coeffs = [] {
    auto all = half_cot_coeffs<Scalar>();
    std::array<Scalar, trig_series_terms> result;
    for (size_t k = 0; k < trig_series_terms; ++k) {
      result[k] = -all[k + 1];
    }
    return result;
  }();
  return series_or_closed_form(s, coeffs, [](const Real<Scalar>& θ) {
    return (Scalar(1) - θ * sin(θ) / (Scalar(2) * (Scalar(1) - cos(θ)))) /
           (θ * θ);
  });
}

/// Number of series terms for similarity_coeffs().
inline constexpr int similarity_series_terms = 20;

/// The coefficients of a similarity transform's exponential map.
///
/// With log scale σ and rotation angle θ,
///
///   A = ∫₀¹ e^(σt) dt = (e^σ − 1)/σ
///   B = 1/θ ∫₀¹ e^(σt) sin θt dt
///   C = 1/θ² ∫₀¹ e^(σt) (1 − cos θt) dt
template <typename Scalar>
struct SimilarityCoeffs {
  /// A = (e^σ − 1)/σ.
  Real<Scalar> A;

  /// B = 1/θ ∫₀¹ e^(σt) sin θt dt.
  Real<Scalar> B;

  /// C = 1/θ² ∫₀¹ e^(σt) (1 − cos θt) dt.
  Real<Scalar> C;
};

/// Returns (e^σ − 1)/σ.
///
/// It's a power series in σ for σ² < 1 and the closed form otherwise.
///
/// @param σ The argument.
/// @return (e^σ − 1)/σ.
template <typename Scalar>
Real<Scalar> expm1_over_x(const Real<Scalar>& σ) {
  const Real<Scalar> zero{Scalar(0)};
  const Real<Scalar> one{Scalar(1)};
  Real<Scalar> σ_sq = σ * σ;
  auto select = [&](const Real<Scalar>& series_value,
                    const Real<Scalar>& closed_form_value) {
    return if_else(σ_sq < one, series_value, closed_form_value);
  };

  // Σₙ σⁿ/(n + 1)!
  Real<Scalar> series_σ = select(σ, zero);
  Scalar factorial(1);
  for (int n = 2; n <= similarity_series_terms; ++n) {
    factorial *= Scalar(n);
  }
  Real<Scalar> series{Scalar(1) / factorial};
  for (int n = similarity_series_terms - 2; n >= 0; --n) {
    factorial /= Scalar(n + 2);
    series = Scalar(1) / factorial + series_σ * series;
  }

  Real<Scalar> closed_σ = select(one, σ);
  return select(series, (exp(closed_σ) - Scalar(1)) / closed_σ);
}

/// Returns the coefficients of a similarity transform's exponential map.
///
/// They're power series in σ and θ² for σ² + θ² < 1 and closed forms
/// otherwise, so their values and derivatives are accurate everywhere. With 20
/// terms, each series' truncation error is below ε for σ² + θ² < 1.
///
/// @param σ The log scale.
/// @param s θ², where θ is the rotation angle.
/// @return The coefficients.
template <typename Scalar>
SimilarityCoeffs<Scalar> similarity_coeffs(const Real<Scalar>& σ,
                                           const Real<Scalar>& s) {
  // Both branches are evaluated, and an unselected branch's infinite partial
  // derivatives would turn its zero adjoint into NaN, so each branch gets a
  // safe constant input when it's not selected
  const Real<Scalar> zero{Scalar(0)};
  const Real<Scalar> one{Scalar(1)};
  Real<Scalar> λ_sq = σ * σ + s;
  auto select = [&](const Real<Scalar>& series_value,
                    const Real<Scalar>& closed_form_value) {
    return if_else(λ_sq < one, series_value, closed_form_value);
  };

  // With λ = σ + iθ and λⁿ = pₙ + iqₙ, the series are
  //
  //   A = Σₙ σⁿ/(n + 1)!
  //   B = Σₙ Qₙ/(n + 1)!, Qₙ = qₙ/θ
  //   C = Σₙ Pₙ/(n + 1)!, Pₙ = (σⁿ − pₙ)/θ²
  //
  // λⁿ⁺¹ = λλⁿ gives Qₙ₊₁ = σQₙ + pₙ and Pₙ₊₁ = σPₙ + Qₙ, with
  // pₙ = σⁿ − θ²Pₙ, so no term divides by θ.
  Real<Scalar> series_σ = select(σ, zero);
  Real<Scalar> series_s = select(s, zero);
  Real<Scalar> series_A{Scalar(0)};
  Real<Scalar> series_B{Scalar(0)};
  Real<Scalar> series_C{Scalar(0)};
  Real<Scalar> σ_n{Scalar(1)};
  Real<Scalar> Q{Scalar(0)};
  Real<Scalar> P{Scalar(0)};
  Scalar factorial(1);
  for (int n = 0; n < similarity_series_terms; ++n) {
    factorial *= Scalar(n + 1);
    series_A += σ_n / factorial;
    if (n >= 1) {
      series_B += Q / factorial;
    }
    if (n >= 2) {
      series_C += P / factorial;
    }
    Real<Scalar> p = σ_n - series_s * P;
    P = series_σ * P + Q;
    Q = series_σ * Q + p;
    σ_n *= series_σ;
  }

  // The closed forms use C = (A − Re f)/θ² for θ² ≥ σ² and
  //
  //   C = (e^σσ²(1 − cos θ)/θ² + e^σ − 1 − σe^σ sin θ/θ)/(σ(σ² + θ²))
  //
  // otherwise, where f = (e^λ − 1)/λ, so neither divides by a small θ² or σ.
  Real<Scalar> closed_σ = select(one, σ);
  Real<Scalar> closed_s = select(one, s);
  Real<Scalar> e = exp(closed_σ);
  Real<Scalar> L = closed_σ * closed_σ + closed_s;
  Real<Scalar> sin_θ_over_θ = sin_over_theta(closed_s);
  Real<Scalar> c2 = one_minus_cos_over_theta_sq(closed_s);
  Real<Scalar> cos_θ = Scalar(1) - closed_s * c2;
  Real<Scalar> closed_A = expm1_over_x(closed_σ);
  Real<Scalar> closed_B =
      (e * sin_θ_over_θ * closed_σ - e * cos_θ + Scalar(1)) / L;

  Real<Scalar> σ_sq = closed_σ * closed_σ;
  auto select_C = [&](const Real<Scalar>& σ_form, const Real<Scalar>& θ_form) {
    return if_else(closed_s < σ_sq, σ_form, θ_form);
  };

  Real<Scalar> θ_form_s = select_C(one, closed_s);
  Real<Scalar> re_f =
      ((e * cos_θ - Scalar(1)) * closed_σ + e * closed_s * sin_θ_over_θ) / L;
  Real<Scalar> θ_form_C = (closed_A - re_f) / θ_form_s;

  Real<Scalar> σ_form_σ = select_C(closed_σ, one);
  Real<Scalar> σ_form_C =
      (e * σ_sq * c2 + e - Scalar(1) - closed_σ * e * sin_θ_over_θ) /
      (σ_form_σ * L);

  return {select(series_A, closed_A), select(series_B, closed_B),
          select(series_C, select_C(σ_form_C, θ_form_C))};
}

}  // namespace slp::detail
