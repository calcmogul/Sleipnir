// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <gch/small_vector.hpp>

// See docs/algorithms.md#Works_cited for citation definitions.

namespace slp {

/// Filter entry consisting of cost and constraint violation.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct FilterEntry {
  /// Type alias for dense vector.
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  /// The cost function's value
  Scalar cost{0};

  /// The constraint violation
  Scalar constraint_violation{0};

  constexpr FilterEntry() = default;

  /// Constructs a FilterEntry.
  ///
  /// @param cost The cost function's value.
  /// @param constraint_violation The constraint violation.
  explicit constexpr FilterEntry(Scalar cost,
                                 Scalar constraint_violation = Scalar(0))
      : cost{cost}, constraint_violation{constraint_violation} {}

  /// Constructs a Sequential Quadratic Programming filter entry.
  ///
  /// @param f The cost function value.
  /// @param c_e The equality constraint values (nonzero means violation).
  FilterEntry(Scalar f, const DenseVector& c_e)
      : FilterEntry{f, c_e.template lpNorm<1>()} {}

  /// Constructs an interior-point method filter entry.
  ///
  /// @param f The cost function value.
  /// @param v The log-domain variables.
  /// @param c_i The inequality constraint values (negative means violation).
  /// @param sqrt_μ Square root of the barrier parameter.
  FilterEntry(Scalar f, DenseVector& v, const DenseVector& c_i, Scalar sqrt_μ) {
    // s = √(μ)e⁻ᵛ
    DenseVector s = sqrt_μ * (-v).array().exp().matrix();

    cost = f - sqrt_μ * sqrt_μ * s.array().log().sum();
    constraint_violation = (c_i - s).template lpNorm<1>();
  }

  /// Returns true if this filter entry is dominated by another.
  ///
  /// @param entry The other entry.
  /// @return True if this filter entry is dominated by another.
  constexpr bool dominated_by(const FilterEntry<Scalar>& entry) const {
    return entry.cost <= cost &&
           entry.constraint_violation <= constraint_violation;
  }
};

/// Step filter.
///
/// See the section on filters in chapter 15 of [1].
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class Filter {
 public:
  /// The minimum constraint violation
  static constexpr Scalar min_constraint_violation{1e-4};

  /// The maximum constraint violation
  Scalar max_constraint_violation{1e4};

  /// Constructs an empty filter.
  Filter() {
    // Initial filter entry rejects constraint violations above max
    m_filter.emplace_back(std::numeric_limits<Scalar>::infinity(),
                          max_constraint_violation);
  }

  /// Resets the filter.
  void reset() {
    m_filter.clear();

    // Initial filter entry rejects constraint violations above max
    m_filter.emplace_back(std::numeric_limits<Scalar>::infinity(),
                          max_constraint_violation);
  }

  /// Returns true if the given iterate is accepted by the filter.
  ///
  /// @param entry The entry to attempt adding to the filter.
  /// @param α The step size (0, 1].
  /// @return True if the given iterate is accepted by the filter.
  bool try_add(const FilterEntry<Scalar>& entry, Scalar α) {
    if (is_acceptable(entry, α)) {
      add(entry);
      return true;
    } else {
      return false;
    }
  }

  /// Returns the most recently added filter entry.
  ///
  /// @return The most recently added filter entry.
  const FilterEntry<Scalar>& back() const { return m_filter.back(); }

 private:
  static constexpr Scalar γ_cost{1e-8};
  static constexpr Scalar γ_constraint{1e-5};

  gch::small_vector<FilterEntry<Scalar>> m_filter;

  /// Adds a new entry to the filter.
  ///
  /// @param entry The entry to add to the filter.
  void add(const FilterEntry<Scalar>& entry) {
    // Remove dominated entries
    erase_if(m_filter,
             [&](const auto& elem) { return elem.dominated_by(entry); });

    m_filter.push_back(entry);
  }

  /// Returns true if the given entry is acceptable to the filter.
  ///
  /// @param entry The entry to check.
  /// @param α The step size (0, 1].
  /// @return True if the given entry is acceptable to the filter.
  bool is_acceptable(const FilterEntry<Scalar>& entry, Scalar α) {
    using std::isfinite;
    using std::pow;

    if (!isfinite(entry.cost) || !isfinite(entry.constraint_violation)) {
      return false;
    }

    // If current filter entry is better than all prior ones in some respect,
    // accept it.
    //
    // See equation (2.13) of [4].
    Scalar ϕ = pow(α, Scalar(1.5));
    if (entry.constraint_violation <= min_constraint_violation) {
      // Accept only optimality improvement when constraint violation is low
      return std::ranges::all_of(m_filter, [&](const auto& elem) {
        return entry.cost <= elem.cost - ϕ * γ_cost * elem.constraint_violation;
      });
    } else {
      // Accept either optimality or constraint violation improvement
      return std::ranges::all_of(m_filter, [&](const auto& elem) {
        return entry.cost <=
                   elem.cost - ϕ * γ_cost * elem.constraint_violation ||
               entry.constraint_violation <=
                   (Scalar(1) - ϕ * γ_constraint) * elem.constraint_violation;
      });
    }
  }
};

}  // namespace slp
