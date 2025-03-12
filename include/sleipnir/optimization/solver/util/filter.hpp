// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/util/constraint_violation.hpp"

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
      : FilterEntry{f, slp::constraint_violation(c_e)} {}

  /// Constructs an interior-point method filter entry.
  ///
  /// @param f The cost function value.
  /// @param v The log-domain variables.
  /// @param c_e The equality constraint values (nonzero means violation).
  /// @param c_i The inequality constraint values (negative means violation).
  /// @param sqrt_μ Square root of the barrier parameter.
  FilterEntry(Scalar f, DenseVector& v, const DenseVector& c_e,
              const DenseVector& c_i, Scalar sqrt_μ) {
    // s = √(μ)e⁻ᵛ
    DenseVector s = sqrt_μ * (-v).array().exp().matrix();

    cost = f - sqrt_μ * sqrt_μ * s.array().log().sum();
    constraint_violation = slp::constraint_violation(c_e, c_i, s);
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
  Scalar min_constraint_violation;

  /// The maximum constraint violation
  Scalar max_constraint_violation;

  /// Constructs an empty filter.
  ///
  /// @param initial_constraint_violation The optimization problem's initial
  ///     constraint violation.
  explicit constexpr Filter(Scalar initial_constraint_violation = Scalar(0)) {
    using std::max;

    min_constraint_violation =
        Scalar(1e-4) * max(Scalar(1), initial_constraint_violation);
    max_constraint_violation =
        Scalar(1e4) * max(Scalar(1), initial_constraint_violation);
  }

  /// Resets the filter.
  void reset() {
    m_filter.clear();
    m_last_rejection_due_to_filter = false;
    m_resets = 0;
  }

  /// Resets the filter and lowers the maximum constraint violation if the
  /// filter may be impeding progress.
  ///
  /// @param current_entry The entry corresponding to the current iterate.
  /// @return True if the filter was reset.
  bool try_reset(const FilterEntry<Scalar>& current_entry) {
    constexpr int max_resets = 5;

    // The filter may be impeding progress if it rejected the most recent trial
    // entry, and the maximum constraint violation is more than a tenth of the
    // current iterate's. The number of resets is capped until the next reset().
    //
    // See section 3.2 case I of [2].
    if (m_resets >= max_resets ||
        max_constraint_violation <=
            current_entry.constraint_violation / Scalar(10) ||
        !m_last_rejection_due_to_filter) {
      return false;
    }

    max_constraint_violation *= Scalar(0.1);
    m_filter.clear();
    m_last_rejection_due_to_filter = false;
    ++m_resets;

    return true;
  }

  /// Returns true if the given trial entry is acceptable to the filter.
  ///
  /// @param current_entry The entry corresponding to the current iterate.
  /// @param trial_entry The entry corresponding to the trial iterate.
  /// @param D_ϕ Cost directional derivative along the search direction.
  /// @param α The step size (0, 1].
  /// @return True if the given entry is acceptable to the filter.
  bool try_add(const FilterEntry<Scalar>& current_entry,
               const FilterEntry<Scalar>& trial_entry, Scalar D_ϕ, Scalar α) {
    using std::isfinite;
    using std::pow;

    // Reject steps with nonfinite cost or constraint violation above maximum
    if (!isfinite(trial_entry.cost) ||
        trial_entry.constraint_violation > max_constraint_violation) {
      return false;
    }

    // Switching condition
    bool switching_condition =
        D_ϕ < Scalar(0) &&
        α * pow(-D_ϕ, s_ϕ) > pow(current_entry.constraint_violation, s_θ);

    // Armijo condition
    constexpr Scalar η_ϕ(1e-8);
    bool armijo_condition =
        trial_entry.cost <= current_entry.cost + η_ϕ * α * D_ϕ;

    // Sufficient decrease condition
    bool sufficient_decrease =
        trial_entry.cost <=
            current_entry.cost - γ_cost * current_entry.constraint_violation ||
        trial_entry.constraint_violation <=
            (Scalar(1) - γ_constraint) * current_entry.constraint_violation;

    // If constraint violation is below threshold and switching condition is
    // true, check Armijo condition for step rejection. Otherwise, check
    // sufficient decrease condition.
    if (current_entry.constraint_violation <= min_constraint_violation &&
        switching_condition) {
      if (!armijo_condition) {
        m_last_rejection_due_to_filter = false;
        return false;
      }
    } else if (!sufficient_decrease) {
      m_last_rejection_due_to_filter = false;
      return false;
    }

    // Reject steps in filter (i.e., dominated by any filter entry)
    if (in_filter(trial_entry)) {
      m_last_rejection_due_to_filter = true;
      return false;
    }

    // Augment filter with accepted iterate if switching condition or Armijo
    // condition are false
    if (!switching_condition || !armijo_condition) {
      add(current_entry);
    }

    return true;
  }

  /// Adds the given entry to the filter.
  ///
  /// The entry is shifted by the sufficient decrease margins, so iterates that
  /// don't sufficiently decrease the cost or constraint violation relative to
  /// it are rejected afterward.
  ///
  /// @param entry The entry to add to the filter.
  void add(const FilterEntry<Scalar>& entry) {
    FilterEntry<Scalar> margin_entry{
        entry.cost - γ_cost * entry.constraint_violation,
        (Scalar(1) - γ_constraint) * entry.constraint_violation};

    // Remove dominated entries
    erase_if(m_filter,
             [&](const auto& elem) { return elem.dominated_by(margin_entry); });

    m_filter.push_back(margin_entry);
  }

  /// Returns true if the given entry is acceptable to the filter.
  ///
  /// @param entry The entry.
  /// @return True if the given entry is acceptable to the filter.
  bool is_acceptable(const FilterEntry<Scalar>& entry) const {
    using std::isfinite;

    return isfinite(entry.cost) &&
           entry.constraint_violation <= max_constraint_violation &&
           !in_filter(entry);
  }

  /// Returns the minimum step size of the backtracking line search.
  ///
  /// Below this step size, no trial iterate can satisfy the sufficient decrease
  /// condition or both the switching condition and Armijo condition, so
  /// feasibility restoration should be invoked.
  ///
  /// @param current_entry The entry corresponding to the current iterate.
  /// @param D_ϕ Cost directional derivative along the search direction.
  /// @return The minimum step size.
  Scalar min_step_size(const FilterEntry<Scalar>& current_entry,
                       Scalar D_ϕ) const {
    // The minimum step size goes to zero as the constraint violation does so
    // feasibility restoration is never invoked near feasible points where the
    // search direction is a descent direction.
    //
    // See equation (23) of [2].

    using std::min;
    using std::pow;

    // Minimum step size safety factor
    constexpr Scalar γ_α(0.05);

    const Scalar θ = current_entry.constraint_violation;

    if (D_ϕ >= Scalar(0)) {
      return γ_α * γ_constraint;
    }

    Scalar α_min = min(γ_constraint, γ_cost * θ / -D_ϕ);
    if (θ <= min_constraint_violation) {
      α_min = min(α_min, pow(θ, s_θ) / pow(-D_ϕ, s_ϕ));
    }
    return γ_α * α_min;
  }

 private:
  static constexpr Scalar γ_cost{1e-8};
  static constexpr Scalar γ_constraint{1e-5};

  // Switching condition exponents
  static constexpr Scalar s_ϕ{2.3};
  static constexpr Scalar s_θ{1.1};

  gch::small_vector<FilterEntry<Scalar>> m_filter;

  bool m_last_rejection_due_to_filter = false;

  // The number of resets done by try_reset() since the last reset()
  int m_resets = 0;

  /// Returns true if the given entry is in the filter.
  ///
  /// @param entry The entry.
  /// @return True if the given entry is in the filter.
  bool in_filter(const FilterEntry<Scalar>& entry) const {
    // An entry is in the filter if it's dominated by any filter entry
    return std::any_of(m_filter.begin(), m_filter.end(), [&](const auto& elem) {
      return entry.dominated_by(elem);
    });
  }
};

/// Detects a solver stall.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class StallDetector {
 public:
  /// Returns true if the solver stalled.
  ///
  /// @param current_entry The entry corresponding to the current iterate.
  /// @param trial_entry The entry corresponding to the accepted trial iterate.
  /// @param α The accepted step size.
  /// @param min_constraint_violation The minimum constraint violation from the
  ///     filter.
  /// @return True if the solver stalled.
  bool update(const FilterEntry<Scalar>& current_entry,
              const FilterEntry<Scalar>& trial_entry, Scalar α,
              Scalar min_constraint_violation) {
    // If step is tiny, current iterate is infeasible, and accepted trial
    // iterate made insufficient progress toward feasibility, increment stalled
    // step counter
    if (α < Scalar(1e-3) &&
        current_entry.constraint_violation > min_constraint_violation &&
        trial_entry.constraint_violation >
            Scalar(0.99) * current_entry.constraint_violation) {
      ++m_stalled_step_counter;

      // If there were too many consecutive tiny steps, report a stall
      if (m_stalled_step_counter >= 5) {
        m_stalled_step_counter = 0;
        return true;
      }
    } else {
      // A break in stalled steps resets the counter
      m_stalled_step_counter = 0;
    }

    return false;
  }

 private:
  int m_stalled_step_counter = 0;
};

}  // namespace slp
