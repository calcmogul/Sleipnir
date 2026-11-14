// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/util/assert.hpp"

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

/// Bound constraint metadata.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Bounds {
  /// Which constraints, if any, are bound constraints.
  Eigen::ArrayX<bool> bound_constraint_mask;

  /// The tightest bounds on each decision variable.
  gch::small_vector<std::pair<Scalar, Scalar>> bounds;

  /// Whether or not the constraints are feasible (given previously encountered
  /// bounds).
  gch::small_vector<std::pair<Eigen::Index, Eigen::Index>>
      conflicting_bound_indices;
};

/// A "bound constraint" is any linear constraint in one scalar variable.
///
/// Computes which constraints, if any, are bound constraints, the tightest
/// bounds on each decision variable, and whether or not they're feasible (given
/// previously encountered bounds),
///
/// @tparam Scalar Scalar type.
/// @param decision_variables Decision variables corresponding to each column of
///     A_i.
/// @param inequality_constraints Variables representing the left-hand side of
///     cᵢ(decision_variables) ≥ 0.
/// @param A_i The Jacobian of inequality_constraints wrt decision_variables,
///     evaluated anywhere.
template <typename Scalar>
Bounds<Scalar> get_bounds(std::span<Variable<Scalar>> decision_variables,
                          std::span<Variable<Scalar>> inequality_constraints,
                          const Eigen::SparseMatrix<Scalar>& A_i) {
  // NB: Casting to long is unspecified if the size of decision_variable.size()
  // is greater than the max long value, but then we wouldn't be able to fill
  // A_i correctly anyway.
  slp_assert(static_cast<Eigen::Index>(decision_variables.size()) ==
             A_i.cols());
  slp_assert(static_cast<Eigen::Index>(inequality_constraints.size()) ==
             A_i.rows());

  // For each row of A_i, count its structural nonzeros and record the column
  // and value of its most recently visited nonzero. If the count is 1, that's
  // the row's only nonzero; otherwise, the recorded column and value are
  // meaningless, but those rows aren't bound constraints, so they're never
  // read.
  gch::small_vector<int> row_num_nonzeros(A_i.rows(), 0);
  gch::small_vector<int> row_nonzero_col(A_i.rows());
  gch::small_vector<Scalar> row_nonzero_value(A_i.rows());
  for (int col = 0; col < A_i.outerSize(); ++col) {
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it{A_i, col}; it;
         ++it) {
      ++row_num_nonzeros[it.row()];
      row_nonzero_col[it.row()] = it.col();
      row_nonzero_value[it.row()] = it.value();
    }
  }

  // Maps each decision variable's index to the indices of its upper and lower
  // bounds if they exist, or NO_BOUND if they do not; used only for bookkeeping
  // to compute conflicting bounds
  constexpr Eigen::Index NO_BOUND = -1;
  gch::small_vector<std::pair<Eigen::Index, Eigen::Index>>
      decision_var_indices_to_constraint_indices{decision_variables.size(),
                                                 {NO_BOUND, NO_BOUND}};
  // Lists pairs of indices of bound constraints in the inequality constraint
  // list that conflict with each other
  gch::small_vector<std::pair<Eigen::Index, Eigen::Index>>
      conflicting_bound_indices;

  // Maps each decision variable's index to its upper and lower bounds
  gch::small_vector<std::pair<Scalar, Scalar>> decision_var_indices_to_bounds{
      decision_variables.size(),
      {-std::numeric_limits<Scalar>::infinity(),
       std::numeric_limits<Scalar>::infinity()}};

  // Vector corresponding to the inequality constraints where the i-th element
  // is 1 if the i-th inequality constraint is a bound and 0 otherwise.
  Eigen::ArrayX<bool> bound_constraint_mask{inequality_constraints.size()};
  bound_constraint_mask.fill(false);

  for (typename decltype(inequality_constraints)::size_type constraint_index =
           0;
       constraint_index < inequality_constraints.size(); ++constraint_index) {
    // A constraint is a bound iff it is linear and its gradient has a
    // single nonzero value.
    if (inequality_constraints[constraint_index].type() !=
        ExpressionType::LINEAR) {
      continue;
    }
    if (row_num_nonzeros[constraint_index] != 1) {
      // Constraint is not a bound because either:
      //
      // 1. It's in more than one variable
      // 2. It's in no decision variables (e.g., it only depends on variables
      //    that aren't decision variables of this problem)
      continue;
    }

    // Claim: The bound given by a bound constraint is the constraint evaluated
    // at zero divided by the nonzero element of the constraint's gradient.
    //
    // Proof: If c(x) is a bound constraint, then by definition c is a linear
    // function in one variable, hence there exist a, b ∈ ℝ s.t. c(x) = axᵢ + b
    // and a ≠ 0. The gradient of c is then aeᵢ (where eᵢ denotes the i-th basis
    // element), and c(0) = b. If c(x) ≥ 0, then since either a < 0 or a > 0, we
    // have either x ≤ -b/a or x ≥ -b/a, respectively. ∎
    //
    // The nonzero value of the j-th constraint's gradient.
    const auto constraint_coefficient = row_nonzero_value[constraint_index];

    // The Jacobian's sparsity pattern is structural, so the single stored value
    // can be zero if the constraint's terms cancel (e.g., x − x + 1 ≥ 0). The
    // constraint is then constant and not a bound.
    if (constraint_coefficient == Scalar(0)) {
      continue;
    }

    const auto decision_variable_index = row_nonzero_col[constraint_index];
    const auto decision_variable_value =
        decision_variables[decision_variable_index].value();
    Scalar constraint_constant;
    // We need to evaluate this constraint *exactly* at zero.
    if (decision_variable_value != Scalar(0)) {
      decision_variables[decision_variable_index].set_value(Scalar(0));
      constraint_constant = inequality_constraints[constraint_index].value();
      decision_variables[decision_variable_index].set_value(
          decision_variable_value);
    } else {
      constraint_constant = inequality_constraints[constraint_index].value();
    }

    using std::isfinite;

    // The constraint's constant or coefficient can be nonfinite if the user
    // wrote a constraint like x ≥ −∞ or ∞x ≥ 1. Note that this constraint will
    // cause the solver to report a nonfinite initial guess later.
    if (!isfinite(constraint_constant) || !isfinite(constraint_coefficient)) {
      continue;
    }

    // Update bounds; we assume constraints of the form c(x) ≥ 0.
    auto& [lower_bound, upper_bound] =
        decision_var_indices_to_bounds[decision_variable_index];
    auto& [lower_index, upper_index] =
        decision_var_indices_to_constraint_indices[decision_variable_index];
    const auto detected_bound = -constraint_constant / constraint_coefficient;
    if (constraint_coefficient < Scalar(0) && detected_bound < upper_bound) {
      upper_bound = detected_bound;
      upper_index = constraint_index;
    } else if (constraint_coefficient > Scalar(0) &&
               detected_bound > lower_bound) {
      lower_bound = detected_bound;
      lower_index = constraint_index;
    }

    // Update conflicting bounds
    if (lower_bound > upper_bound) {
      conflicting_bound_indices.emplace_back(lower_index, upper_index);
    }

    // Set the bound constraint mask appropriately.
    bound_constraint_mask[constraint_index] = true;
  }
  return {bound_constraint_mask, decision_var_indices_to_bounds,
          conflicting_bound_indices};
}

/// Projects the decision variables onto the given bounds, while ensuring some
/// configurable distance from the boundary if possible. This is designed to
/// match the algorithm given in section 3.6 of [2].
///
/// @param x A vector of decision variables.
/// @param decision_variable_indices_to_bounds An array of bounds (stored
///     [lower, upper]) for each decision variable in x.
/// @param κ_1 A constant controlling distance from the lower or upper bound
///     when the difference between the upper and lower bound is small.
/// @param κ_2 A constant controlling distance from the lower or upper bound
///     when the difference between the upper and lower bound is large
///     (including when one of the bounds is ±∞).
template <typename Derived>
  requires(static_cast<bool>(Eigen::DenseBase<Derived>::IsVectorAtCompileTime))
void project_onto_bounds(
    Eigen::DenseBase<Derived>& x,
    std::span<const std::pair<typename Eigen::DenseBase<Derived>::Scalar,
                              typename Eigen::DenseBase<Derived>::Scalar>>
        decision_variable_indices_to_bounds,
    const typename Eigen::DenseBase<Derived>::Scalar κ_1 =
        typename Eigen::DenseBase<Derived>::Scalar(1e-2),
    const typename Eigen::DenseBase<Derived>::Scalar κ_2 =
        typename Eigen::DenseBase<Derived>::Scalar(1e-2)) {
  using Scalar = typename Eigen::DenseBase<Derived>::Scalar;

  using std::abs;
  using std::isfinite;

  slp_assert(κ_1 > Scalar(0) && κ_2 > Scalar(0) && κ_2 < Scalar(0.5));

  Eigen::Index decision_variable_idx = 0;
  for (const auto& [lower, upper] : decision_variable_indices_to_bounds) {
    Scalar& x_i = x[decision_variable_idx++];

    // We assume that bound infeasibility is handled elsewhere.
    slp_assert(lower <= upper);

    // See section 3.6 of [2]
    if (isfinite(lower) && isfinite(upper)) {
      auto p_L = std::min(κ_1 * std::max(Scalar(1), abs(lower)),
                          κ_2 * (upper - lower));
      auto p_U = std::min(κ_1 * std::max(Scalar(1), abs(upper)),
                          κ_2 * (upper - lower));
      x_i = std::min(std::max(lower + p_L, x_i), upper - p_U);
    } else if (isfinite(lower)) {
      x_i = std::max(x_i, lower + κ_1 * std::max(Scalar(1), abs(lower)));
    } else if (isfinite(upper)) {
      x_i = std::min(x_i, upper - κ_1 * std::max(Scalar(1), abs(upper)));
    }
  }
}
}  // namespace slp
