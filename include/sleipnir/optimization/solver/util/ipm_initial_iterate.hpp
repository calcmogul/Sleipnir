// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/ipm_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/util/barrier_parameter.hpp"
#include "sleipnir/optimization/solver/util/constraint_violation.hpp"
#include "sleipnir/optimization/solver/util/first_derivatives.hpp"
#include "sleipnir/optimization/solver/util/fraction_to_the_boundary_rule.hpp"
#include "sleipnir/optimization/solver/util/function_values.hpp"
#include "sleipnir/optimization/solver/util/iterates.hpp"
#include "sleipnir/optimization/solver/util/kkt_system.hpp"

namespace slp {

/// Returns the interior-point method's default initial barrier parameter.
///
/// @tparam Scalar Scalar type.
/// @param[in] matrices Matrix callbacks.
/// @return The default initial barrier parameter.
template <typename Scalar>
Scalar ipm_default_barrier_parameter(
    const IPMMatrixCallbacks<Scalar>& matrices) {
  return Scalar(0.1) * matrices.scaling.f;
}

/// Fills an interior-point method initial iterate's unset (NaN) and
/// out-of-bounds entries.
///
/// @tparam Scalar Scalar type.
/// @param[in] c_i The inequality constraints at the initial guess.
/// @param[in] μ_min The barrier parameter minimum.
/// @param[in,out] s The inequality constraint slack variables.
/// @param[in,out] y The equality constraint dual variables.
/// @param[in,out] z The inequality constraint dual variables.
template <typename Scalar>
void fill_ipm_initial_iterate(const Eigen::VectorX<Scalar>& c_i, Scalar μ_min,
                              Eigen::VectorX<Scalar>& s,
                              Eigen::VectorX<Scalar>& y,
                              Eigen::VectorX<Scalar>& z) {
  using std::isfinite;
  using std::isnan;

  auto is_interior = [](Scalar value) {
    return value > Scalar(0) && isfinite(value);
  };

  // If any zᵢ is set, the rest go on the central path for μ_min
  const bool any_z_set =
      std::ranges::any_of(z, [](Scalar value) { return !isnan(value); });

  for (int row = 0; row < s.rows(); ++row) {
    if (!is_interior(s[row])) {
      if (is_interior(z[row])) {
        // Put the slack-dual pair on the central path for μ_min
        s[row] = μ_min / z[row];
      } else if (c_i[row] >= Scalar(1e-2)) {
        // cᵢ(x) is sufficiently feasible, so start the constraint with zero
        // violation
        s[row] = c_i[row];
      } else {
        // Make Σ = S⁻¹Z ≈ I
        s[row] = Scalar(1);
      }
    }
    s[row] = std::max(s[row], c_i[row]);

    if (!is_interior(z[row])) {
      z[row] = any_z_set ? μ_min / s[row] : Scalar(1);
    }
  }

  for (int row = 0; row < y.rows(); ++row) {
    if (!isfinite(y[row])) {
      y[row] = Scalar(0);
    }
  }
}

/// An interior-point method iterate's slack variables, inequality constraint
/// dual variables, and barrier parameter.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct IPMCenteredIterate {
  /// The inequality constraint slack variables.
  Eigen::VectorX<Scalar> s;

  /// The inequality constraint dual variables.
  Eigen::VectorX<Scalar> z;

  /// The barrier parameter.
  Scalar μ;
};

/// Chooses the interior-point method's initial barrier parameter from the
/// initial iterate and centers the iterate's slack-dual pairs for it.
///
/// If no barrier parameter below the default one is valid, the iterate is
/// replaced with the default initial iterate.
///
/// @tparam Scalar Scalar type.
/// @param[in] matrices Matrix callbacks.
/// @param[in,out] kkt_system The KKT system, which is overwritten.
/// @param[in,out] iterates The initial iterate. Only s, y, and z are modified.
/// @param[in] function_values The cost and constraints at the iterate's
///     decision variables.
/// @param[in] first_derivatives The cost gradient and constraint Jacobians at
///     the iterate's decision variables.
/// @param[in,out] H The Lagrangian Hessian at the iterate.
/// @param[in] μ_min The barrier parameter minimum.
/// @param[in] τ The fraction-to-the-boundary rule scale factor.
/// @param[in] bound_constraint_mask Which inequality constraints are bounds.
/// @param[out] μ The barrier parameter.
template <typename Scalar>
void choose_ipm_initial_iterate(
    const IPMMatrixCallbacks<Scalar>& matrices,
    IPMKKTSystem<Scalar>& kkt_system, IPMIterates<Scalar>& iterates,
    const IPMFunctionValues<Scalar>& function_values,
    const IPMFirstDerivatives<Scalar>& first_derivatives,
    Eigen::SparseMatrix<Scalar>& H, Scalar μ_min, Scalar τ,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Scalar& μ) {
  using DenseVector = Eigen::VectorX<Scalar>;
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  using std::sqrt;

  auto& [x, s, y, z] = iterates;
  const auto& [f, c_e, c_i] = function_values;
  const auto& [g, A_e, A_i] = first_derivatives;

  const DenseVector stationarity_without_z = g - A_e.transpose() * y;

  // Returns the iterate centered for the smallest barrier parameter it can
  // start at, or std::nullopt if it can't start below the default initial
  // barrier parameter. include_equality_constraints is whether the equality
  // constraint violation counts toward the iterate's KKT error.
  auto center = [&](bool include_equality_constraints)
      -> std::optional<IPMCenteredIterate<Scalar>> {
    // Constraint violation before centering
    Scalar E_fixed = (c_i - s).template lpNorm<Eigen::Infinity>();
    if (include_equality_constraints) {
      E_fixed = std::max(E_fixed, c_e.template lpNorm<Eigen::Infinity>());
    }

    std::optional<IPMCenteredIterate<Scalar>> result;

    // Find the smallest μ' in the barrier parameter sequence whose centered
    // iterate satisfies the barrier subproblem tolerance for its predecessor
    Scalar pred_μ = ipm_default_barrier_parameter(matrices);
    while (pred_μ > μ_min) {
      IPMCenteredIterate<Scalar> trial{s, z,
                                       next_barrier_parameter(pred_μ, μ_min)};
      auto& [trial_s, trial_z, trial_μ] = trial;

      // Raise each slack-dual pair with sᵢzᵢ < μ' onto the central path
      // sᵢzᵢ = μ'
      const Scalar sqrt_trial_μ = sqrt(trial_μ);
      for (int row = 0; row < trial_s.rows(); ++row) {
        const Scalar sz = trial_s[row] * trial_z[row];
        if (sz >= trial_μ) {
          continue;
        }

        if (std::max(trial_s[row], trial_z[row]) >= sqrt_trial_μ) {
          // Raise the smaller of sᵢ and zᵢ
          if (trial_s[row] < trial_z[row]) {
            trial_s[row] = trial_μ / trial_z[row];
          } else {
            trial_z[row] = trial_μ / trial_s[row];
          }
        } else {
          // Raise both by d.
          //
          //   (sᵢ + d)(zᵢ + d) = μ'
          //   d² + (sᵢ + zᵢ)d + sᵢzᵢ − μ' = 0
          //   d = 2(μ' − sᵢzᵢ)/(sᵢ + zᵢ + √((sᵢ − zᵢ)² + 4μ'))
          const Scalar d = Scalar(2) * (trial_μ - sz) /
                           (trial_s[row] + trial_z[row] +
                            sqrt((trial_s[row] - trial_z[row]) *
                                     (trial_s[row] - trial_z[row]) +
                                 Scalar(4) * trial_μ));
          trial_s[row] += d;
          trial_z[row] += d;
        }
      }

      // E = max(‖∇f − Aₑᵀy − Aᵢᵀz‖_∞, ‖Sz − μe‖_∞, ‖cₑ‖_∞, ‖cᵢ − s‖_∞)
      const Scalar E =
          std::max({E_fixed,
                    (stationarity_without_z - A_i.transpose() * trial_z)
                        .template lpNorm<Eigen::Infinity>(),
                    (trial_s.cwiseProduct(trial_z).array() - pred_μ)
                        .matrix()
                        .template lpNorm<Eigen::Infinity>()});
      if (E <= barrier_subproblem_tolerance(pred_μ)) {
        result = trial;
      }

      pred_μ = trial_μ;
    }

    return result;
  };

  // Returns whether a Newton step from x, y, and the given iterate makes good
  // progress on the equality constraint violation. iterate_H is the Lagrangian
  // Hessian at x, y, and the iterate's z.
  auto step_reduces_equality_violation =
      [&](const SparseMatrix& iterate_H,
          const IPMCenteredIterate<Scalar>& iterate) {
        kkt_system.build(iterate_H, A_e, A_i, iterate.s, iterate.z);
        if (!kkt_system.factorize()) {
          return false;
        }
        const auto step = kkt_system.solve(g, A_e, A_i, y, iterate.s, iterate.z,
                                           iterate.μ, c_e, c_i - iterate.s);

        const Scalar α =
            fraction_to_the_boundary_rule<Scalar>(iterate.s, step.p_s, τ);
        const Scalar α_z =
            fraction_to_the_boundary_rule<Scalar>(iterate.z, step.p_z, τ);

        // Reject short steps
        if (std::min(α, α_z) < Scalar(0.75)) {
          return false;
        }

        const DenseVector trial_c_e = matrices.c_e(x + α * step.p_x);
        if (!trial_c_e.allFinite()) {
          return false;
        }

        // The Newton step satisfies Aₑpˣ = −cₑ, so the linearized equality
        // constraint violation is zero after a full step and (1 − α)‖cₑ(x)‖₁
        // after the fraction-to-the-boundary step α. Require the actual
        // violation to drop by at least half as much.
        //
        //   ‖cₑ(x + αpˣ)‖₁ ≤ (1 − α/2)‖cₑ(x)‖₁
        return constraint_violation(trial_c_e) <=
               (Scalar(1) - Scalar(0.5) * α) * constraint_violation(c_e);
      };

  // Ignore the equality constraint violation when choosing the barrier
  // parameter if a Newton step removes it
  std::optional<IPMCenteredIterate<Scalar>> iterate;
  std::optional<SparseMatrix> iterate_H;
  if (auto c_e_ignored = center(false)) {
    auto c_e_included = center(true);
    if (c_e_included && c_e_included->μ == c_e_ignored->μ) {
      iterate = std::move(c_e_included);
    } else {
      SparseMatrix c_e_ignored_H =
          c_e_ignored->z == z ? H : matrices.H(x, y, c_e_ignored->z);
      if (step_reduces_equality_violation(c_e_ignored_H, *c_e_ignored)) {
        iterate = std::move(c_e_ignored);
        iterate_H = std::move(c_e_ignored_H);
      } else {
        iterate = std::move(c_e_included);
      }
    }
  }

  if (iterate) {
    if (iterate->z != z) {
      H = iterate_H ? std::move(*iterate_H) : matrices.H(x, y, iterate->z);
    }
    s = std::move(iterate->s);
    z = std::move(iterate->z);
    μ = iterate->μ;
  } else {
    // Start from the default initial iterate
    s.setConstant(std::numeric_limits<Scalar>::quiet_NaN());
    DenseVector y_0 = DenseVector::Constant(
        y.rows(), std::numeric_limits<Scalar>::quiet_NaN());
    DenseVector z_0 = DenseVector::Constant(
        z.rows(), std::numeric_limits<Scalar>::quiet_NaN());
    fill_ipm_initial_iterate(c_i, μ_min, s, y_0, z_0);
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    s = bound_constraint_mask.select(c_i, s);
#endif
    if (y_0 != y || z_0 != z) {
      y = std::move(y_0);
      z = std::move(z_0);
      H = matrices.H(x, y, z);
    }
    μ = ipm_default_barrier_parameter(matrices);
  }
}

}  // namespace slp
