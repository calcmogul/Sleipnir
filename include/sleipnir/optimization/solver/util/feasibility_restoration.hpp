// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <span>

#include <Eigen/Core>
#include <gch/small_vector.hpp>

#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/interior_point.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/sqp_matrix_callbacks.hpp"
#include "sleipnir/util/function_ref.hpp"

namespace slp {

/// Finds the iterate that minimizes the constraint violation while not
/// deviating too far from the starting point. This is a fallback procedure when
/// the normal Sequential Quadratic Programming method fails to converge to a
/// feasible point.
///
/// @tparam Scalar Scalar type.
/// @param[in] matrix_callbacks Matrix callbacks.
/// @param[in] callback The user callback.
/// @param[in] options Solver options.
/// @param[in,out] x The current iterate from the normal solve.
/// @return The exit status.
template <typename Scalar>
ExitStatus feasibility_restoration(
    const SQPMatrixCallbacks<Scalar>& matrix_callbacks,
    function_ref<bool(const IterationInfo<Scalar>& info)> callback,
    const Options& options, Eigen::Vector<Scalar, Eigen::Dynamic>& x) {
  // Feasibility restoration
  //
  //        min  ρ Σ (pₑ + nₑ) + ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)
  //         x
  //       pₑ,nₑ
  //
  //   s.t. cₑ(x) - pₑ + nₑ = 0
  //        pₑ ≥ 0
  //        nₑ ≥ 0
  //
  // where ρ = 1000, ζ = √μ where μ is the barrier parameter, xᵣ is original
  // iterate before feasibility restoration, and Dᵣ is a scaling matrix defined
  // by
  //
  //   Dᵣ = diag(min(1, 1/|xᵣ⁽¹⁾|), …, min(1, 1/|xᵣ|⁽ⁿ⁾))
  //
  // Let x' = [x  pₑ  nₑ]ᵀ, cₑ' refer to the equality constraints, and cᵢ' refer
  // to the inequality constraints, Aₑ' = ∂cₑ'/∂x', and Aᵢ' = ∂cᵢ'/∂x'.
  //
  //   f(x') = ρ Σ (pₑ + nₑ) + ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)
  //
  //            [ζDᵣ(x − xᵣ)]
  //   ∂f/∂x' = [     ρ     ]
  //            [     ρ     ]
  //
  //              [ζDᵣ  0  0]
  //   ∂²f/∂x'² = [ 0   0  0]
  //              [ 0   0  0]
  //
  //   Aₑ' = [Aₑ  −I  I]
  //
  //   Aᵢ' = [0  I  0]
  //         [0  0  I]

  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  using std::abs;
  using std::sqrt;

  constexpr Scalar ρ(1000.0);
  Scalar μ(options.tolerance / 10.0);

  gch::small_vector<Variable<Scalar>> fr_decision_variables;
  fr_decision_variables.reserve(decision_variables.size() +
                                2 * equality_constraints.size());

  // Assign x
  fr_decision_variables.assign(decision_variables.begin(),
                               decision_variables.end());

  // Allocate pₑ and nₑ
  for (size_t row = 0; row < 2 * equality_constraints.size(); ++row) {
    fr_decision_variables.emplace_back();
  }

  auto it = fr_decision_variables.cbegin();

  VariableMatrix xAD{std::span{it, it + decision_variables.size()}};
  it += decision_variables.size();

  VariableMatrix p_e{std::span{it, it + equality_constraints.size()}};
  it += equality_constraints.size();

  VariableMatrix n_e{std::span{it, it + equality_constraints.size()}};
  it += equality_constraints.size();

  // Set initial values for pₑ and nₑ.
  //
  //
  // From equation (33) of [2]:
  //                       ______________________
  //       μ − ρ c(x)     /(μ − ρ c(x))²   μ c(x)
  //   n = −−−−−−−−−− +  / (−−−−−−−−−−)  + −−−−−−     (1)
  //           2ρ       √  (    2ρ    )      2ρ
  //
  // The quadratic formula:
  //             ________
  //       -b + √b² - 4ac
  //   x = −−−−−−−−−−−−−−                             (2)
  //             2a
  //
  // Rearrange (1) to fit the quadratic formula better:
  //                     _________________________
  //       μ - ρ c(x) + √(μ - ρ c(x))² + 2ρ μ c(x)
  //   n = −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
  //                         2ρ
  //
  // Solve for coefficients:
  //
  //   a = ρ                                          (3)
  //   b = ρ c(x) - μ                                 (4)
  //
  //   -4ac = μ c(x) 2ρ
  //   -4(ρ)c = 2ρ μ c(x)
  //   -4c = 2μ c(x)
  //   c = -μ c(x)/2                                  (5)
  //
  //   p = c(x) + n                                   (6)
  for (int row = 0; row < p_e.rows(); ++row) {
    Scalar c_e = equality_constraints[row].value();

    constexpr Scalar a = 2 * ρ;
    Scalar b = ρ * c_e - μ;
    Scalar c = -μ * c_e / Scalar(2);

    Scalar n = -b * sqrt(b * b - Scalar(4) * a * c) / (Scalar(2) * a);
    Scalar p = c_e + n;

    p_e[row].set_value(p);
    n_e[row].set_value(n);
  }

  // cₑ(x) - pₑ + nₑ = 0
  gch::small_vector<Variable<Scalar>> fr_equality_constraints;
  fr_equality_constraints.assign(equality_constraints.begin(),
                                 equality_constraints.end());
  for (size_t row = 0; row < fr_equality_constraints.size(); ++row) {
    auto& constraint = fr_equality_constraints[row];
    constraint = constraint - p_e[row] + n_e[row];
  }

  // cᵢ(x) - s - pᵢ + nᵢ = 0
  gch::small_vector<Variable<Scalar>> fr_inequality_constraints;

  // pₑ ≥ 0
  // nₑ ≥ 0
  std::ranges::copy(p_e, std::back_inserter(fr_inequality_constraints));
  std::ranges::copy(n_e, std::back_inserter(fr_inequality_constraints));

  Variable J = Scalar(0);

  // J += ρ Σ (pₑ + nₑ)
  for (auto& elem : p_e) {
    J += elem;
  }
  for (auto& elem : n_e) {
    J += elem;
  }
  J *= ρ;

  // Dᵣ = diag(min(1, 1/|xᵣ⁽¹⁾|), …, min(1, 1/|xᵣ|⁽ⁿ⁾)
  DenseVector D_r{x.rows()};
  for (int row = 0; row < D_r.rows(); ++row) {
    D_r[row] = std::min(Scalar(1), Scalar(1) / abs(x[row]));
  }

  // J += ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)
  for (int row = 0; row < x.rows(); ++row) {
    J +=
        sqrt(μ) / Scalar(2) * D_r[row] * slp::pow(xAD[row] - x[row], Scalar(2));
  }

  DenseVector fr_x = VariableMatrix{fr_decision_variables}.value();

  // Set up initial value for inequality constraint slack variables
  DenseVector fr_s{fr_inequality_constraints.size()};
  fr_s.setOnes();

  gch::small_vector<std::function<bool(const IterationInfo<Scalar>& info)>>
      callbacks{callback};
  auto status = InteriorPoint(fr_decision_variables, fr_equality_constraints,
                              fr_inequality_constraints, J, callbacks, options,
                              true, fr_x, fr_s);

  x = fr_x.segment(0, decision_variables.size());

  return status;
}

/// Finds the iterate that minimizes the constraint violation while not
/// deviating too far from the starting point. This is a fallback procedure when
/// the normal interior-point method fails to converge to a feasible point.
///
/// @param[in] matrix_callbacks Matrix callbacks.
/// @param[in] μ Barrier parameter.
/// @param[in] callback The user callback.
/// @param[in] options Solver options.
/// @param[in,out] x The current iterate from the normal solve.
/// @param[in,out] s The current inequality constraint slack variables from the
///     normal solve.
/// @return The exit status.
template <typename Scalar>
ExitStatus feasibility_restoration(
    const InteriorPointMatrixCallbacks<Scalar>& matrix_callbacks, Scalar μ,
    function_ref<bool(const IterationInfo<Scalar>& info)> callback,
    const Options& options, Eigen::Vector<Scalar, Eigen::Dynamic>& x,
    Eigen::Vector<Scalar, Eigen::Dynamic>& s) {
  // Feasibility restoration
  //
  //        min  ρ Σ (pₑ + nₑ + pᵢ + nᵢ) + ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)
  //         x
  //       pₑ,nₑ
  //       pᵢ,nᵢ
  //
  //   s.t. cₑ(x) - pₑ + nₑ = 0
  //        cᵢ(x) - s - pᵢ + nᵢ = 0
  //        pₑ ≥ 0
  //        nₑ ≥ 0
  //        pᵢ ≥ 0
  //        nᵢ ≥ 0
  //
  // where ρ = 1000, ζ = √μ where μ is the barrier parameter, xᵣ is original
  // iterate before feasibility restoration, and Dᵣ is a scaling matrix defined
  // by
  //
  //   Dᵣ = diag(min(1, 1/|xᵣ⁽¹⁾|), …, min(1, 1/|xᵣ|⁽ⁿ⁾)
  //
  // Let x' = [x  pₑ  nₑ  pᵢ  nᵢ]ᵀ, cₑ' refer to the equality constraints, and
  // cᵢ' refer to the inequality constraints, Aₑ' = ∂cₑ'/∂x', and
  // Aᵢ' = ∂cᵢ'/∂x'.
  //
  // f(x') = ρ Σ (pₑ + nₑ + pᵢ + nᵢ) + ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)
  //
  //            [ζDᵣ(x − xᵣ)]
  //            [     ρ     ]
  //   ∂f/∂x' = [     ρ     ]
  //            [     ρ     ]
  //            [     ρ     ]
  //
  //              [ζDᵣ  0  0  0  0]
  //              [ 0   0  0  0  0]
  //   ∂²f/∂x'² = [ 0   0  0  0  0]
  //              [ 0   0  0  0  0]
  //              [ 0   0  0  0  0]
  //
  //   Aₑ' = [Aₑ  −I  I   0  0]
  //         [Aᵢ   0  0  −I  I]
  //
  //         [0  I  0  0  0]
  //   Aᵢ' = [0  0  I  0  0]
  //         [0  0  0  I  0]
  //         [0  0  0  0  I]

  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  using std::abs;
  using std::sqrt;

  constexpr Scalar ρ(1000.0);

  gch::small_vector<Variable<Scalar>> fr_decision_variables;
  fr_decision_variables.reserve(decision_variables.size() +
                                2 * equality_constraints.size() +
                                2 * inequality_constraints.size());

  // Assign x
  fr_decision_variables.assign(decision_variables.begin(),
                               decision_variables.end());

  // Allocate pₑ, nₑ, pᵢ, and nᵢ
  for (size_t row = 0; row < 2 * equality_constraints.size() +
                                 2 * inequality_constraints.size();
       ++row) {
    fr_decision_variables.emplace_back();
  }

  auto it = fr_decision_variables.cbegin();

  VariableMatrix xAD{std::span{it, it + decision_variables.size()}};
  it += decision_variables.size();

  VariableMatrix p_e{std::span{it, it + equality_constraints.size()}};
  it += equality_constraints.size();

  VariableMatrix n_e{std::span{it, it + equality_constraints.size()}};
  it += equality_constraints.size();

  VariableMatrix p_i{std::span{it, it + inequality_constraints.size()}};
  it += inequality_constraints.size();

  VariableMatrix n_i{std::span{it, it + inequality_constraints.size()}};

  // Set initial values for pₑ, nₑ, pᵢ, and nᵢ.
  //
  //
  // From equation (33) of [2]:
  //                       ______________________
  //       μ − ρ c(x)     /(μ − ρ c(x))²   μ c(x)
  //   n = −−−−−−−−−− +  / (−−−−−−−−−−)  + −−−−−−     (1)
  //           2ρ       √  (    2ρ    )      2ρ
  //
  // The quadratic formula:
  //             ________
  //       -b + √b² - 4ac
  //   x = −−−−−−−−−−−−−−                             (2)
  //             2a
  //
  // Rearrange (1) to fit the quadratic formula better:
  //                     _________________________
  //       μ - ρ c(x) + √(μ - ρ c(x))² + 2ρ μ c(x)
  //   n = −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
  //                         2ρ
  //
  // Solve for coefficients:
  //
  //   a = ρ                                          (3)
  //   b = ρ c(x) - μ                                 (4)
  //
  //   -4ac = μ c(x) 2ρ
  //   -4(ρ)c = 2ρ μ c(x)
  //   -4c = 2μ c(x)
  //   c = -μ c(x)/2                                  (5)
  //
  //   p = c(x) + n                                   (6)
  for (int row = 0; row < p_e.rows(); ++row) {
    Scalar c_e = equality_constraints[row].value();

    constexpr Scalar a = Scalar(2) * ρ;
    Scalar b = ρ * c_e - μ;
    Scalar c = -μ * c_e / Scalar(2);

    Scalar n = -b * sqrt(b * b - Scalar(4) * a * c) / (Scalar(2) * a);
    Scalar p = c_e + n;

    p_e[row].set_value(p);
    n_e[row].set_value(n);
  }
  for (int row = 0; row < p_i.rows(); ++row) {
    Scalar c_i = inequality_constraints[row].value() - s[row];

    constexpr Scalar a = Scalar(2) * ρ;
    Scalar b = ρ * c_i - μ;
    Scalar c = -μ * c_i / Scalar(2);

    Scalar n = -b * sqrt(b * b - Scalar(4) * a * c) / (Scalar(2) * a);
    Scalar p = c_i + n;

    p_i[row].set_value(p);
    n_i[row].set_value(n);
  }

  // cₑ(x) - pₑ + nₑ = 0
  gch::small_vector<Variable<Scalar>> fr_equality_constraints;
  fr_equality_constraints.assign(equality_constraints.begin(),
                                 equality_constraints.end());
  for (size_t row = 0; row < fr_equality_constraints.size(); ++row) {
    auto& constraint = fr_equality_constraints[row];
    constraint = constraint - p_e[row] + n_e[row];
  }

  // cᵢ(x) - s - pᵢ + nᵢ = 0
  gch::small_vector<Variable<Scalar>> fr_inequality_constraints;
  fr_inequality_constraints.assign(inequality_constraints.begin(),
                                   inequality_constraints.end());
  for (size_t row = 0; row < fr_inequality_constraints.size(); ++row) {
    auto& constraint = fr_inequality_constraints[row];
    constraint = constraint - s[row] - p_i[row] + n_i[row];
  }

  // pₑ ≥ 0
  // pᵢ ≥ 0
  // nₑ ≥ 0
  // nᵢ ≥ 0
  std::ranges::copy(p_e, std::back_inserter(fr_inequality_constraints));
  std::ranges::copy(p_i, std::back_inserter(fr_inequality_constraints));
  std::ranges::copy(n_e, std::back_inserter(fr_inequality_constraints));
  std::ranges::copy(n_i, std::back_inserter(fr_inequality_constraints));

  Variable J = Scalar(0);

  // J += ρ Σ (pₑ + nₑ + pᵢ + nᵢ)
  for (auto& elem : p_e) {
    J += elem;
  }
  for (auto& elem : p_i) {
    J += elem;
  }
  for (auto& elem : n_e) {
    J += elem;
  }
  for (auto& elem : n_i) {
    J += elem;
  }
  J *= ρ;

  // Dᵣ = diag(min(1, 1/|xᵣ⁽¹⁾|), …, min(1, 1/|xᵣ|⁽ⁿ⁾))
  DenseVector D_r{x.rows()};
  for (int row = 0; row < D_r.rows(); ++row) {
    D_r[row] = std::min(Scalar(1), Scalar(1) / abs(x[row]));
  }

  // J += ζ/2 (x - xᵣ)ᵀDᵣ(x - xᵣ)
  for (int row = 0; row < x.rows(); ++row) {
    J +=
        sqrt(μ) / Scalar(2) * D_r[row] * slp::pow(xAD[row] - x[row], Scalar(2));
  }

  DenseVector fr_x = VariableMatrix{fr_decision_variables}.value();

  // Set up initial value for inequality constraint slack variables
  DenseVector fr_s{fr_inequality_constraints.size()};
  fr_s.segment(0, inequality_constraints.size()) = s;
  fr_s.segment(inequality_constraints.size(),
               fr_s.size() - inequality_constraints.size())
      .setOnes();

  gch::small_vector<std::function<bool(const IterationInfo<Scalar>& info)>>
      callbacks{callback};
  auto status = interior_point<Scalar>(
      fr_decision_variables, fr_equality_constraints, fr_inequality_constraints,
      J, callbacks, options, true, fr_x, fr_s);

  x = fr_x.segment(0, decision_variables.size());
  s = fr_s.segment(0, inequality_constraints.size());

  return status;
}

}  // namespace slp
