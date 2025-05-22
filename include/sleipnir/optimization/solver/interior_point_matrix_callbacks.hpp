// Copyright (c) Sleipnir contributors

#pragma once

#include <functional>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace slp {

/**
 * Matrix callbacks for the interior-point method solver.
 *
 * @tparam Scalar Scalar type.
 */
template <typename Scalar>
struct InteriorPointMatrixCallbacks {
  /// Cost function value f(x) getter.
  ///
  /// <table>
  ///   <tr>
  ///     <th>Variable</th>
  ///     <th>Rows</th>
  ///     <th>Columns</th>
  ///   </tr>
  ///   <tr>
  ///     <td>x</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>f(x)</td>
  ///     <td>1</td>
  ///     <td>1</td>
  ///   </tr>
  /// </table>
  std::function<Scalar(const Eigen::Vector<Scalar, Eigen::Dynamic>& x)> f;

  /// Cost function gradient ∇f(x) getter.
  ///
  /// <table>
  ///   <tr>
  ///     <th>Variable</th>
  ///     <th>Rows</th>
  ///     <th>Columns</th>
  ///   </tr>
  ///   <tr>
  ///     <td>x</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>∇f(x)</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  /// </table>
  std::function<Eigen::SparseVector<Scalar>(
      const Eigen::Vector<Scalar, Eigen::Dynamic>& x)>
      g;

  /// Lagrangian Hessian ∇ₓₓ²L(x, v, √(μ)) getter.
  ///
  /// L(xₖ, vₖ, √(μ)) = f(xₖ) − √(μ)eᵛᵀcᵢ(xₖ)
  ///
  /// <table>
  ///   <tr>
  ///     <th>Variable</th>
  ///     <th>Rows</th>
  ///     <th>Columns</th>
  ///   </tr>
  ///   <tr>
  ///     <td>x</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>v</td>
  ///     <td>num_inequality_constraints</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>∇ₓₓ²L(x, v, √(μ))</td>
  ///     <td>num_decision_variables</td>
  ///     <td>num_decision_variables</td>
  ///   </tr>
  /// </table>
  std::function<Eigen::SparseMatrix<Scalar>(
      const Eigen::Vector<Scalar, Eigen::Dynamic>& x,
      const Eigen::Vector<Scalar, Eigen::Dynamic>& v, Scalar sqrt_μ)>
      H;

  /// Inequality constraint value cᵢ(x) getter.
  ///
  /// <table>
  ///   <tr>
  ///     <th>Variable</th>
  ///     <th>Rows</th>
  ///     <th>Columns</th>
  ///   </tr>
  ///   <tr>
  ///     <td>x</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>cᵢ(x)</td>
  ///     <td>num_inequality_constraints</td>
  ///     <td>1</td>
  ///   </tr>
  /// </table>
  std::function<Eigen::Vector<Scalar, Eigen::Dynamic>(
      const Eigen::Vector<Scalar, Eigen::Dynamic>& x)>
      c_i;

  /// Inequality constraint Jacobian ∂cᵢ/∂x getter.
  ///
  /// @verbatim
  ///         [∇ᵀcᵢ₁(xₖ)]
  /// Aᵢ(x) = [∇ᵀcᵢ₂(xₖ)]
  ///         [    ⋮    ]
  ///         [∇ᵀcᵢₘ(xₖ)]
  /// @endverbatim
  ///
  /// <table>
  ///   <tr>
  ///     <th>Variable</th>
  ///     <th>Rows</th>
  ///     <th>Columns</th>
  ///   </tr>
  ///   <tr>
  ///     <td>x</td>
  ///     <td>num_decision_variables</td>
  ///     <td>1</td>
  ///   </tr>
  ///   <tr>
  ///     <td>Aᵢ(x)</td>
  ///     <td>num_inequality_constraints</td>
  ///     <td>num_decision_variables</td>
  ///   </tr>
  /// </table>
  std::function<Eigen::SparseMatrix<Scalar>(
      const Eigen::Vector<Scalar, Eigen::Dynamic>& x)>
      A_i;
};

}  // namespace slp
