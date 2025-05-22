// Copyright (c) Sleipnir contributors

#pragma once

#include <functional>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace slp {

/// Matrix callbacks for the interior-point method solver.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct InteriorPointMatrixCallbacks {
  /// Type alias for dense vector.
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  /// Type alias for sparse vector.
  using SparseVector = Eigen::SparseVector<Scalar>;

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
  std::function<Scalar(const DenseVector& x)> f;

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
  std::function<SparseVector(const DenseVector& x)> g;

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
  std::function<SparseMatrix(const DenseVector& x, const DenseVector& v,
                             Scalar sqrt_μ)>
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
  std::function<DenseVector(const DenseVector& x)> c_i;

  /// Inequality constraint Jacobian ∂cᵢ/∂x getter.
  ///
  /// ```
  ///         [∇ᵀcᵢ₁(xₖ)]
  /// Aᵢ(x) = [∇ᵀcᵢ₂(xₖ)]
  ///         [    ⋮    ]
  ///         [∇ᵀcᵢₘ(xₖ)]
  /// ```
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
  std::function<SparseMatrix(const DenseVector& x)> A_i;
};

}  // namespace slp
