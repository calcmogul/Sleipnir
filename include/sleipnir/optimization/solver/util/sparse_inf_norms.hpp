// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace slp {

/// Computes the row-wise infinity norm of a sparse matrix.
///
/// @param mat Sparse matrix.
/// @return A dense vector where each element is the infinity norm of the
///         corresponding row of the sparse matrix.
template <typename Derived>
auto sparse_inf_norms(const Eigen::SparseCompressedBase<Derived>& mat)
    -> Eigen::VectorX<typename Derived::Scalar> {
  using Scalar = typename Derived::Scalar;
  using DenseVector = Eigen::VectorX<Scalar>;

  DenseVector norms = DenseVector::Zero(mat.rows());
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (typename Derived::InnerIterator it{mat, k}; it; ++it) {
      using std::abs;
      norms[it.row()] = std::max(norms[it.row()], abs(it.value()));
    }
  }

  return norms;
}

}  // namespace slp
