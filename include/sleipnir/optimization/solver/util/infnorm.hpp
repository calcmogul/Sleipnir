// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/SparseCore>

namespace slp {

/// Returns the infinity norm of A.
///
/// @tparam Scalar Scalar type.
/// @return The infinity norm of A.
template <typename Scalar>
Scalar infnorm(const Eigen::SparseMatrix<Scalar>& A) {
  Scalar infnorm(0);
  for (int col = 0; col < A.cols(); ++col) {
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it{A, col}; it;
         ++it) {
      using std::abs;
      infnorm = std::max(infnorm, abs(it.value()));
    }
  }
  return infnorm;
}

}  // namespace slp
