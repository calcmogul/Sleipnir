// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/SparseCore>

namespace slp {

/**
 * Returns the infinity norm of A.
 *
 * @return The infinity norm of A.
 */
inline double infnorm(const Eigen::SparseMatrix<double>& A) {
  double infnorm = 0.0;
  for (int col = 0; col < A.cols(); ++col) {
    for (Eigen::SparseMatrix<double>::InnerIterator it{A, col}; it; ++it) {
      infnorm = std::max(infnorm, std::abs(it.value()));
    }
  }
  return infnorm;
}

}  // namespace slp
