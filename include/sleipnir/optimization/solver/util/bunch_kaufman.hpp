// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "sleipnir/optimization/solver/util/inertia.hpp"

namespace slp {

/// Wrapper around Eigen's Bunch-Kaufman decomposition with the inertia exposed.
///
/// @tparam MatrixType Matrix type.
/// @tparam UpLo The triangular part that will be used for the decomposition:
///     Lower (default) or Upper. The other triangular part won't be read.
template <typename MatrixType, int UpLo = Eigen::Lower>
class BunchKaufman : public Eigen::BunchKaufman<MatrixType, UpLo> {
 public:
  using Eigen::BunchKaufman<MatrixType, UpLo>::BunchKaufman;

  /// Returns the matrix inertia.
  ///
  /// @return The matrix inertia.
  Inertia inertia() const {
    using Base = Eigen::BunchKaufman<MatrixType, UpLo>;
    return {static_cast<int>(Base::m_n_pos), static_cast<int>(Base::m_n_neg),
            static_cast<int>(Base::m_n_zero)};
  }
};

}  // namespace slp
