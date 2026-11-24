// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>

#include <Eigen/Core>

#include "sleipnir/optimization/solver/util/manifold_type.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

/// The unit sphere Sⁿ⁻¹ in ℝⁿ, with points stored as n × 1 vectors.
///
/// The retraction is R_x(p) = (x + Bp)/‖x + Bp‖, where B is an orthonormal
/// basis for the tangent space at x.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class SphereManifold final : public ManifoldType<Scalar> {
 public:
  using typename ManifoldType<Scalar>::DenseVector;
  using typename ManifoldType<Scalar>::DenseMatrix;
  using typename ManifoldType<Scalar>::Triplets;

  int ambient_size(int rows, [[maybe_unused]] int cols) const override {
    slp_assert(rows >= 2 && cols == 1);
    return rows;
  }

  int tangent_size(int rows, [[maybe_unused]] int cols) const override {
    return rows - 1;
  }

  /// Normalizes x. If x is zero, it's set to e₁.
  ///
  /// @param[in] block The block being projected.
  /// @param[in,out] x The block's decision variables.
  void project([[maybe_unused]] const ManifoldBlock<Scalar>& block,
               Eigen::Ref<DenseVector> x) const override {
    Scalar norm = x.norm();
    if (norm > Scalar(0)) {
      x /= norm;
    } else {
      x.setZero();
      x[0] = Scalar(1);
    }
  }

  void retract(const ManifoldBlock<Scalar>&, Eigen::Ref<const DenseVector> x,
               Eigen::Ref<const DenseVector> p,
               Eigen::Ref<DenseVector> result) const override {
    // R_x(p) = (x + Bp)/‖x + Bp‖
    result = x + tangent_basis(x) * p;
    result.normalize();
  }

  DenseMatrix jacobian(const ManifoldBlock<Scalar>&,
                       Eigen::Ref<const DenseVector> x) const override {
    return tangent_basis(x);
  }

  void append_jacobian(const ManifoldBlock<Scalar>& block,
                       Eigen::Ref<const DenseVector> x, int tangent_offset,
                       Triplets& triplets) const override {
    using std::abs;

    // The entries of tangent_basis(x), in column-major order
    const int d = x.rows();
    const Scalar sign = x[0] >= Scalar(0) ? Scalar(1) : Scalar(-1);
    const Scalar denominator = Scalar(1) + abs(x[0]);
    auto v = [&](int i) { return i == 0 ? x[0] + sign : x[i]; };
    for (int col = 0; col < d - 1; ++col) {
      for (int row = 0; row < d; ++row) {
        Scalar value = -v(row) * v(col + 1) / denominator;
        if (row == col + 1) {
          value += Scalar(1);
        }
        triplets.emplace_back(block.offset() + row, tangent_offset + col,
                              value);
      }
    }
  }

  void hessian_correction(const ManifoldBlock<Scalar>& block,
                          Eigen::Ref<const DenseVector> x,
                          Eigen::Ref<const DenseVector> v, int tangent_offset,
                          Triplets& triplets) const override {
    // R_x(p) = (x + Bp)/√(1 + ‖p‖²) = x + Bp − x‖p‖²/2 + O(‖p‖³), so
    // ∇ₚₚ²Rᵢ = −xᵢI and Σᵢ vᵢ ∇ₚₚ²Rᵢ = −(xᵀv)I
    append_scaled_identity(-x.dot(v), tangent_offset, block.tangent_size(),
                           triplets);
  }

 private:
  /// Returns an orthonormal basis B for the tangent space of the unit sphere
  /// at x.
  ///
  /// The Householder reflection P = I − vvᵀ/(1 + |x₀|), where
  /// v = x + sign(x₀)e₀, is orthogonal and maps e₀ to −sign(x₀)x. Its other
  /// columns are orthonormal and orthogonal to x, so they're the basis.
  static DenseMatrix tangent_basis(Eigen::Ref<const DenseVector> x) {
    using std::abs;

    int d = x.rows();
    Scalar sign = x[0] >= Scalar(0) ? Scalar(1) : Scalar(-1);
    DenseVector v = x;
    v[0] += sign;

    DenseMatrix B =
        -v * v.segment(1, d - 1).transpose() / (Scalar(1) + abs(x[0]));
    B.bottomRows(d - 1).diagonal().array() += Scalar(1);
    return B;
  }
};

/// The unit sphere manifold type.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
inline const SphereManifold<Scalar> sphere_manifold{};

}  // namespace slp
