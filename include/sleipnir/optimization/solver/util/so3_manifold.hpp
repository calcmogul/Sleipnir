// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <limits>

#include <Eigen/Core>

#include "sleipnir/optimization/solver/util/manifold_type.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

/// Unit quaternions (w, x, y, z), which represent SO(3), stored as 4 × 1
/// vectors.
///
/// The retraction is R_q(p) = q ⊗ exp(p/2), which rotates q by the rotation
/// vector p in the body frame.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class SO3Manifold final : public ManifoldType<Scalar> {
 public:
  using typename ManifoldType<Scalar>::DenseVector;
  using typename ManifoldType<Scalar>::DenseMatrix;
  using typename ManifoldType<Scalar>::Triplets;

  int ambient_size([[maybe_unused]] int rows,
                   [[maybe_unused]] int cols) const override {
    slp_assert(rows == 4 && cols == 1);
    return 4;
  }

  int tangent_size(int, int) const override { return 3; }

  /// Normalizes q. If q is zero, it's set to the identity rotation.
  ///
  /// @param[in] block The block being projected.
  /// @param[in,out] q The block's decision variables.
  void project([[maybe_unused]] const ManifoldBlock<Scalar>& block,
               Eigen::Ref<DenseVector> q) const override {
    Scalar norm = q.norm();
    if (norm > Scalar(0)) {
      q /= norm;
    } else {
      q = Eigen::Vector<Scalar, 4>{Scalar(1), Scalar(0), Scalar(0), Scalar(0)};
    }
  }

  void retract(const ManifoldBlock<Scalar>&, Eigen::Ref<const DenseVector> q,
               Eigen::Ref<const DenseVector> p,
               Eigen::Ref<DenseVector> result) const override {
    using std::cos;
    using std::sin;
    using std::sqrt;

    // R_q(p) = q ⊗ exp(p/2)
    //
    // exp(φ) = [cos‖φ‖, sin‖φ‖ φ/‖φ‖]
    Eigen::Vector<Scalar, 3> φ = p / Scalar(2);
    Scalar θ = φ.norm();

    // sin(θ)/θ = 1 − θ²/6 + θ⁴/120 − …, so the first two terms are exact to
    // machine precision when θ⁴/120 < ε
    Scalar sinc;
    if (θ < sqrt(sqrt(std::numeric_limits<Scalar>::epsilon()))) {
      sinc = Scalar(1) - θ * θ / Scalar(6);
    } else {
      sinc = sin(θ) / θ;
    }

    Eigen::Vector<Scalar, 4> r{cos(θ), sinc * φ[0], sinc * φ[1], sinc * φ[2]};
    result = left_product_matrix(q) * r;
    result.normalize();
  }

  DenseMatrix jacobian(const ManifoldBlock<Scalar>&,
                       Eigen::Ref<const DenseVector> q) const override {
    // J = ½ L(q)[:, 1:4]
    return left_product_matrix(q).template rightCols<3>() / Scalar(2);
  }

  void hessian_correction(const ManifoldBlock<Scalar>& block,
                          Eigen::Ref<const DenseVector> q,
                          Eigen::Ref<const DenseVector> v, int tangent_offset,
                          Triplets& triplets) const override {
    // R_q(p) = q ⊗ exp(p/2) = q ⊗ [1 − ‖p‖²/8, p/2] + O(‖p‖³)
    //        = q + Jp − q‖p‖²/8 + O(‖p‖³),
    //
    // so ∇ₚₚ²Rᵢ = −¼qᵢI and Σᵢ vᵢ ∇ₚₚ²Rᵢ = −¼(qᵀv)I
    append_scaled_identity(-q.dot(v) / Scalar(4), tangent_offset,
                           block.tangent_size(), triplets);
  }

 private:
  /// Returns the matrix L(q) such that q ⊗ r = L(q)r for quaternions in
  /// (w, x, y, z) order.
  static Eigen::Matrix<Scalar, 4, 4> left_product_matrix(
      Eigen::Ref<const DenseVector> q) {
    Scalar w = q[0];
    Scalar x = q[1];
    Scalar y = q[2];
    Scalar z = q[3];

    return Eigen::Matrix<Scalar, 4, 4>{
        {w, -x, -y, -z}, {x, w, -z, y}, {y, z, w, -x}, {z, -y, x, w}};
  }
};

/// The SO(3) manifold type.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
inline const SO3Manifold<Scalar> so3_manifold{};

}  // namespace slp
