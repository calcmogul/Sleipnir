// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>

#include <Eigen/Core>

#include "sleipnir/optimization/solver/util/manifold_type.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

/// Unit complex numbers (cos θ, sin θ), which represent SO(2), stored as 2 × 1
/// vectors.
///
/// The retraction is R_z(p) = z exp(ip), which rotates z by p radians.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class SO2Manifold final : public ManifoldType<Scalar> {
 public:
  using typename ManifoldType<Scalar>::DenseVector;
  using typename ManifoldType<Scalar>::DenseMatrix;
  using typename ManifoldType<Scalar>::Triplets;

  int ambient_size([[maybe_unused]] int rows,
                   [[maybe_unused]] int cols) const override {
    slp_assert(rows == 2 && cols == 1);
    return 2;
  }

  int tangent_size(int, int) const override { return 1; }

  /// Normalizes z. If z is zero, it's set to the identity rotation.
  ///
  /// @param[in] block The block being projected.
  /// @param[in,out] z The block's decision variables.
  void project([[maybe_unused]] const ManifoldBlock<Scalar>& block,
               Eigen::Ref<DenseVector> z) const override {
    Scalar norm = z.norm();
    if (norm > Scalar(0)) {
      z /= norm;
    } else {
      z = Eigen::Vector<Scalar, 2>{Scalar(1), Scalar(0)};
    }
  }

  void retract(const ManifoldBlock<Scalar>&, Eigen::Ref<const DenseVector> z,
               Eigen::Ref<const DenseVector> p,
               Eigen::Ref<DenseVector> result) const override {
    using std::cos;
    using std::sin;

    // R_z(p) = z exp(ip)
    Scalar c = cos(p[0]);
    Scalar s = sin(p[0]);
    result = Eigen::Vector<Scalar, 2>{c * z[0] - s * z[1], s * z[0] + c * z[1]};
    result.normalize();
  }

  DenseMatrix jacobian(const ManifoldBlock<Scalar>&,
                       Eigen::Ref<const DenseVector> z) const override {
    // J = iz
    return Eigen::Vector<Scalar, 2>{-z[1], z[0]};
  }

  void append_jacobian(const ManifoldBlock<Scalar>& block,
                       Eigen::Ref<const DenseVector> z, int tangent_offset,
                       Triplets& triplets) const override {
    // J = iz
    triplets.emplace_back(block.offset(), tangent_offset, -z[1]);
    triplets.emplace_back(block.offset() + 1, tangent_offset, z[0]);
  }

  void hessian_correction(const ManifoldBlock<Scalar>&,
                          Eigen::Ref<const DenseVector> z,
                          Eigen::Ref<const DenseVector> v, int tangent_offset,
                          Triplets& triplets) const override {
    // R_z(p) = z cos p + iz sin p = z + izp − zp²/2 + O(p³), so
    // ∇ₚₚ²Rᵢ = −zᵢ and Σᵢ vᵢ ∇ₚₚ²Rᵢ = −zᵀv
    triplets.emplace_back(tangent_offset, tangent_offset, -z.dot(v));
  }
};

/// The SO(2) manifold type.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
inline const SO2Manifold<Scalar> so2_manifold{};

}  // namespace slp
