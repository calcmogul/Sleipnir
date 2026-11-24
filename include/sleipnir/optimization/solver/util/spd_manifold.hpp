// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/util/manifold_type.hpp"
#include "sleipnir/util/assert.hpp"

namespace slp {

/// Symmetric positive definite n × n matrices, stored as the lower triangle in
/// row-major order.
///
/// The SPD matrices are an open subset of the symmetric matrices, so the
/// tangent coordinates are the lower triangle's entries. The retraction is
/// R_P(ξ) = L exp(L⁻¹ξL⁻ᵀ) Lᵀ, where P = LLᵀ, and the boundary barrier is
/// b(P) = −log det P.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class SPDManifold final : public ManifoldType<Scalar> {
 public:
  using typename ManifoldType<Scalar>::DenseVector;
  using typename ManifoldType<Scalar>::DenseMatrix;
  using typename ManifoldType<Scalar>::Triplets;

  int ambient_size(int rows, [[maybe_unused]] int cols) const override {
    slp_assert(rows >= 1 && rows == cols);
    return rows * (rows + 1) / 2;
  }

  int tangent_size(int rows, int cols) const override {
    return ambient_size(rows, cols);
  }

  /// Leaves P unchanged if each eigenvalue is at least nε times the largest,
  /// where n is the matrix's dimension. Smaller eigenvalues are numerically
  /// zero, so they're raised to √ε times the largest, which keeps the
  /// condition number of their contribution at most 1/√ε. If P has no
  /// positive eigenvalue, it's set to the identity.
  ///
  /// @param[in] block The block being projected.
  /// @param[in,out] x The block's decision variables.
  void project(const ManifoldBlock<Scalar>& block,
               Eigen::Ref<DenseVector> x) const override {
    using std::sqrt;

    constexpr Scalar ε = std::numeric_limits<Scalar>::epsilon();

    const int n = block.rows();
    DenseMatrix P = unpack_symmetric(n, x);
    Eigen::SelfAdjointEigenSolver<DenseMatrix> eigensolver{P};
    DenseVector λ = eigensolver.eigenvalues();

    Scalar λ_max = λ.maxCoeff();
    if (!(λ_max > Scalar(0))) {
      P.setIdentity();
    } else {
      // The eigensolver's error is about nελ_max, so smaller eigenvalues are
      // indistinguishable from zero
      Scalar λ_zero = Scalar(n) * ε * λ_max;
      if (λ.minCoeff() >= λ_zero) {
        return;
      }

      λ = (λ.array() < λ_zero).select(sqrt(ε) * λ_max, λ);
      const auto& V = eigensolver.eigenvectors();
      P = V * λ.asDiagonal() * V.transpose();
    }
    pack_lower(P, x);
  }

  void retract(const ManifoldBlock<Scalar>& block,
               Eigen::Ref<const DenseVector> x, Eigen::Ref<const DenseVector> p,
               Eigen::Ref<DenseVector> result) const override {
    // R_P(ξ) = L exp(L⁻¹ξL⁻ᵀ) Lᵀ where P = LLᵀ
    //
    // exp() of a symmetric matrix is positive definite, so R_P(ξ) is too.
    // Since exp(M) = I + M + M²/2 + O(‖M‖³),
    //
    //   R_P(ξ) = P + ξ + ½ξP⁻¹ξ + O(‖ξ‖³)
    const int n = block.rows();
    Eigen::LLT<DenseMatrix> llt{unpack_symmetric(n, x)};
    DenseMatrix L = llt.matrixL();

    // M = L⁻¹ξL⁻ᵀ
    DenseMatrix M = llt.matrixL().solve(unpack_symmetric(n, p));
    M = llt.matrixL().solve(M.transpose().eval());

    Eigen::SelfAdjointEigenSolver<DenseMatrix> eigensolver{(M + M.transpose()) /
                                                           Scalar(2)};

    // L exp(M) Lᵀ = (LV exp(Λ/2))(LV exp(Λ/2))ᵀ where M = VΛVᵀ
    DenseMatrix W = L * eigensolver.eigenvectors() *
                    (eigensolver.eigenvalues() / Scalar(2))
                        .array()
                        .exp()
                        .matrix()
                        .asDiagonal();
    pack_lower(W * W.transpose(), result);
  }

  DenseMatrix jacobian(const ManifoldBlock<Scalar>& block,
                       Eigen::Ref<const DenseVector>) const override {
    // J = I since R_P(ξ) = P + ξ + O(‖ξ‖²)
    return DenseMatrix::Identity(block.size(), block.size());
  }

  void append_jacobian(const ManifoldBlock<Scalar>& block,
                       Eigen::Ref<const DenseVector>, int tangent_offset,
                       Triplets& triplets) const override {
    for (int i = 0; i < block.size(); ++i) {
      triplets.emplace_back(block.offset() + i, tangent_offset + i, Scalar(1));
    }
  }

  /// Appends the dense lower triangle of the Hessian correction.
  ///
  /// With G the symmetric matrix such that vᵀx = tr(GX), and R_P(ξ) =
  /// P + ξ + ½ξP⁻¹ξ + O(‖ξ‖³), the second-order term of vᵀR_P(ξ) is
  /// ½tr(GξP⁻¹ξ). For tangent coordinates a and b with basis matrices Eₐ and
  /// E_b, its Hessian is
  ///
  /// ```
  ///   Cₐ_b = ½(tr(GEₐP⁻¹E_b) + tr(GE_bP⁻¹Eₐ))
  /// ```
  ///
  /// Coordinate (i, j) has Eₐ = eᵢeᵢᵀ if i = j and eᵢeⱼᵀ + eⱼeᵢᵀ otherwise,
  /// and tr(G eᵣeₛᵀ P⁻¹ eₜeᵤᵀ) = Gᵤᵣ(P⁻¹)ₛₜ.
  ///
  /// @param[in] block The block.
  /// @param[in] x The point on the manifold.
  /// @param[in] v The ambient gradient.
  /// @param[in] tangent_offset Index of the block's first tangent coordinate.
  /// @param[in,out] triplets The triplets to append to.
  void hessian_correction(const ManifoldBlock<Scalar>& block,
                          Eigen::Ref<const DenseVector> x,
                          Eigen::Ref<const DenseVector> v, int tangent_offset,
                          Triplets& triplets) const override {
    const int m = block.size();
    const int n = block.rows();

    DenseMatrix P = unpack_symmetric(n, x);
    DenseMatrix P_inv =
        Eigen::LLT<DenseMatrix>{P}.solve(DenseMatrix::Identity(n, n));

    // vᵀx = Σᵢ vᵢᵢXᵢᵢ + Σᵢ>ⱼ vᵢⱼXᵢⱼ = tr(GX) with Gᵢᵢ = vᵢᵢ and
    // Gᵢⱼ = Gⱼᵢ = vᵢⱼ/2
    DenseMatrix G = unpack_symmetric(n, v) / Scalar(2);
    G.diagonal() *= Scalar(2);

    // The (r, s) index pairs with Eₐ = Σ eᵣeₛᵀ
    struct Coordinate {
      int i;
      int j;
    };
    gch::small_vector<Coordinate> coordinates;
    coordinates.reserve(m);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        coordinates.push_back({i, j});
      }
    }

    // tr(GEₐP⁻¹E_b)
    auto trace_term = [&](const Coordinate& a, const Coordinate& b) {
      Scalar sum{0};
      auto add = [&](int r, int s) {
        auto add_b = [&](int t, int u) { sum += G(u, r) * P_inv(s, t); };
        add_b(b.i, b.j);
        if (b.i != b.j) {
          add_b(b.j, b.i);
        }
      };
      add(a.i, a.j);
      if (a.i != a.j) {
        add(a.j, a.i);
      }
      return sum;
    };

    // Column-major lower triangle
    for (int b = 0; b < m; ++b) {
      for (int a = b; a < m; ++a) {
        Scalar value = (trace_term(coordinates[a], coordinates[b]) +
                        trace_term(coordinates[b], coordinates[a])) /
                       Scalar(2);
        triplets.emplace_back(tangent_offset + a, tangent_offset + b, value);
      }
    }
  }

  bool has_boundary() const override { return true; }

  /// Returns b(P) = −log det P.
  ///
  /// @param block The block.
  /// @param x The point on the manifold.
  /// @return The barrier function's value, or ∞ if P isn't positive definite.
  Scalar barrier(const ManifoldBlock<Scalar>& block,
                 Eigen::Ref<const DenseVector> x) const override {
    Eigen::LLT<DenseMatrix> llt{unpack_symmetric(block.rows(), x)};
    if (llt.info() != Eigen::Success) {
      return std::numeric_limits<Scalar>::infinity();
    }

    // log det P = 2 Σᵢ log Lᵢᵢ where P = LLᵀ
    return Scalar(-2) * llt.matrixLLT().diagonal().array().log().sum();
  }

  /// log det R_P(ξ) = log det P + tr(L⁻¹ξL⁻ᵀ) = log det P + tr(P⁻¹ξ) exactly,
  /// so b(R_P(ξ)) is linear in ξ and its Hessian is zero. The gradient entry
  /// of tangent coordinate (i, j) is −(P⁻¹)ᵢᵢ if i = j and −2(P⁻¹)ᵢⱼ
  /// otherwise.
  ///
  /// @param[in] block The block.
  /// @param[in] x The point on the manifold.
  /// @param[out] g The gradient in tangent coordinates.
  void barrier_gradient(const ManifoldBlock<Scalar>& block,
                        Eigen::Ref<const DenseVector> x,
                        Eigen::Ref<DenseVector> g) const override {
    const int n = block.rows();
    DenseMatrix P_inv = Eigen::LLT<DenseMatrix>{unpack_symmetric(n, x)}.solve(
        DenseMatrix::Identity(n, n));
    P_inv *= Scalar(-2);
    P_inv.diagonal() /= Scalar(2);
    pack_lower(P_inv, g);
  }

 private:
  /// Returns the index of entry (i, j), i ≥ j, in a row-major lower triangle.
  static constexpr int lower_index(int i, int j) { return i * (i + 1) / 2 + j; }

  /// Returns the n × n symmetric matrix whose lower triangle in row-major
  /// order is x.
  static DenseMatrix unpack_symmetric(int n, Eigen::Ref<const DenseVector> x) {
    DenseMatrix P{n, n};
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        P(i, j) = x[lower_index(i, j)];
        P(j, i) = x[lower_index(i, j)];
      }
    }
    return P;
  }

  /// Writes the lower triangle of P in row-major order to x.
  static void pack_lower(const DenseMatrix& P, Eigen::Ref<DenseVector> x) {
    for (int i = 0; i < P.rows(); ++i) {
      for (int j = 0; j <= i; ++j) {
        x[lower_index(i, j)] = P(i, j);
      }
    }
  }
};

/// The symmetric positive definite manifold type.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
inline const SPDManifold<Scalar> spd_manifold{};

}  // namespace slp
