// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/util/assert.hpp"

// See docs/algorithms.md#Manifold_optimization for a derivation of the
// retractions and Hessian corrections used here.

namespace slp {

/// Manifold type of a decision variable block.
enum class ManifoldType : uint8_t {
  /// Unit sphere Sⁿ⁻¹ in ℝⁿ.
  SPHERE,
  /// Unit complex numbers (cos θ, sin θ), which represent SO(2).
  SO2,
  /// Unit quaternions (w, x, y, z), which represent SO(3).
  SO3,
  /// Symmetric positive definite n×n matrices, stored as the lower triangle in
  /// row-major order.
  SPD
};

/// A contiguous block of decision variables constrained to a manifold.
struct ManifoldBlock {
  /// Manifold type.
  ManifoldType type;

  /// Index of the block's first decision variable.
  int offset;

  /// Number of decision variables in the block (the ambient dimension).
  int size;

  /// Returns the dimension of the block's tangent space.
  ///
  /// @return The dimension of the block's tangent space.
  constexpr int tangent_size() const {
    if (type == ManifoldType::SO3) {
      return 3;
    } else if (type == ManifoldType::SPD) {
      // The SPD matrices are an open subset of the symmetric matrices
      return size;
    } else {
      return size - 1;
    }
  }
};

/// The manifold the decision variables lie on.
///
/// Decision variables outside every block are Euclidean, so the manifold is a
/// product of the blocks' manifolds and a Euclidean space.
///
/// A point x on the manifold is stored as an ambient vector of
/// num_decision_variables entries. Steps p are tangent vectors with
/// tangent_dimension() entries, and the retraction R_x(p) maps them back onto
/// the manifold. Each tangent coordinate of a Euclidean decision variable is
/// that decision variable, so R_x(p) = x + p in a Euclidean space.
///
/// Optimizing f(x) on the manifold optimizes f(R_x(p)) over p near p = 0. Its
/// derivatives at p = 0 are
///
/// ```
///   ∇ₚf = Jᵀ∇ₓf
///   ∇ₚₚ²f = Jᵀ∇ₓₓ²fJ + Σᵢ (∇ₓf)ᵢ ∇ₚₚ²Rᵢ
/// ```
///
/// where J = ∂R/∂p is the retraction's Jacobian. The sphere, SO(2), and SO(3)
/// retractions have ∇ₚₚ²Rᵢ = −κxᵢI with a block-specific constant κ, so the
/// second term is −κ(xᵀ∇ₓf)I within those blocks. The SPD retraction's second
/// term is dense within its block.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class Manifold {
 public:
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  /// Type alias for sparse vector.
  using SparseVector = Eigen::SparseVector<Scalar>;

  /// Constructs a Euclidean space.
  Manifold() = default;

  /// Constructs a manifold from manifold blocks.
  ///
  /// @param blocks The manifold blocks, sorted by offset and non-overlapping.
  explicit Manifold(gch::small_vector<ManifoldBlock> blocks)
      : m_blocks{std::move(blocks)} {
    [[maybe_unused]]
    int end = 0;
    for (const auto& block : m_blocks) {
      slp_assert(block.offset >= end);
      slp_assert(block.tangent_size() >= 1);
      end = block.offset + block.size;
      m_tangent_dimension_reduction += block.size - block.tangent_size();
    }
  }

  /// Returns true if the manifold is a Euclidean space.
  ///
  /// @return True if the manifold is a Euclidean space.
  bool is_euclidean() const { return m_blocks.empty(); }

  /// Returns the manifold blocks.
  ///
  /// @return The manifold blocks.
  const gch::small_vector<ManifoldBlock>& blocks() const { return m_blocks; }

  /// Returns the dimension of the tangent space.
  ///
  /// @param ambient_dimension The number of decision variables.
  /// @return The dimension of the tangent space.
  int tangent_dimension(int ambient_dimension) const {
    return ambient_dimension - m_tangent_dimension_reduction;
  }

  /// Projects x onto the manifold.
  ///
  /// Each sphere, SO(2), and SO(3) block is normalized. A block with zero norm
  /// is set to the block's default point: e₁ for a sphere and the identity
  /// rotation for SO(2) and SO(3).
  ///
  /// An SPD block is unchanged if each eigenvalue is at least nε times the
  /// largest, where n is the matrix's dimension. Smaller eigenvalues are
  /// numerically zero, so they're raised to √ε times the largest, which keeps
  /// the condition number of their contribution at most 1/√ε. A block with no
  /// positive eigenvalue is set to the identity.
  ///
  /// @param[in,out] x The point to project.
  void project(DenseVector& x) const {
    for (const auto& block : m_blocks) {
      auto x_b = x.segment(block.offset, block.size);

      if (block.type == ManifoldType::SPD) {
        project_spd(x_b);
        continue;
      }

      Scalar norm = x_b.norm();
      if (norm > Scalar(0)) {
        x_b /= norm;
      } else {
        x_b.setZero();
        x_b[0] = Scalar(1);
      }
    }
  }

  /// Computes the retraction R_x(αp).
  ///
  /// @param[in] x The point on the manifold.
  /// @param[in] α The step size.
  /// @param[in] p The tangent vector.
  /// @param[out] result The retracted point.
  void retract(const DenseVector& x, Scalar α, const DenseVector& p,
               DenseVector& result) const {
    if (is_euclidean()) {
      result = x + α * p;
      return;
    }

    result.resize(x.rows());
    for_each_segment(
        x.rows(),
        [&](int ambient_offset, int tangent_offset, int size) {
          result.segment(ambient_offset, size) =
              x.segment(ambient_offset, size) +
              α * p.segment(tangent_offset, size);
        },
        [&](const ManifoldBlock& block, int tangent_offset) {
          auto x_b = x.segment(block.offset, block.size);
          auto result_b = result.segment(block.offset, block.size);
          DenseVector p_b = α * p.segment(tangent_offset, block.tangent_size());

          if (block.type == ManifoldType::SPD) {
            // R_P(ξ) = L exp(L⁻¹ξL⁻ᵀ) Lᵀ where P = LLᵀ
            //
            // exp() of a symmetric matrix is positive definite, so R_P(ξ) is
            // too. Since exp(M) = I + M + M²/2 + O(‖M‖³),
            //
            //   R_P(ξ) = P + ξ + ½ξP⁻¹ξ + O(‖ξ‖³)
            Eigen::LLT<Eigen::MatrixX<Scalar>> llt{unpack_symmetric(x_b)};
            Eigen::MatrixX<Scalar> L = llt.matrixL();

            // M = L⁻¹ξL⁻ᵀ
            Eigen::MatrixX<Scalar> M =
                llt.matrixL().solve(unpack_symmetric(p_b));
            M = llt.matrixL().solve(M.transpose().eval());

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigensolver{
                (M + M.transpose()) / Scalar(2)};

            // L exp(M) Lᵀ = (LV exp(Λ/2))(LV exp(Λ/2))ᵀ where M = VΛVᵀ
            Eigen::MatrixX<Scalar> W = L * eigensolver.eigenvectors() *
                                       (eigensolver.eigenvalues() / Scalar(2))
                                           .array()
                                           .exp()
                                           .matrix()
                                           .asDiagonal();
            pack_lower(W * W.transpose(), result_b);
            return;
          }

          if (block.type == ManifoldType::SPHERE) {
            // R_x(p) = (x + Bp)/‖x + Bp‖
            result_b = x_b + sphere_tangent_basis(x_b) * p_b;
          } else if (block.type == ManifoldType::SO2) {
            // R_z(p) = z exp(ip), which rotates z by p radians
            using std::cos;
            using std::sin;

            Scalar c = cos(p_b[0]);
            Scalar s = sin(p_b[0]);
            result_b = Eigen::Vector<Scalar, 2>{c * x_b[0] - s * x_b[1],
                                                s * x_b[0] + c * x_b[1]};
          } else {
            // R_q(p) = q ⊗ exp(p/2)
            //
            // exp(φ) = [cos‖φ‖, sin‖φ‖ φ/‖φ‖]
            using std::cos;
            using std::sin;
            using std::sqrt;

            Eigen::Vector<Scalar, 3> φ = p_b / Scalar(2);
            Scalar θ = φ.norm();

            // sin(θ)/θ = 1 − θ²/6 + θ⁴/120 − …, so the first two terms are
            // exact to machine precision when θ⁴/120 < ε
            Scalar sinc;
            if (θ < sqrt(sqrt(std::numeric_limits<Scalar>::epsilon()))) {
              sinc = Scalar(1) - θ * θ / Scalar(6);
            } else {
              sinc = sin(θ) / θ;
            }

            Eigen::Vector<Scalar, 4> r{cos(θ), sinc * φ[0], sinc * φ[1],
                                       sinc * φ[2]};
            result_b = quaternion_left_product_matrix(x_b) * r;
          }

          // Normalize to keep the iterate on the manifold despite roundoff
          result_b.normalize();
        });
  }

  /// Returns true if the manifold has a boundary, which is the case if it has
  /// an SPD block.
  ///
  /// @return True if the manifold has a boundary.
  bool has_boundary() const {
    return std::ranges::any_of(m_blocks, [](const ManifoldBlock& block) {
      return block.type == ManifoldType::SPD;
    });
  }

  /// Returns the boundary barrier function b(x) = −Σ log det P, where the sum
  /// is over SPD blocks P.
  ///
  /// @param x The point on the manifold.
  /// @return The barrier function's value, or ∞ if a block isn't positive
  ///     definite.
  Scalar barrier(const DenseVector& x) const {
    using std::log;

    Scalar b{0};
    for (const auto& block : m_blocks) {
      if (block.type != ManifoldType::SPD) {
        continue;
      }

      Eigen::LLT<Eigen::MatrixX<Scalar>> llt{
          unpack_symmetric(x.segment(block.offset, block.size))};
      if (llt.info() != Eigen::Success) {
        return std::numeric_limits<Scalar>::infinity();
      }

      // log det P = 2 Σᵢ log Lᵢᵢ where P = LLᵀ
      b -= Scalar(2) * llt.matrixLLT().diagonal().array().log().sum();
    }
    return b;
  }

  /// Returns the gradient of the boundary barrier function b(R_x(p)) with
  /// respect to p at p = 0.
  ///
  /// For an SPD block, log det R_P(ξ) = log det P + tr(L⁻¹ξL⁻ᵀ)
  /// = log det P + tr(P⁻¹ξ) exactly, so b(R_x(p)) is linear in p within each
  /// block and its Hessian with respect to p is zero. The gradient entry of
  /// tangent coordinate (i, j) is −(P⁻¹)ᵢᵢ if i = j and −2(P⁻¹)ᵢⱼ otherwise.
  ///
  /// @param x The point on the manifold.
  /// @return The gradient in tangent coordinates.
  DenseVector barrier_gradient(const DenseVector& x) const {
    DenseVector g = DenseVector::Zero(tangent_dimension(x.rows()));
    for_each_segment(
        x.rows(), [](int, int, int) {},
        [&](const ManifoldBlock& block, int tangent_offset) {
          if (block.type != ManifoldType::SPD) {
            return;
          }

          Eigen::MatrixX<Scalar> P =
              unpack_symmetric(x.segment(block.offset, block.size));
          Eigen::MatrixX<Scalar> P_inv =
              Eigen::LLT<Eigen::MatrixX<Scalar>>{P}.solve(
                  Eigen::MatrixX<Scalar>::Identity(P.rows(), P.cols()));
          P_inv *= Scalar(-2);
          P_inv.diagonal() /= Scalar(2);

          auto g_b = g.segment(tangent_offset, block.size);
          pack_lower(P_inv, g_b);
        });
    return g;
  }

  /// Returns the infinity norm of Jp/(1 + |x|), where J is the retraction's
  /// Jacobian at x, and the absolute value and division are elementwise.
  ///
  /// Jp is the first-order change in x along p, so this is the largest change
  /// in a decision variable relative to its magnitude.
  ///
  /// @param x The point on the manifold.
  /// @param p The tangent vector.
  /// @return The infinity norm of Jp/(1 + |x|).
  Scalar relative_step_inf_norm(const DenseVector& x,
                                const DenseVector& p) const {
    if (is_euclidean()) {
      return (p.array().abs() / (Scalar(1) + x.array().abs()))
          .matrix()
          .template lpNorm<Eigen::Infinity>();
    }

    DenseVector dx{x.rows()};
    for_each_segment(
        x.rows(),
        [&](int ambient_offset, int tangent_offset, int size) {
          dx.segment(ambient_offset, size) = p.segment(tangent_offset, size);
        },
        [&](const ManifoldBlock& block, int tangent_offset) {
          dx.segment(block.offset, block.size) =
              block_jacobian(block, x.segment(block.offset, block.size)) *
              p.segment(tangent_offset, block.tangent_size());
        });

    return (dx.array().abs() / (Scalar(1) + x.array().abs()))
        .matrix()
        .template lpNorm<Eigen::Infinity>();
  }

  /// Returns the retraction's Jacobian J = ∂R/∂p at p = 0.
  ///
  /// Every entry of a block's Jacobian is stored, even if it's zero, so the
  /// sparsity pattern doesn't depend on x.
  ///
  /// @param x The point on the manifold.
  /// @return The retraction's Jacobian.
  SparseMatrix jacobian(const DenseVector& x) const {
    gch::small_vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(x.rows());

    for_each_segment(
        x.rows(),
        [&](int ambient_offset, int tangent_offset, int size) {
          for (int i = 0; i < size; ++i) {
            triplets.emplace_back(ambient_offset + i, tangent_offset + i,
                                  Scalar(1));
          }
        },
        [&](const ManifoldBlock& block, int tangent_offset) {
          if (block.type == ManifoldType::SPD) {
            // J = I since R_P(ξ) = P + ξ + O(‖ξ‖²)
            for (int i = 0; i < block.size; ++i) {
              triplets.emplace_back(block.offset + i, tangent_offset + i,
                                    Scalar(1));
            }
            return;
          }

          auto J_b = block_jacobian(block, x.segment(block.offset, block.size));
          for (int col = 0; col < J_b.cols(); ++col) {
            for (int row = 0; row < J_b.rows(); ++row) {
              triplets.emplace_back(block.offset + row, tangent_offset + col,
                                    J_b(row, col));
            }
          }
        });

    SparseMatrix J{x.rows(), tangent_dimension(x.rows())};
    J.setFromTriplets(triplets.begin(), triplets.end());
    return J;
  }

  /// Returns the Hessian correction Σᵢ vᵢ ∇ₚₚ²Rᵢ at p = 0, where v is the
  /// gradient of a function with respect to x.
  ///
  /// Only the lower triangle is returned. It's diagonal for sphere, SO(2), and
  /// SO(3) blocks and dense for SPD blocks. Every entry in those patterns is
  /// stored, even if it's zero, so the sparsity pattern doesn't depend on x or
  /// v.
  ///
  /// @param x The point on the manifold.
  /// @param v The ambient gradient.
  /// @return The lower triangle of the Hessian correction.
  SparseMatrix hessian_correction(const DenseVector& x,
                                  const DenseVector& v) const {
    gch::small_vector<Eigen::Triplet<Scalar>> triplets;
    for_each_segment(
        x.rows(), [](int, int, int) {},
        [&](const ManifoldBlock& block, int tangent_offset) {
          if (block.type == ManifoldType::SPD) {
            spd_hessian_correction(x.segment(block.offset, block.size),
                                   v.segment(block.offset, block.size),
                                   tangent_offset, triplets);
            return;
          }

          // ∇ₚₚ²Rᵢ = −κxᵢI, so Σᵢ vᵢ ∇ₚₚ²Rᵢ = −κ(xᵀv)I
          Scalar value = -curvature_constant(block) *
                         x.segment(block.offset, block.size)
                             .dot(v.segment(block.offset, block.size));
          for (int i = 0; i < block.tangent_size(); ++i) {
            triplets.emplace_back(tangent_offset + i, tangent_offset + i,
                                  value);
          }
        });

    int k = tangent_dimension(x.rows());
    SparseMatrix C{k, k};
    C.setFromSortedTriplets(triplets.begin(), triplets.end());
    return C;
  }

 private:
  gch::small_vector<ManifoldBlock> m_blocks;
  int m_tangent_dimension_reduction = 0;

  /// Calls euclidean(ambient_offset, tangent_offset, size) for each run of
  /// Euclidean decision variables and block(block, tangent_offset) for each
  /// manifold block, in decision variable order.
  template <typename EuclideanFunc, typename BlockFunc>
  void for_each_segment(int ambient_dimension, EuclideanFunc&& euclidean,
                        BlockFunc&& block_func) const {
    int ambient_offset = 0;
    int tangent_offset = 0;
    for (const auto& block : m_blocks) {
      int size = block.offset - ambient_offset;
      if (size > 0) {
        euclidean(ambient_offset, tangent_offset, size);
      }
      tangent_offset += size;

      block_func(block, tangent_offset);
      ambient_offset = block.offset + block.size;
      tangent_offset += block.tangent_size();
    }

    int size = ambient_dimension - ambient_offset;
    if (size > 0) {
      euclidean(ambient_offset, tangent_offset, size);
    }
  }

  /// Returns κ in ∇ₚₚ²Rᵢ = −κxᵢI.
  static Scalar curvature_constant(const ManifoldBlock& block) {
    // Sphere: R_x(p) = (x + Bp)/√(1 + ‖p‖²) = x + Bp − x‖p‖²/2 + O(‖p‖³)
    //
    // SO(2): R_z(p) = z cos p + iz sin p = z + izp − zp²/2 + O(p³)
    //
    // SO(3): R_q(p) = q ⊗ exp(p/2) = q ⊗ [1 − ‖p‖²/8, p/2] + O(‖p‖³)
    //               = q + J p − q‖p‖²/8 + O(‖p‖³)
    return block.type == ManifoldType::SO3 ? Scalar(0.25) : Scalar(1);
  }

  /// Returns the block's retraction Jacobian at x_b.
  template <typename Derived>
  static Eigen::MatrixX<Scalar> block_jacobian(
      const ManifoldBlock& block, const Eigen::MatrixBase<Derived>& x_b) {
    if (block.type == ManifoldType::SPD) {
      return Eigen::MatrixX<Scalar>::Identity(block.size, block.size);
    } else if (block.type == ManifoldType::SPHERE) {
      return sphere_tangent_basis(x_b);
    } else if (block.type == ManifoldType::SO2) {
      // J = iz
      return Eigen::Vector<Scalar, 2>{-x_b[1], x_b[0]};
    } else {
      // J = ½ L(q)[:, 1:4]
      return quaternion_left_product_matrix(x_b).template rightCols<3>() /
             Scalar(2);
    }
  }

  /// Returns an orthonormal basis B for the tangent space of the unit sphere
  /// at x.
  ///
  /// The Householder reflection P = I − vvᵀ/(1 + |x₀|), where
  /// v = x + sign(x₀)e₀, is orthogonal and maps e₀ to −sign(x₀)x. Its other
  /// columns are orthonormal and orthogonal to x, so they're the basis.
  template <typename Derived>
  static Eigen::MatrixX<Scalar> sphere_tangent_basis(
      const Eigen::MatrixBase<Derived>& x) {
    using std::abs;

    int d = x.rows();
    Scalar sign = x[0] >= Scalar(0) ? Scalar(1) : Scalar(-1);
    DenseVector v = x;
    v[0] += sign;

    Eigen::MatrixX<Scalar> B =
        -v * v.segment(1, d - 1).transpose() / (Scalar(1) + abs(x[0]));
    B.bottomRows(d - 1).diagonal().array() += Scalar(1);
    return B;
  }

  /// Returns the dimension n of an n×n symmetric matrix whose lower triangle
  /// has size entries.
  static int symmetric_dimension(int size) {
    // size = n(n + 1)/2, so n = (√(8 size + 1) − 1)/2
    int n = 0;
    while (n * (n + 1) / 2 < size) {
      ++n;
    }
    slp_assert(n * (n + 1) / 2 == size);
    return n;
  }

  /// Returns the index of entry (i, j), i ≥ j, in a row-major lower triangle.
  static constexpr int lower_index(int i, int j) { return i * (i + 1) / 2 + j; }

  /// Returns the symmetric matrix whose lower triangle in row-major order is
  /// x.
  template <typename Derived>
  static Eigen::MatrixX<Scalar> unpack_symmetric(
      const Eigen::MatrixBase<Derived>& x) {
    int n = symmetric_dimension(x.rows());
    Eigen::MatrixX<Scalar> P{n, n};
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        P(i, j) = x[lower_index(i, j)];
        P(j, i) = x[lower_index(i, j)];
      }
    }
    return P;
  }

  /// Writes the lower triangle of P in row-major order to x.
  template <typename Derived>
  static void pack_lower(const Eigen::MatrixX<Scalar>& P,
                         Eigen::MatrixBase<Derived>& x) {
    for (int i = 0; i < P.rows(); ++i) {
      for (int j = 0; j <= i; ++j) {
        x[lower_index(i, j)] = P(i, j);
      }
    }
  }

  /// Projects the lower triangle x_b of a symmetric matrix onto the SPD
  /// matrices.
  template <typename Derived>
  static void project_spd(Eigen::MatrixBase<Derived>& x_b) {
    using std::sqrt;

    constexpr Scalar ε = std::numeric_limits<Scalar>::epsilon();

    Eigen::MatrixX<Scalar> P = unpack_symmetric(x_b);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigensolver{P};
    DenseVector λ = eigensolver.eigenvalues();

    Scalar λ_max = λ.maxCoeff();
    if (!(λ_max > Scalar(0))) {
      P.setIdentity();
    } else {
      // The eigensolver's error is about nελ_max, so smaller eigenvalues are
      // indistinguishable from zero
      Scalar λ_zero = Scalar(P.rows()) * ε * λ_max;
      if (λ.minCoeff() >= λ_zero) {
        return;
      }

      λ = (λ.array() < λ_zero).select(sqrt(ε) * λ_max, λ);
      const auto& V = eigensolver.eigenvectors();
      P = V * λ.asDiagonal() * V.transpose();
    }
    pack_lower(P, x_b);
  }

  /// Appends the lower triangle of an SPD block's Hessian correction.
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
  template <typename Derived1, typename Derived2>
  static void spd_hessian_correction(
      const Eigen::MatrixBase<Derived1>& x_b,
      const Eigen::MatrixBase<Derived2>& v_b, int tangent_offset,
      gch::small_vector<Eigen::Triplet<Scalar>>& triplets) {
    const int m = x_b.rows();
    const int n = symmetric_dimension(m);

    Eigen::MatrixX<Scalar> P = unpack_symmetric(x_b);
    Eigen::MatrixX<Scalar> P_inv = Eigen::LLT<Eigen::MatrixX<Scalar>>{P}.solve(
        Eigen::MatrixX<Scalar>::Identity(n, n));

    // vᵀx = Σᵢ vᵢᵢXᵢᵢ + Σᵢ>ⱼ vᵢⱼXᵢⱼ = tr(GX) with Gᵢᵢ = vᵢᵢ and
    // Gᵢⱼ = Gⱼᵢ = vᵢⱼ/2
    Eigen::MatrixX<Scalar> G = unpack_symmetric(v_b) / Scalar(2);
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

    // Column-major lower triangle, as setFromSortedTriplets() requires
    for (int b = 0; b < m; ++b) {
      for (int a = b; a < m; ++a) {
        Scalar value = (trace_term(coordinates[a], coordinates[b]) +
                        trace_term(coordinates[b], coordinates[a])) /
                       Scalar(2);
        triplets.emplace_back(tangent_offset + a, tangent_offset + b, value);
      }
    }
  }

  /// Returns the matrix L(q) such that q ⊗ r = L(q)r for quaternions in
  /// (w, x, y, z) order.
  template <typename Derived>
  static Eigen::Matrix<Scalar, 4, 4> quaternion_left_product_matrix(
      const Eigen::MatrixBase<Derived>& q) {
    Scalar w = q[0];
    Scalar x = q[1];
    Scalar y = q[2];
    Scalar z = q[3];

    return Eigen::Matrix<Scalar, 4, 4>{
        {w, -x, -y, -z}, {x, w, -z, y}, {y, z, w, -x}, {z, -y, x, w}};
  }
};

}  // namespace slp
