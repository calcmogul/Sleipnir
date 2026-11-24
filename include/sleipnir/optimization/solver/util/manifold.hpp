// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

#include "sleipnir/optimization/solver/util/manifold_type.hpp"
#include "sleipnir/util/assert.hpp"

// See docs/algorithms.md#Manifold_optimization for a derivation of the
// retractions and Hessian corrections used here.

namespace slp {

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
/// where J = ∂R/∂p is the retraction's Jacobian. Each block's manifold type
/// supplies its part of J and of the second term (see ManifoldType).
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class Manifold {
 public:
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  /// Constructs a Euclidean space.
  Manifold() = default;

  /// Constructs a manifold from manifold blocks.
  ///
  /// @param blocks The manifold blocks, sorted by offset and non-overlapping.
  explicit Manifold(gch::small_vector<ManifoldBlock<Scalar>> blocks)
      : m_blocks{std::move(blocks)} {
    [[maybe_unused]]
    int end = 0;
    for (const auto& block : m_blocks) {
      slp_assert(block.offset() >= end);
      slp_assert(block.tangent_size() >= 1);
      end = block.offset() + block.size();
      m_tangent_dimension_reduction += block.size() - block.tangent_size();
      m_has_boundary = m_has_boundary || block.type().has_boundary();
    }
  }

  /// Returns true if the manifold is a Euclidean space.
  ///
  /// @return True if the manifold is a Euclidean space.
  bool is_euclidean() const { return m_blocks.empty(); }

  /// Returns the manifold blocks.
  ///
  /// @return The manifold blocks.
  const gch::small_vector<ManifoldBlock<Scalar>>& blocks() const {
    return m_blocks;
  }

  /// Returns the dimension of the tangent space.
  ///
  /// @param ambient_dimension The number of decision variables.
  /// @return The dimension of the tangent space.
  int tangent_dimension(int ambient_dimension) const {
    return ambient_dimension - m_tangent_dimension_reduction;
  }

  /// Projects x onto the manifold.
  ///
  /// @param[in,out] x The point to project.
  void project(DenseVector& x) const {
    for (const auto& block : m_blocks) {
      block.type().project(block, x.segment(block.offset(), block.size()));
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
        [&](const ManifoldBlock<Scalar>& block, int tangent_offset) {
          DenseVector p_b = α * p.segment(tangent_offset, block.tangent_size());
          block.type().retract(block, x.segment(block.offset(), block.size()),
                               p_b,
                               result.segment(block.offset(), block.size()));
        });
  }

  /// Returns true if the manifold has a boundary, which is the case if any
  /// block has one.
  ///
  /// @return True if the manifold has a boundary.
  bool has_boundary() const { return m_has_boundary; }

  /// Returns the boundary barrier function b(x), the sum of the blocks'
  /// barrier functions.
  ///
  /// @param x The point on the manifold.
  /// @return The barrier function's value, or ∞ if x is outside a block's
  ///     boundary.
  Scalar barrier(const DenseVector& x) const {
    Scalar b{0};
    for (const auto& block : m_blocks) {
      if (block.type().has_boundary()) {
        b += block.type().barrier(block,
                                  x.segment(block.offset(), block.size()));
      }
    }
    return b;
  }

  /// Returns the gradient of the boundary barrier function b(R_x(p)) with
  /// respect to p at p = 0.
  ///
  /// @param x The point on the manifold.
  /// @return The gradient in tangent coordinates.
  DenseVector barrier_gradient(const DenseVector& x) const {
    DenseVector g = DenseVector::Zero(tangent_dimension(x.rows()));
    for_each_segment(
        x.rows(), [](int, int, int) {},
        [&](const ManifoldBlock<Scalar>& block, int tangent_offset) {
          if (block.type().has_boundary()) {
            block.type().barrier_gradient(
                block, x.segment(block.offset(), block.size()),
                g.segment(tangent_offset, block.tangent_size()));
          }
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
        [&](const ManifoldBlock<Scalar>& block, int tangent_offset) {
          dx.segment(block.offset(), block.size()) =
              block.type().jacobian(block,
                                    x.segment(block.offset(), block.size())) *
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
        [&](const ManifoldBlock<Scalar>& block, int tangent_offset) {
          block.type().append_jacobian(block,
                                       x.segment(block.offset(), block.size()),
                                       tangent_offset, triplets);
        });

    SparseMatrix J{x.rows(), tangent_dimension(x.rows())};
    J.setFromTriplets(triplets.begin(), triplets.end());
    return J;
  }

  /// Returns the Hessian correction Σᵢ vᵢ ∇ₚₚ²Rᵢ at p = 0, where v is the
  /// gradient of a function with respect to x.
  ///
  /// Only the lower triangle is returned. Every entry in the blocks' sparsity
  /// patterns is stored, even if it's zero, so the sparsity pattern doesn't
  /// depend on x or v.
  ///
  /// @param x The point on the manifold.
  /// @param v The ambient gradient.
  /// @return The lower triangle of the Hessian correction.
  SparseMatrix hessian_correction(const DenseVector& x,
                                  const DenseVector& v) const {
    gch::small_vector<Eigen::Triplet<Scalar>> triplets;
    for_each_segment(
        x.rows(), [](int, int, int) {},
        [&](const ManifoldBlock<Scalar>& block, int tangent_offset) {
          block.type().hessian_correction(
              block, x.segment(block.offset(), block.size()),
              v.segment(block.offset(), block.size()), tangent_offset,
              triplets);
        });

    // The blocks are diagonal blocks in increasing order, and each appends
    // its triplets in column-major order, so all of them are sorted
    int k = tangent_dimension(x.rows());
    SparseMatrix C{k, k};
    C.setFromSortedTriplets(triplets.begin(), triplets.end());
    return C;
  }

 private:
  gch::small_vector<ManifoldBlock<Scalar>> m_blocks;
  int m_tangent_dimension_reduction = 0;
  bool m_has_boundary = false;

  /// Calls euclidean(ambient_offset, tangent_offset, size) for each run of
  /// Euclidean decision variables and block_func(block, tangent_offset) for
  /// each manifold block, in decision variable order.
  template <typename EuclideanFunc, typename BlockFunc>
  void for_each_segment(int ambient_dimension, EuclideanFunc&& euclidean,
                        BlockFunc&& block_func) const {
    int ambient_offset = 0;
    int tangent_offset = 0;
    for (const auto& block : m_blocks) {
      int size = block.offset() - ambient_offset;
      if (size > 0) {
        euclidean(ambient_offset, tangent_offset, size);
      }
      tangent_offset += size;

      block_func(block, tangent_offset);
      ambient_offset = block.offset() + block.size();
      tangent_offset += block.tangent_size();
    }

    int size = ambient_dimension - ambient_offset;
    if (size > 0) {
      euclidean(ambient_offset, tangent_offset, size);
    }
  }
};

}  // namespace slp
