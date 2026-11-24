// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

namespace slp {

template <typename Scalar>
class ManifoldBlock;

/// A manifold that a block of decision variables can be constrained to.
///
/// A manifold type holds no per-block state, so one instance can serve every
/// block of that type. Each block refers to its type and supplies the shape,
/// offset, and sizes (see ManifoldBlock).
///
/// A point x on a block is stored as block.size() ambient decision variables,
/// and steps p are tangent vectors with block.tangent_size() entries. The
/// retraction R_x(p) maps a step back onto the manifold, and must satisfy
/// R_x(0) = x and have a Jacobian J = ∂R/∂p at p = 0 of full column rank.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class ManifoldType {
 public:
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;
  /// Type alias for dense matrix.
  using DenseMatrix = Eigen::MatrixX<Scalar>;
  /// Type alias for a list of sparse matrix triplets.
  using Triplets = gch::small_vector<Eigen::Triplet<Scalar>>;

  virtual ~ManifoldType() = default;

  /// Returns the number of decision variables that store a rows × cols point.
  ///
  /// @param rows The number of rows in the point's matrix shape.
  /// @param cols The number of columns in the point's matrix shape.
  /// @return The ambient dimension.
  virtual int ambient_size(int rows, int cols) const = 0;

  /// Returns the dimension of the tangent space at a rows × cols point.
  ///
  /// @param rows The number of rows in the point's matrix shape.
  /// @param cols The number of columns in the point's matrix shape.
  /// @return The dimension of the tangent space.
  virtual int tangent_size(int rows, int cols) const = 0;

  /// Projects x onto the manifold.
  ///
  /// @param[in] block The block being projected.
  /// @param[in,out] x The block's decision variables.
  virtual void project(const ManifoldBlock<Scalar>& block,
                       Eigen::Ref<DenseVector> x) const = 0;

  /// Computes the retraction R_x(p).
  ///
  /// @param[in] block The block being retracted.
  /// @param[in] x The point on the manifold.
  /// @param[in] p The tangent vector.
  /// @param[out] result The retracted point.
  virtual void retract(const ManifoldBlock<Scalar>& block,
                       Eigen::Ref<const DenseVector> x,
                       Eigen::Ref<const DenseVector> p,
                       Eigen::Ref<DenseVector> result) const = 0;

  /// Returns the retraction's Jacobian J = ∂R/∂p at p = 0.
  ///
  /// @param block The block.
  /// @param x The point on the manifold.
  /// @return The block.size() × block.tangent_size() Jacobian.
  virtual DenseMatrix jacobian(const ManifoldBlock<Scalar>& block,
                               Eigen::Ref<const DenseVector> x) const = 0;

  /// Appends the retraction's Jacobian J = ∂R/∂p at p = 0.
  ///
  /// The default appends every entry of jacobian(block, x), even if it's
  /// zero, so the sparsity pattern doesn't depend on x. Types whose Jacobian
  /// has a sparse pattern independent of x can override this to append only
  /// that pattern.
  ///
  /// @param[in] block The block.
  /// @param[in] x The point on the manifold.
  /// @param[in] tangent_offset Index of the block's first tangent coordinate.
  /// @param[in,out] triplets The triplets to append to.
  virtual void append_jacobian(const ManifoldBlock<Scalar>& block,
                               Eigen::Ref<const DenseVector> x,
                               int tangent_offset, Triplets& triplets) const;

  /// Appends the lower triangle of the Hessian correction Σᵢ vᵢ ∇ₚₚ²Rᵢ at
  /// p = 0, where v is the gradient of a function with respect to x.
  ///
  /// Triplets are appended in column-major order. Every entry in the
  /// correction's sparsity pattern is appended, even if it's zero, so the
  /// pattern doesn't depend on x or v.
  ///
  /// @param[in] block The block.
  /// @param[in] x The point on the manifold.
  /// @param[in] v The ambient gradient.
  /// @param[in] tangent_offset Index of the block's first tangent coordinate.
  /// @param[in,out] triplets The triplets to append to.
  virtual void hessian_correction(const ManifoldBlock<Scalar>& block,
                                  Eigen::Ref<const DenseVector> x,
                                  Eigen::Ref<const DenseVector> v,
                                  int tangent_offset,
                                  Triplets& triplets) const = 0;

  /// Returns true if the manifold has a boundary the iterates must stay
  /// inside.
  ///
  /// A manifold with a boundary needs the interior-point method's barrier
  /// function.
  ///
  /// @return True if the manifold has a boundary.
  virtual bool has_boundary() const { return false; }

  /// Returns the boundary barrier function b(x).
  ///
  /// @param block The block.
  /// @param x The point on the manifold.
  /// @return The barrier function's value, or ∞ if x is outside the boundary.
  virtual Scalar barrier(
      [[maybe_unused]] const ManifoldBlock<Scalar>& block,
      [[maybe_unused]] Eigen::Ref<const DenseVector> x) const {
    return Scalar(0);
  }

  /// Computes the gradient of b(R_x(p)) with respect to p at p = 0.
  ///
  /// The barrier's Hessian with respect to p must be zero at p = 0.
  ///
  /// @param[in] block The block.
  /// @param[in] x The point on the manifold.
  /// @param[out] g The gradient in tangent coordinates, which is zero on
  ///     entry.
  virtual void barrier_gradient(
      [[maybe_unused]] const ManifoldBlock<Scalar>& block,
      [[maybe_unused]] Eigen::Ref<const DenseVector> x,
      [[maybe_unused]] Eigen::Ref<DenseVector> g) const {}
};

/// A contiguous block of decision variables constrained to a manifold.
///
/// The block's decision variables store a point with a rows × cols matrix
/// shape. The manifold type maps the shape to the number of decision
/// variables, which need not be rows × cols (e.g., a symmetric matrix stores
/// only its lower triangle).
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class ManifoldBlock {
 public:
  /// Constructs a manifold block.
  ///
  /// @param type The manifold type, which must outlive the block.
  /// @param offset Index of the block's first decision variable.
  /// @param rows The number of rows in the point's matrix shape.
  /// @param cols The number of columns in the point's matrix shape.
  ManifoldBlock(const ManifoldType<Scalar>& type, int offset, int rows,
                int cols = 1)
      : m_type{&type},
        m_offset{offset},
        m_rows{rows},
        m_cols{cols},
        m_size{type.ambient_size(rows, cols)},
        m_tangent_size{type.tangent_size(rows, cols)} {}

  /// A temporary manifold type would dangle.
  ManifoldBlock(const ManifoldType<Scalar>&& type, int offset, int rows,
                int cols = 1) = delete;

  /// Returns the manifold type.
  ///
  /// @return The manifold type.
  const ManifoldType<Scalar>& type() const { return *m_type; }

  /// Returns the index of the block's first decision variable.
  ///
  /// @return The index of the block's first decision variable.
  int offset() const { return m_offset; }

  /// Returns the number of rows in the point's matrix shape.
  ///
  /// @return The number of rows in the point's matrix shape.
  int rows() const { return m_rows; }

  /// Returns the number of columns in the point's matrix shape.
  ///
  /// @return The number of columns in the point's matrix shape.
  int cols() const { return m_cols; }

  /// Returns the number of decision variables in the block (the ambient
  /// dimension).
  ///
  /// @return The number of decision variables in the block.
  int size() const { return m_size; }

  /// Returns the dimension of the block's tangent space.
  ///
  /// @return The dimension of the block's tangent space.
  int tangent_size() const { return m_tangent_size; }

 private:
  const ManifoldType<Scalar>* m_type;
  int m_offset;
  int m_rows;
  int m_cols;
  int m_size;
  int m_tangent_size;
};

template <typename Scalar>
void ManifoldType<Scalar>::append_jacobian(const ManifoldBlock<Scalar>& block,
                                           Eigen::Ref<const DenseVector> x,
                                           int tangent_offset,
                                           Triplets& triplets) const {
  DenseMatrix J = jacobian(block, x);
  for (int col = 0; col < J.cols(); ++col) {
    for (int row = 0; row < J.rows(); ++row) {
      triplets.emplace_back(block.offset() + row, tangent_offset + col,
                            J(row, col));
    }
  }
}

/// Appends value times the identity on a block's tangent coordinates.
///
/// @param[in] value The diagonal value.
/// @param[in] tangent_offset Index of the block's first tangent coordinate.
/// @param[in] tangent_size Dimension of the block's tangent space.
/// @param[in,out] triplets The triplets to append to.
template <typename Scalar>
void append_scaled_identity(
    Scalar value, int tangent_offset, int tangent_size,
    gch::small_vector<Eigen::Triplet<Scalar>>& triplets) {
  for (int i = 0; i < tangent_size; ++i) {
    triplets.emplace_back(tangent_offset + i, tangent_offset + i, value);
  }
}

}  // namespace slp
