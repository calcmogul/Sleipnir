// Copyright (c) Sleipnir contributors

#pragma once

#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/ipm_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/newton_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/sqp_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/util/manifold.hpp"

namespace slp {

/// Converts matrix callbacks that take derivatives with respect to the ambient
/// decision variables x into matrix callbacks that take derivatives with
/// respect to the tangent vector p of the retraction R_x(p) at p = 0.
///
/// The returned callbacks reference this object and the ambient callbacks, so
/// both must outlive them.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class TangentSpace {
 public:
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  /// Type alias for sparse vector.
  using SparseVector = Eigen::SparseVector<Scalar>;

  /// Constructs a TangentSpace.
  ///
  /// @param manifold The manifold the decision variables lie on.
  explicit TangentSpace(const Manifold<Scalar>& manifold)
      : m_manifold{manifold} {}

  TangentSpace(const TangentSpace&) = delete;
  TangentSpace& operator=(const TangentSpace&) = delete;

  /// Returns Newton matrix callbacks in the tangent space.
  ///
  /// @param ambient Newton matrix callbacks in the ambient space.
  /// @return Newton matrix callbacks in the tangent space.
  NewtonMatrixCallbacks<Scalar> callbacks(
      const NewtonMatrixCallbacks<Scalar>& ambient) {
    return NewtonMatrixCallbacks<Scalar>{
        m_manifold.tangent_dimension(ambient.num_decision_variables),
        ambient.f,
        [&](const DenseVector& x) -> SparseVector {
          return gradient(x, ambient.g);
        },
        [&](const DenseVector& x) -> SparseMatrix {
          DenseVector v = cached(m_g, x, ambient.g).toDense();
          return hessian(x, ambient.H(x), v);
        },
        ambient.scaling,
        {}};
  }

  /// Returns SQP matrix callbacks in the tangent space.
  ///
  /// @param ambient SQP matrix callbacks in the ambient space.
  /// @return SQP matrix callbacks in the tangent space.
  SQPMatrixCallbacks<Scalar> callbacks(
      const SQPMatrixCallbacks<Scalar>& ambient) {
    return SQPMatrixCallbacks<Scalar>{
        m_manifold.tangent_dimension(ambient.num_decision_variables),
        ambient.num_equality_constraints,
        ambient.f,
        [&](const DenseVector& x) -> SparseVector {
          return gradient(x, ambient.g);
        },
        [&](const DenseVector& x, const DenseVector& y) -> SparseMatrix {
          // ∇ₓL = ∇f − Aₑᵀy
          DenseVector v = cached(m_g, x, ambient.g).toDense();
          v -= cached(m_A_e, x, ambient.A_e).transpose() * y;
          return hessian(x, ambient.H(x, y), v);
        },
        [&](const DenseVector& x, const DenseVector& y) -> SparseMatrix {
          // −Aₑᵀy
          DenseVector v = -(cached(m_A_e, x, ambient.A_e).transpose() * y);
          return hessian(x, ambient.H_c(x, y), v);
        },
        ambient.c_e,
        [&](const DenseVector& x) -> SparseMatrix {
          return jacobian(x, m_A_e, ambient.A_e);
        },
        ambient.scaling,
        {}};
  }

  /// Returns interior-point method matrix callbacks in the tangent space.
  ///
  /// @param ambient Interior-point method matrix callbacks in the ambient
  ///     space.
  /// @return Interior-point method matrix callbacks in the tangent space.
  IPMMatrixCallbacks<Scalar> callbacks(
      const IPMMatrixCallbacks<Scalar>& ambient) {
    return IPMMatrixCallbacks<Scalar>{
        m_manifold.tangent_dimension(ambient.num_decision_variables),
        ambient.num_equality_constraints,
        ambient.num_inequality_constraints,
        ambient.f,
        [&](const DenseVector& x) -> SparseVector {
          return gradient(x, ambient.g);
        },
        [&](const DenseVector& x, const DenseVector& y,
            const DenseVector& z) -> SparseMatrix {
          // ∇ₓL = ∇f − Aₑᵀy − Aᵢᵀz
          DenseVector v = cached(m_g, x, ambient.g).toDense();
          v -= cached(m_A_e, x, ambient.A_e).transpose() * y;
          v -= cached(m_A_i, x, ambient.A_i).transpose() * z;
          return hessian(x, ambient.H(x, y, z), v);
        },
        [&](const DenseVector& x, const DenseVector& y,
            const DenseVector& z) -> SparseMatrix {
          // −Aₑᵀy − Aᵢᵀz
          DenseVector v = -(cached(m_A_e, x, ambient.A_e).transpose() * y);
          v -= cached(m_A_i, x, ambient.A_i).transpose() * z;
          return hessian(x, ambient.H_c(x, y, z), v);
        },
        ambient.c_e,
        [&](const DenseVector& x) -> SparseMatrix {
          return jacobian(x, m_A_e, ambient.A_e);
        },
        ambient.c_i,
        [&](const DenseVector& x) -> SparseMatrix {
          return jacobian(x, m_A_i, ambient.A_i);
        },
        ambient.scaling,
        {}};
  }

 private:
  /// A value computed at a point x.
  template <typename T>
  struct Cache {
    DenseVector x;
    T value;
  };

  const Manifold<Scalar>& m_manifold;

  // The Hessian correction needs the ambient gradient and constraint
  // Jacobians, which the solvers usually just evaluated at the same x, so
  // they're cached by x
  Cache<SparseMatrix> m_J;
  Cache<SparseVector> m_g;
  Cache<SparseMatrix> m_A_e;
  Cache<SparseMatrix> m_A_i;

  /// Returns cache's value at x, calling f(x) to update it if it's stale.
  template <typename T, typename F>
  static const T& cached(Cache<T>& cache, const DenseVector& x, F&& f) {
    if (cache.x.rows() != x.rows() || cache.x != x) {
      cache.value = f(x);
      cache.x = x;
    }
    return cache.value;
  }

  /// Returns the retraction's Jacobian J at x.
  const SparseMatrix& retraction_jacobian(const DenseVector& x) {
    return cached(m_J, x,
                  [&](const DenseVector& x) { return m_manifold.jacobian(x); });
  }

  /// Returns Jᵀ∇f.
  template <typename F>
  SparseVector gradient(const DenseVector& x, F&& g) {
    const auto& ambient_g = cached(m_g, x, g);
    return retraction_jacobian(x).transpose() * ambient_g;
  }

  /// Returns AJ.
  template <typename F>
  SparseMatrix jacobian(const DenseVector& x, Cache<SparseMatrix>& cache,
                        F&& A) {
    const auto& ambient_A = cached(cache, x, A);
    return ambient_A * retraction_jacobian(x);
  }

  /// Returns the lower triangle of JᵀHJ + Σᵢ vᵢ∇ₚₚ²Rᵢ, where H is a lower
  /// triangular ambient Hessian and v is the gradient of the same function.
  SparseMatrix hessian(const DenseVector& x, const SparseMatrix& H,
                       const DenseVector& v) {
    const auto& J = retraction_jacobian(x);

    SparseMatrix H_full = H.template selfadjointView<Eigen::Lower>();
    SparseMatrix JᵀHJ = J.transpose() * H_full * J;
    SparseMatrix result = JᵀHJ.template triangularView<Eigen::Lower>();
    return result + m_manifold.hessian_correction(x, v);
  }
};

}  // namespace slp
