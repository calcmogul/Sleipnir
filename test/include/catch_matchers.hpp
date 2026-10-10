// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <concepts>
#include <format>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_tostring.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

template <typename T>
struct WithinAbs : Catch::Matchers::MatcherGenericBase {
  WithinAbs(T target, T margin) : target{target}, margin{margin} {}

  bool match(const T& matchee) const {
    using std::abs;
    return abs(target - matchee) <= margin;
  }

  std::string describe() const override {
    return std::format("\n==\n{}", target);
  }

 private:
  T target;
  T margin;
};

/// Any Eigen::Matrix (i.e., not an expression template).
template <typename T>
concept DenseMatrix = std::same_as<
    T, Eigen::Matrix<typename T::Scalar, T::RowsAtCompileTime,
                     T::ColsAtCompileTime, T::Options, T::MaxRowsAtCompileTime,
                     T::MaxColsAtCompileTime>>;

/// Any Eigen::SparseMatrix (i.e., not an expression template).
template <typename T>
concept SparseMatrix =
    std::same_as<T, Eigen::SparseMatrix<typename T::Scalar, T::Options,
                                        typename T::StorageIndex>>;

template <typename Matrix>
  requires DenseMatrix<Matrix> || SparseMatrix<Matrix>
struct MatrixWithinAbs : Catch::Matchers::MatcherGenericBase {
  using Scalar = typename Matrix::Scalar;

  MatrixWithinAbs(Matrix target, Scalar margin)
      : target{std::move(target)}, margin{margin} {}

  bool match(const Matrix& matchee) const {
    using std::abs;
    using std::isnan;

    if (target.rows() != matchee.rows() || target.cols() != matchee.cols()) {
      return false;
    }

    Matrix error = target - matchee;

    if constexpr (DenseMatrix<Matrix>) {
      for (int row = 0; row < error.rows(); ++row) {
        for (int col = 0; col < error.cols(); ++col) {
          if (isnan(error(row, col)) || abs(error(row, col)) > margin) {
            return false;
          }
        }
      }
    } else {
      for (int col = 0; col < error.outerSize(); ++col) {
        for (typename Matrix::InnerIterator it{error, col}; it; ++it) {
          if (isnan(it.value()) || abs(it.value()) > margin) {
            return false;
          }
        }
      }
    }

    return true;
  }

  /// Prevents implicit sparse-to-dense conversion of matchee.
  template <typename Derived>
    requires DenseMatrix<Matrix>
  bool match(const Eigen::SparseMatrixBase<Derived>& matchee) const = delete;

  /// Prevents implicit dense-to-sparse conversion of matchee.
  template <typename Derived>
    requires SparseMatrix<Matrix>
  bool match(const Eigen::DenseBase<Derived>& matchee) const = delete;

  std::string describe() const override {
    return std::format("\n==\n{}", Catch::Detail::stringify(target));
  }

 private:
  Matrix target;
  Scalar margin;
};

/// Plain dense matrix deduces dynamic-size dense matrix specialization so
/// matchees with mismatched sizes fail the match instead of Eigen producing a
/// resize assertion.
template <DenseMatrix M>
MatrixWithinAbs(const M&, typename M::Scalar) -> MatrixWithinAbs<
    Eigen::Matrix<typename M::Scalar, Eigen::Dynamic, Eigen::Dynamic>>;

/// Dense expression template deduces dense matrix specialization.
template <typename Derived>
MatrixWithinAbs(const Eigen::DenseBase<Derived>&, typename Derived::Scalar)
    -> MatrixWithinAbs<Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic,
                                     Eigen::Dynamic>>;

/// Sparse expression template deduces sparse matrix specialization.
template <typename Derived>
MatrixWithinAbs(const Eigen::SparseMatrixBase<Derived>&,
                typename Derived::Scalar)
    -> MatrixWithinAbs<Eigen::SparseMatrix<typename Derived::Scalar>>;
