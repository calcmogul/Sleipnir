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

/// Scalar absolute tolerance matcher.
template <typename T>
struct ScalarWithinAbsMatcher : Catch::Matchers::MatcherGenericBase {
  ScalarWithinAbsMatcher(T target, T margin)
      : target{std::move(target)}, margin{std::move(margin)} {}

  bool match(const T& matchee) const {
    using std::abs;
    return abs(target - matchee) <= margin;
  }

  std::string describe() const override {
    return std::format("is within {} of {}", margin, target);
  }

 private:
  T target;
  T margin;
};

/// Creates a scalar absolute tolerance matcher.
template <typename T>
ScalarWithinAbsMatcher<T> WithinAbs(T target, T margin) {
  return {std::move(target), std::move(margin)};
}

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

/// Matrix absolute tolerance matcher.
template <typename Matrix>
  requires DenseMatrix<Matrix> || SparseMatrix<Matrix>
struct MatrixWithinAbsMatcher : Catch::Matchers::MatcherGenericBase {
  using Scalar = typename Matrix::Scalar;

  MatrixWithinAbsMatcher(Matrix target, Scalar margin)
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
    return std::format("\nis within {} elementwise of\n{}", margin,
                       Catch::StringMaker<Matrix>::convert(target));
  }

 private:
  Matrix target;
  Scalar margin;
};

/// Creates a dense matrix absolute tolerance matcher.
///
/// Uses a dynamic-size target so matchees with mismatched sizes fail the match
/// instead of Eigen producing a resize assertion.
template <typename Derived>
MatrixWithinAbsMatcher<
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>>
WithinAbs(const Eigen::DenseBase<Derived>& target,
          typename Derived::Scalar margin) {
  return {target, margin};
}

/// Creates a sparse matrix absolute tolerance matcher.
template <typename Derived>
MatrixWithinAbsMatcher<Eigen::SparseMatrix<typename Derived::Scalar>> WithinAbs(
    const Eigen::SparseMatrixBase<Derived>& target,
    typename Derived::Scalar margin) {
  return {target, margin};
}
