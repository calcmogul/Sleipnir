// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <concepts>
#include <sstream>
#include <string>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/matchers/catch_matchers_templated.hpp>

template <typename Derived>
  requires std::derived_from<Derived, Eigen::DenseBase<Derived>> ||
           std::derived_from<Derived, Eigen::SparseCompressedBase<Derived>>
struct ApproxMatrix : Catch::Matchers::MatcherGenericBase {
  ApproxMatrix(const Derived& mat, typename Derived::Scalar tol)
      : mat{mat}, tol{tol} {}

  template <typename OtherDerived>
    requires std::derived_from<OtherDerived, Eigen::DenseBase<Derived>> ||
             std::derived_from<OtherDerived,
                               Eigen::SparseCompressedBase<Derived>>
  bool match(const OtherDerived& other) const {
    using std::abs;

    if (mat.rows() != other.rows() || mat.cols() != other.cols()) {
      return false;
    }

    for (Eigen::Index row = 0; row < mat.rows(); ++row) {
      for (Eigen::Index col = 0; col < mat.cols(); ++col) {
        if (abs(mat.coeff(row, col) - other.coeff(row, col)) > tol) {  // NOLINT
          return false;
        }
      }
    }

    return true;
  }

  std::string describe() const override {
    return (std::ostringstream{} << "\n==\n" << mat).str();
  }

 private:
  const Derived& mat;
  typename Derived::Scalar tol;
};
