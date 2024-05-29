// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <concepts>
#include <sstream>
#include <string>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <fmt/format.h>

template <typename T>
struct WithinAbs : Catch::Matchers::MatcherGenericBase {
  WithinAbs(T target, T margin) : target{target}, margin{margin} {}

  bool match(const T& matchee) const {
    using std::abs;
    return abs(target - matchee) <= margin;
  }

  std::string describe() const override {
    return fmt::format("\n==\n{}", target);
  }

 private:
  T target;
  T margin;
};

template <typename Derived>
  requires std::derived_from<Derived, Eigen::DenseBase<Derived>> ||
           std::derived_from<Derived, Eigen::SparseCompressedBase<Derived>>
struct MatrixWithinAbs : Catch::Matchers::MatcherGenericBase {
  MatrixWithinAbs(const Derived& target, typename Derived::Scalar margin)
      : target{target}, margin{margin} {}

  template <typename OtherDerived>
    requires std::derived_from<OtherDerived, Eigen::DenseBase<OtherDerived>> ||
             std::derived_from<OtherDerived,
                               Eigen::SparseCompressedBase<OtherDerived>>
  bool match(const OtherDerived& matchee) const {
    using std::abs;

    if (target.rows() != matchee.rows() || target.cols() != matchee.cols()) {
      return false;
    }

    for (Eigen::Index row = 0; row < target.rows(); ++row) {
      for (Eigen::Index col = 0; col < target.cols(); ++col) {
        if (abs(target.coeff(row, col) - matchee.coeff(row, col)) > margin) {
          return false;
        }
      }
    }

    return true;
  }

  std::string describe() const override {
    return (std::ostringstream{} << "\n==\n" << target).str();
  }

 private:
  const Derived& target;
  typename Derived::Scalar margin;
};
