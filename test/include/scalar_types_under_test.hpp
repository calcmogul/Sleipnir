// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <boost/decimal.hpp>
#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/optimization/problem.hpp>

namespace Eigen {

/**
 * NumTraits specialization that allows instantiating Eigen types with
 * boost::decimal64_t.
 */
template <>
struct NumTraits<boost::decimal::decimal64_t>
    : GenericNumTraits<boost::decimal::decimal64_t> {
  /// Is complex.
  static constexpr int IsComplex = 0;
  /// Is integer.
  static constexpr int IsInteger = 0;
  /// Is signed.
  static constexpr int IsSigned = 1;
  /// Require initialization.
  static constexpr int RequireInitialization = 1;
  /// Read cost.
  static constexpr int ReadCost = 1;
  /// Add cost.
  static constexpr int AddCost = 3;
  /// Multiply cost.
  static constexpr int MulCost = 3;
};

}  // namespace Eigen

extern template class slp::OCP<boost::decimal::decimal64_t>;
extern template class slp::Problem<boost::decimal::decimal64_t>;

#define SCALAR_TYPES_UNDER_TEST double, boost::decimal::decimal64_t
