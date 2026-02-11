// Copyright (c) Sleipnir contributors

#pragma once

#include <format>
#include <string>

#include <catch2/catch_tostring.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "coord.hpp"

namespace Catch {

template <>
struct StringMaker<slp::ExpressionType> {
  static std::string convert(const slp::ExpressionType& type) {
    using enum slp::ExpressionType;

    switch (type) {
      case NONE:
        return "NONE";
      case CONSTANT:
        return "CONSTANT";
      case LINEAR:
        return "LINEAR";
      case QUADRATIC:
        return "QUADRATIC";
      case NONLINEAR:
        return "NONLINEAR";
    }

    return "";
  }
};

template <>
struct StringMaker<slp::ExitStatus> {
  static std::string convert(const slp::ExitStatus& exit_status) {
    using enum slp::ExitStatus;

    switch (exit_status) {
      case SUCCESS:
        return "SUCCESS";
      case CALLBACK_REQUESTED_STOP:
        return "CALLBACK_REQUESTED_STOP";
      case TOO_FEW_DOFS:
        return "TOO_FEW_DOFS";
      case LOCALLY_INFEASIBLE:
        return "LOCALLY_INFEASIBLE";
      case GLOBALLY_INFEASIBLE:
        return "GLOBALLY_INFEASIBLE";
      case FACTORIZATION_FAILED:
        return "FACTORIZATION_FAILED";
      case FEASIBILITY_RESTORATION_FAILED:
        return "FEASIBILITY_RESTORATION_FAILED";
      case NONFINITE_INITIAL_COST_OR_CONSTRAINTS:
        return "NONFINITE_INITIAL_COST_OR_CONSTRAINTS";
      case DIVERGING_ITERATES:
        return "DIVERGING_ITERATES";
      case MAX_ITERATIONS_EXCEEDED:
        return "MAX_ITERATIONS_EXCEEDED";
      case TIMEOUT:
        return "TIMEOUT";
    }

    return "";
  }
};

template <>
struct StringMaker<Coord> {
  static std::string convert(const Coord& coord) {
    if (coord.sign == '+' || coord.sign == '-' || coord.sign == '0') {
      return std::format("{{{}, {}, '{}'}}", coord.row, coord.col, coord.sign);
    } else {
      return std::format("{{{}, {}, char({})}}", coord.row, coord.col,
                         static_cast<int>(coord.sign));
    }
  }
};

}  // namespace Catch
