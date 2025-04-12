// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("decision_variable - T init assign", "[decision_variable]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  // T zero init
  auto x = problem.decision_variable();
  CHECK(x.value() == T(0));

  // T assignment
  x.set_value(T(1));
  CHECK(x.value() == T(1));
  x.set_value(T(2));
  CHECK(x.value() == T(2));
}

TEMPLATE_TEST_CASE("decision_variable - Vector init assign",
                   "[decision_variable]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  // Vector zero init
  auto y = problem.decision_variable(2);
  CHECK(y.value(0) == T(0));
  CHECK(y.value(1) == T(0));

  // Vector assignment
  y[0].set_value(T(1));
  y[1].set_value(T(2));
  CHECK(y.value(0) == T(1));
  CHECK(y.value(1) == T(2));
  y[0].set_value(T(3));
  y[1].set_value(T(4));
  CHECK(y.value(0) == T(3));
  CHECK(y.value(1) == T(4));
}

TEMPLATE_TEST_CASE("decision_variable - Static matrix init assign",
                   "[decision_variable]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  // Matrix zero init
  auto z = problem.decision_variable(3, 2);
  CHECK(z.value(0, 0) == T(0));
  CHECK(z.value(0, 1) == T(0));
  CHECK(z.value(1, 0) == T(0));
  CHECK(z.value(1, 1) == T(0));
  CHECK(z.value(2, 0) == T(0));
  CHECK(z.value(2, 1) == T(0));

  // Matrix assignment; element comparison
  z.set_value(Eigen::Matrix<T, 3, 2>{{T(1), T(2)}, {T(3), T(4)}, {T(5), T(6)}});
  CHECK(z.value(0, 0) == T(1));
  CHECK(z.value(0, 1) == T(2));
  CHECK(z.value(1, 0) == T(3));
  CHECK(z.value(1, 1) == T(4));
  CHECK(z.value(2, 0) == T(5));
  CHECK(z.value(2, 1) == T(6));

  // Matrix assignment; matrix comparison
  {
    Eigen::Matrix<T, 3, 2> expected{
        {T(7), T(8)}, {T(9), T(10)}, {T(11), T(12)}};
    z.set_value(expected);
    CHECK(z.value() == expected);
  }

  // Block assignment
  {
    Eigen::Matrix<T, 2, 1> expected_block{{T(1)}, {T(1)}};
    z.block(0, 0, 2, 1).set_value(expected_block);

    Eigen::Matrix<T, 3, 2> expected_result{
        {T(1), T(8)}, {T(1), T(10)}, {T(11), T(12)}};
    CHECK(z.value() == expected_result);
  }

  // Segment assignment
  {
    Eigen::Matrix<T, 2, 1> expected_block{{T(1)}, {T(1)}};
    z.block(0, 0, 3, 1).segment(0, 2).set_value(expected_block);

    Eigen::Matrix<T, 3, 2> expected_result{
        {T(1), T(8)}, {T(1), T(10)}, {T(11), T(12)}};
    CHECK(z.value() == expected_result);
  }
}

TEMPLATE_TEST_CASE("decision_variable - Dynamic matrix init assign",
                   "[decision_variable]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  // Matrix zero init
  auto z = problem.decision_variable(3, 2);
  CHECK(z.value(0, 0) == T(0));
  CHECK(z.value(0, 1) == T(0));
  CHECK(z.value(1, 0) == T(0));
  CHECK(z.value(1, 1) == T(0));
  CHECK(z.value(2, 0) == T(0));
  CHECK(z.value(2, 1) == T(0));

  // Matrix assignment; element comparison
  {
    Eigen::Matrix<T, 3, 2> expected{{T(1), T(2)}, {T(3), T(4)}, {T(5), T(6)}};
    z.set_value(expected);
    CHECK(z.value(0, 0) == T(1));
    CHECK(z.value(0, 1) == T(2));
    CHECK(z.value(1, 0) == T(3));
    CHECK(z.value(1, 1) == T(4));
    CHECK(z.value(2, 0) == T(5));
    CHECK(z.value(2, 1) == T(6));
  }

  // Matrix assignment; matrix comparison
  {
    Eigen::Matrix<T, 3, 2> expected{
        {T(7), T(8)}, {T(9), T(10)}, {T(11), T(12)}};
    z.set_value(expected);
    CHECK(z.value() == expected);
  }

  // Block assignment
  {
    Eigen::Matrix<T, 2, 1> expected_block{{T(1)}, {T(1)}};
    z.block(0, 0, 2, 1).set_value(expected_block);

    Eigen::Matrix<T, 3, 2> expected_result{
        {T(1), T(8)}, {T(1), T(10)}, {T(11), T(12)}};
    CHECK(z.value() == expected_result);
  }

  // Segment assignment
  {
    Eigen::Matrix<T, 2, 1> expected_block{{T(1)}, {T(1)}};
    z.segment(0, 2).set_value(expected_block);

    Eigen::Matrix<T, 3, 2> expected_result{
        {T(1), T(8)}, {T(1), T(10)}, {T(11), T(12)}};
    CHECK(z.value() == expected_result);
  }
}

TEMPLATE_TEST_CASE("decision_variable - Symmetric matrix",
                   "[decision_variable]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  // Matrix zero init
  auto A = problem.symmetric_decision_variable(2);
  CHECK(A.value(0, 0) == T(0));
  CHECK(A.value(0, 1) == T(0));
  CHECK(A.value(1, 0) == T(0));
  CHECK(A.value(1, 1) == T(0));

  // Assign to lower triangle
  A(0, 0).set_value(T(1));
  A(1, 0).set_value(T(2));
  A(1, 1).set_value(T(3));

  // Confirm whole matrix changed
  CHECK(A.value(0, 0) == T(1));
  CHECK(A.value(0, 1) == T(2));
  CHECK(A.value(1, 0) == T(2));
  CHECK(A.value(1, 1) == T(3));
}
