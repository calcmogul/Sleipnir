// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"

TEST_CASE("decision_variable - Scalar init assign", "[decision_variable]") {
  slp::Problem problem;

  // Scalar zero init
  auto x = problem.decision_variable();
  CHECK(x.value() == 0.0);

  // Scalar assignment
  x.set_value(1.0);
  CHECK(x.value() == 1.0);
  x.set_value(2.0);
  CHECK(x.value() == 2.0);
}

TEST_CASE("decision_variable - Vector init assign", "[decision_variable]") {
  slp::Problem problem;

  // Vector zero init
  auto y = problem.decision_variable(2);
  CHECK(y.value(0) == 0.0);
  CHECK(y.value(1) == 0.0);

  // Vector assignment
  y[0].set_value(1.0);
  y[1].set_value(2.0);
  CHECK(y.value(0) == 1.0);
  CHECK(y.value(1) == 2.0);
  y[0].set_value(3.0);
  y[1].set_value(4.0);
  CHECK(y.value(0) == 3.0);
  CHECK(y.value(1) == 4.0);
}

TEST_CASE("decision_variable - Static matrix init assign",
          "[decision_variable]") {
  slp::Problem problem;

  // Matrix zero init
  auto z = problem.decision_variable(3, 2);
  CHECK(z.value(0, 0) == 0.0);
  CHECK(z.value(0, 1) == 0.0);
  CHECK(z.value(1, 0) == 0.0);
  CHECK(z.value(1, 1) == 0.0);
  CHECK(z.value(2, 0) == 0.0);
  CHECK(z.value(2, 1) == 0.0);

  // Matrix assignment; element comparison
  z.set_value(Eigen::Matrix<double, 3, 2>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  CHECK(z.value(0, 0) == 1.0);
  CHECK(z.value(0, 1) == 2.0);
  CHECK(z.value(1, 0) == 3.0);
  CHECK(z.value(1, 1) == 4.0);
  CHECK(z.value(2, 0) == 5.0);
  CHECK(z.value(2, 1) == 6.0);

  // Matrix assignment; matrix comparison
  {
    Eigen::Matrix<double, 3, 2> expected{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    z.set_value(expected);
    CHECK(z.value() == expected);
  }

  // Block assignment
  {
    Eigen::Matrix<double, 2, 1> expected_block{{1.0}, {1.0}};
    z.block(0, 0, 2, 1).set_value(expected_block);

    Eigen::Matrix<double, 3, 2> expected_result{
        {1.0, 8.0}, {1.0, 10.0}, {11.0, 12.0}};
    CHECK(z.value() == expected_result);
  }

  // Segment assignment
  {
    Eigen::Matrix<double, 2, 1> expected_block{{1.0}, {1.0}};
    z.block(0, 0, 3, 1).segment(0, 2).set_value(expected_block);

    Eigen::Matrix<double, 3, 2> expected_result{
        {1.0, 8.0}, {1.0, 10.0}, {11.0, 12.0}};
    CHECK(z.value() == expected_result);
  }
}

TEST_CASE("decision_variable - Dynamic matrix init assign",
          "[decision_variable]") {
  slp::Problem problem;

  // Matrix zero init
  auto z = problem.decision_variable(3, 2);
  CHECK(z.value(0, 0) == 0.0);
  CHECK(z.value(0, 1) == 0.0);
  CHECK(z.value(1, 0) == 0.0);
  CHECK(z.value(1, 1) == 0.0);
  CHECK(z.value(2, 0) == 0.0);
  CHECK(z.value(2, 1) == 0.0);

  // Matrix assignment; element comparison
  {
    Eigen::MatrixXd expected{3, 2};
    expected << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    z.set_value(expected);
    CHECK(z.value(0, 0) == 1.0);
    CHECK(z.value(0, 1) == 2.0);
    CHECK(z.value(1, 0) == 3.0);
    CHECK(z.value(1, 1) == 4.0);
    CHECK(z.value(2, 0) == 5.0);
    CHECK(z.value(2, 1) == 6.0);
  }

  // Matrix assignment; matrix comparison
  {
    Eigen::MatrixXd expected{3, 2};
    expected << 7.0, 8.0, 9.0, 10.0, 11.0, 12.0;
    z.set_value(expected);
    CHECK(z.value() == expected);
  }

  // Block assignment
  {
    Eigen::MatrixXd expected_block{2, 1};
    expected_block << 1.0, 1.0;
    z.block(0, 0, 2, 1).set_value(expected_block);

    Eigen::MatrixXd expected_result{3, 2};
    expected_result << 1.0, 8.0, 1.0, 10.0, 11.0, 12.0;
    CHECK(z.value() == expected_result);
  }

  // Segment assignment
  {
    Eigen::MatrixXd expected_block{2, 1};
    expected_block << 1.0, 1.0;
    z.segment(0, 2).set_value(expected_block);

    Eigen::MatrixXd expected_result{3, 2};
    expected_result << 1.0, 8.0, 1.0, 10.0, 11.0, 12.0;
    CHECK(z.value() == expected_result);
  }
}

TEST_CASE("decision_variable - Symmetric matrix", "[decision_variable]") {
  slp::Problem problem;

  // Matrix zero init
  auto A = problem.symmetric_decision_variable(2);
  CHECK(A.value(0, 0) == 0.0);
  CHECK(A.value(0, 1) == 0.0);
  CHECK(A.value(1, 0) == 0.0);
  CHECK(A.value(1, 1) == 0.0);

  // Assign to lower triangle
  A[0, 0].set_value(1.0);
  A[1, 0].set_value(2.0);
  A[1, 1].set_value(3.0);

  // Confirm whole matrix changed
  CHECK(A.value(0, 0) == 1.0);
  CHECK(A.value(0, 1) == 2.0);
  CHECK(A.value(1, 0) == 2.0);
  CHECK(A.value(1, 1) == 3.0);
}
