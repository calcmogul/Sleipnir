// Copyright (c) Sleipnir contributors

#include <functional>
#include <iterator>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sleipnir/autodiff/variable_matrix.hpp>

TEST_CASE("VariableMatrix - Construct from Eigen::MatrixBase",
          "[VariableMatrix]") {
  slp::VariableMatrix mat{Eigen::MatrixXd{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};

  Eigen::MatrixXd expected{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  CHECK(mat.value() == expected);
}

TEST_CASE("VariableMatrix - Construct from Eigen::DiagonalBase",
          "[VariableMatrix]") {
  slp::VariableMatrix mat{Eigen::VectorXd{{1.0, 2.0, 3.0}}.asDiagonal()};

  Eigen::MatrixXd expected{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
  CHECK(mat.value() == expected);
}

TEST_CASE("VariableMatrix - Assignment to default", "[VariableMatrix]") {
  slp::VariableMatrix mat;

  CHECK(mat.rows() == 0);
  CHECK(mat.cols() == 0);

  mat = slp::VariableMatrix{2, 2};

  CHECK(mat.rows() == 2);
  CHECK(mat.cols() == 2);
  CHECK(mat[0, 0].value() == 0.0);
  CHECK(mat[0, 1].value() == 0.0);
  CHECK(mat[1, 0].value() == 0.0);
  CHECK(mat[1, 1].value() == 0.0);

  mat[0, 0] = 1.0;
  mat[0, 1] = 2.0;
  mat[1, 0] = 3.0;
  mat[1, 1] = 4.0;

  CHECK(mat[0, 0].value() == 1.0);
  CHECK(mat[0, 1].value() == 2.0);
  CHECK(mat[1, 0].value() == 3.0);
  CHECK(mat[1, 1].value() == 4.0);
}

TEST_CASE("VariableMatrix - Assignment aliasing", "[VariableMatrix]") {
  slp::VariableMatrix A{{1.0, 2.0}, {3.0, 4.0}};
  slp::VariableMatrix B{{5.0, 6.0}, {7.0, 8.0}};

  // A and B initially contain different values
  Eigen::MatrixXd expected_A{{1.0, 2.0}, {3.0, 4.0}};
  Eigen::MatrixXd expected_B{{5.0, 6.0}, {7.0, 8.0}};
  CHECK(A.value() == expected_A);
  CHECK(B.value() == expected_B);

  // Make A point to B's storage
  A = B;
  CHECK(A.value() == expected_B);
  CHECK(B.value() == expected_B);

  // Changes to B should be reflected in A
  B[0, 0].set_value(2.0);
  expected_B(0, 0) = 2.0;
  CHECK(A.value() == expected_B);
  CHECK(B.value() == expected_B);
}

TEST_CASE("VariableMatrix - Block() member function", "[VariableMatrix]") {
  slp::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // Block assignment
  A.block(1, 1, 2, 2) = Eigen::Matrix<double, 2, 2>{{10.0, 11.0}, {12.0, 13.0}};

  Eigen::Matrix<double, 3, 3> expected1{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 13.0}};
  CHECK(A.value() == expected1);

  // Block-of-block assignment
  A.block(1, 1, 2, 2).block(1, 1, 1, 1) = 14.0;

  Eigen::Matrix<double, 3, 3> expected2{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 14.0}};
  CHECK(A.value() == expected2);
}

TEST_CASE("VariableMatrix - Slicing", "[VariableMatrix]") {
  using namespace slp::slicing;

  slp::VariableMatrix mat{
      {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  CHECK(mat.rows() == 4);
  CHECK(mat.cols() == 4);

  // Single-arg index operator on full matrix
  for (int i = 0; i < mat.rows() * mat.cols(); ++i) {
    CHECK(mat[i] == i + 1);
  }

  // Slice from start
  {
    auto s = mat[slp::Slice{1, _}, slp::Slice{2, _}];
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 2);
    // Single-arg index operator on forward slice
    CHECK(s[0] == 7.0);
    CHECK(s[1] == 8.0);
    CHECK(s[2] == 11.0);
    CHECK(s[3] == 12.0);
    CHECK(s[4] == 15.0);
    CHECK(s[5] == 16.0);
    // Double-arg index operator on forward slice
    CHECK(s[0, 0] == 7.0);
    CHECK(s[0, 1] == 8.0);
    CHECK(s[1, 0] == 11.0);
    CHECK(s[1, 1] == 12.0);
    CHECK(s[2, 0] == 15.0);
    CHECK(s[2, 1] == 16.0);
  }

  // Slice from end
  {
    auto s = mat[slp::Slice{-1, _}, slp::Slice{-2, _}];
    CHECK(s.rows() == 1);
    CHECK(s.cols() == 2);
    // Single-arg index operator on reverse slice
    CHECK(s[0] == 15.0);
    CHECK(s[1] == 16.0);
    // Double-arg index operator on reverse slice
    CHECK(s[0, 0] == 15.0);
    CHECK(s[0, 1] == 16.0);
  }

  // Slice from start with step of 2
  {
    auto s = mat[_, slp::Slice{_, _, 2}];
    CHECK(s.rows() == 4);
    CHECK(s.cols() == 2);
    CHECK(s.value() ==
          Eigen::MatrixXd{{1.0, 3.0}, {5.0, 7.0}, {9.0, 11.0}, {13.0, 15.0}});
  }

  // Slice from end with negative step for row and column
  {
    auto s = mat[slp::Slice{_, _, -1}, slp::Slice{_, _, -2}];
    CHECK(s.rows() == 4);
    CHECK(s.cols() == 2);
    CHECK(s.value() ==
          Eigen::MatrixXd{{16.0, 14.0}, {12.0, 10.0}, {8.0, 6.0}, {4.0, 2.0}});
  }

  // Slice from start and column -1
  {
    auto s = mat[slp::Slice{1, _}, -1];
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 1);
    CHECK(s.value() == Eigen::MatrixXd{{8.0}, {12.0}, {16.0}});
  }

  // Slice from start and column -2
  {
    auto s = mat[slp::Slice{1, _}, -2];
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 1);
    CHECK(s.value() == Eigen::MatrixXd{{7.0}, {11.0}, {15.0}});
  }

  // Block assignment
  {
    auto s = mat[slp::Slice{_, _, 2}, slp::Slice{_, _, 2}];
    CHECK(s.rows() == 2);
    CHECK(s.cols() == 2);
    s = Eigen::MatrixXd{{17.0, 18.0}, {19.0, 20.0}};
    CHECK(mat.value() == Eigen::MatrixXd{{17.0, 2.0, 18.0, 4.0},
                                         {5.0, 6.0, 7.0, 8.0},
                                         {19.0, 10.0, 20.0, 12.0},
                                         {13.0, 14.0, 15.0, 16.0}});
  }
}

TEST_CASE("VariableMatrix - Subslicing", "[VariableMatrix]") {
  using namespace slp::slicing;

  // Block-of-block assignment (row skip forward)
  {
    slp::VariableMatrix mat{5, 5};
    auto s = mat[slp::Slice{_, _, 2}, slp::Slice{_, _, 1}]
                [slp::Slice{1, 3}, slp::Slice{1, 4}];
    CHECK(s.rows() == 2);
    CHECK(s.cols() == 3);
    s = Eigen::MatrixXd{{1, 2, 3}, {4, 5, 6}};

    CHECK(mat.value() == Eigen::MatrixXd{{0, 0, 0, 0, 0},
                                         {0, 0, 0, 0, 0},
                                         {0, 1, 2, 3, 0},
                                         {0, 0, 0, 0, 0},
                                         {0, 4, 5, 6, 0}});
  }

  // Block-of-block assignment (row skip backward)
  {
    slp::VariableMatrix mat{5, 5};
    auto s = mat[slp::Slice{_, _, -2}, slp::Slice{_, _, -1}]
                [slp::Slice{1, 3}, slp::Slice{1, 4}];
    CHECK(s.rows() == 2);
    CHECK(s.cols() == 3);
    s = Eigen::MatrixXd{{1, 2, 3}, {4, 5, 6}};

    CHECK(mat.value() == Eigen::MatrixXd{{0, 6, 5, 4, 0},
                                         {0, 0, 0, 0, 0},
                                         {0, 3, 2, 1, 0},
                                         {0, 0, 0, 0, 0},
                                         {0, 0, 0, 0, 0}});
  }

  // Block-of-block assignment (column skip forward)
  {
    slp::VariableMatrix mat{5, 5};
    auto s = mat[slp::Slice{_, _, 1}, slp::Slice{_, _, 2}]
                [slp::Slice{1, 4}, slp::Slice{1, 3}];
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 2);
    s = Eigen::MatrixXd{{1, 2}, {3, 4}, {5, 6}};

    CHECK(mat.value() == Eigen::MatrixXd{{0, 0, 0, 0, 0},
                                         {0, 0, 1, 0, 2},
                                         {0, 0, 3, 0, 4},
                                         {0, 0, 5, 0, 6},
                                         {0, 0, 0, 0, 0}});
  }

  // Block-of-block assignment (column skip backward)
  {
    slp::VariableMatrix mat{5, 5};
    auto s = mat[slp::Slice{_, _, -1}, slp::Slice{_, _, -2}]
                [slp::Slice{1, 4}, slp::Slice{1, 3}];
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 2);
    s = Eigen::MatrixXd{{1, 2}, {3, 4}, {5, 6}};

    CHECK(mat.value() == Eigen::MatrixXd{{0, 0, 0, 0, 0},
                                         {6, 0, 5, 0, 0},
                                         {4, 0, 3, 0, 0},
                                         {2, 0, 1, 0, 0},
                                         {0, 0, 0, 0, 0}});
  }
}

TEST_CASE("VariableMatrix - Iterators", "[VariableMatrix]") {
  slp::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  auto sub_A = A.block(2, 1, 1, 2);

  CHECK(std::distance(A.begin(), A.end()) == 9);
  CHECK(std::distance(A.cbegin(), A.cend()) == 9);
  CHECK(std::distance(A.rbegin(), A.rend()) == 9);
  CHECK(std::distance(A.crbegin(), A.crend()) == 9);
  CHECK(std::distance(sub_A.begin(), sub_A.end()) == 2);
  CHECK(std::distance(sub_A.cbegin(), sub_A.cend()) == 2);
  CHECK(std::distance(sub_A.rbegin(), sub_A.rend()) == 2);
  CHECK(std::distance(sub_A.crbegin(), sub_A.crend()) == 2);

  int i = 1;
  for (auto& elem : A) {
    CHECK(elem.value() == i);
    ++i;
  }

  i = 9;
  for (auto& elem : A | std::views::reverse) {
    CHECK(elem.value() == i);
    --i;
  }

  i = 8;
  for (auto& elem : sub_A) {
    CHECK(elem.value() == i);
    ++i;
  }

  i = 9;
  for (auto& elem : sub_A | std::views::reverse) {
    CHECK(elem.value() == i);
    --i;
  }
}

TEST_CASE("VariableMatrix - Value", "[VariableMatrix]") {
  using namespace slp::slicing;

  slp::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  Eigen::Matrix<double, 3, 3> expected{
      {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // Full matrix
  CHECK(A.value() == expected);
  CHECK(A.value(3) == 4.0);
  CHECK(A.T().value(3) == 2.0);

  // Block
  CHECK(A.block(1, 1, 2, 2).value() == expected.block(1, 1, 2, 2));
  CHECK(A.block(1, 1, 2, 2).value(2) == 8.0);
  CHECK(A.T().block(1, 1, 2, 2).value(2) == 6.0);

  // Slice
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}].value() ==
        expected.block(1, 1, 2, 2));
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}].value(2) == 8.0);
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}].T().value(2) == 6.0);

  // Block-of-block
  CHECK(A.block(1, 1, 2, 2).block(0, 1, 2, 1).value() ==
        expected.block(1, 1, 2, 2).block(0, 1, 2, 1));
  CHECK(A.block(1, 1, 2, 2).block(0, 1, 2, 1).value(1) == 9.0);
  CHECK(A.block(1, 1, 2, 2).T().block(0, 1, 2, 1).value(1) == 9.0);

  // Slice-of-slice
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}][_, slp::Slice{1, _}].value() ==
        expected.block(1, 1, 2, 2).block(0, 1, 2, 1));
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}][_, slp::Slice{1, _}].value(1) ==
        9.0);
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}].T()[_, slp::Slice{1, _}].value(
            1) == 9.0);
}

TEST_CASE("VariableMatrix - cwise_transform()", "[VariableMatrix]") {
  // VariableMatrix cwise_transform
  slp::VariableMatrix A{{-2.0, -3.0, -4.0}, {-5.0, -6.0, -7.0}};

  slp::VariableMatrix result1 = A.cwise_transform(slp::abs);
  Eigen::Matrix<double, 2, 3> expected1{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};

  // Don't modify original matrix
  CHECK(A.value() == -expected1);

  CHECK(result1.value() == expected1);

  // VariableBlock cwise_transform
  auto sub_A = A.block(0, 0, 2, 2);

  slp::VariableMatrix result2 = sub_A.cwise_transform(slp::abs);
  Eigen::Matrix<double, 2, 2> expected2{{2.0, 3.0}, {5.0, 6.0}};

  // Don't modify original matrix
  CHECK(A.value() == -expected1);
  CHECK(sub_A.value() == -expected2);

  CHECK(result2.value() == expected2);
}

TEST_CASE("VariableMatrix - Zero() static function", "[VariableMatrix]") {
  auto A = slp::VariableMatrix::zero(2, 3);

  for (auto& elem : A) {
    CHECK(elem.value() == 0.0);
  }
}

TEST_CASE("VariableMatrix - Ones() static function", "[VariableMatrix]") {
  auto A = slp::VariableMatrix::ones(2, 3);

  for (auto& elem : A) {
    CHECK(elem.value() == 1.0);
  }
}

TEST_CASE("VariableMatrix - cwise_reduce()", "[VariableMatrix]") {
  slp::VariableMatrix A{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};
  slp::VariableMatrix B{{8.0, 9.0, 10.0}, {11.0, 12.0, 13.0}};
  slp::VariableMatrix result = slp::cwise_reduce(A, B, std::multiplies<>{});

  Eigen::Matrix<double, 2, 3> expected{{16.0, 27.0, 40.0}, {55.0, 72.0, 91.0}};
  CHECK(result.value() == expected);
}

TEST_CASE("VariableMatrix - Block() free function", "[VariableMatrix]") {
  slp::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  slp::VariableMatrix B{{7.0}, {8.0}};

  slp::VariableMatrix mat1 = slp::block({{A, B}});
  Eigen::Matrix<double, 2, 4> expected1{{1.0, 2.0, 3.0, 7.0},
                                        {4.0, 5.0, 6.0, 8.0}};
  CHECK(mat1.rows() == 2);
  CHECK(mat1.cols() == 4);
  CHECK(mat1.value() == expected1);

  slp::VariableMatrix C{{9.0, 10.0, 11.0, 12.0}};

  slp::VariableMatrix mat2 = slp::block({{A, B}, {C}});
  Eigen::Matrix<double, 3, 4> expected2{
      {1.0, 2.0, 3.0, 7.0}, {4.0, 5.0, 6.0, 8.0}, {9.0, 10.0, 11.0, 12.0}};
  CHECK(mat2.rows() == 3);
  CHECK(mat2.cols() == 4);
  CHECK(mat2.value() == expected2);
}

void check_solve(slp::VariableMatrix A, slp::VariableMatrix B) {
  INFO(fmt::format("Solve {}x{}", A.rows(), A.cols()));

  auto X = slp::solve(A, B);

  CHECK(X.rows() == A.cols());
  CHECK(X.cols() == B.cols());
  CHECK((A.value() * X.value() - B.value()).norm() < 1e-12);
}

TEST_CASE("VariableMatrix - Solve() free function", "[VariableMatrix]") {
  // 1x1 special case
  check_solve(slp::VariableMatrix{{2.0}}, slp::VariableMatrix{{5.0}});

  // 2x2 special case
  check_solve(slp::VariableMatrix{{1.0, 2.0}, {3.0, 4.0}},
              slp::VariableMatrix{{5.0}, {6.0}});

  // 3x3 special case
  check_solve(
      slp::VariableMatrix{{1.0, 2.0, 3.0}, {-4.0, -5.0, 6.0}, {7.0, 8.0, 9.0}},
      slp::VariableMatrix{{10.0}, {11.0}, {12.0}});

  // 4x4 special case
  check_solve(slp::VariableMatrix{{1.0, 2.0, 3.0, -4.0},
                                  {-5.0, 6.0, 7.0, 8.0},
                                  {9.0, 10.0, 11.0, 12.0},
                                  {13.0, 14.0, 15.0, 16.0}},
              slp::VariableMatrix{{17.0}, {18.0}, {19.0}, {20.0}});

  // 5x5 general case
  check_solve(slp::VariableMatrix{{1.0, 2.0, 3.0, -4.0, 5.0},
                                  {-5.0, 6.0, 7.0, 8.0, 9.0},
                                  {9.0, 10.0, 11.0, 12.0, 13.0},
                                  {13.0, 14.0, 15.0, 16.0, 17.0},
                                  {17.0, 18.0, 19.0, 20.0, 21.0}},
              slp::VariableMatrix{{21.0}, {22.0}, {23.0}, {24.0}, {25.0}});
}
