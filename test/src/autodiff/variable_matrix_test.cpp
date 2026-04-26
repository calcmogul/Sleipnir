// Copyright (c) Sleipnir contributors

#include <cmath>
#include <format>
#include <functional>
#include <iterator>
#include <ranges>

#include <Eigen/Core>
#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/slice.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "catch_matchers.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("VariableMatrix - Construct from Eigen::MatrixBase",
                   "[VariableMatrix]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::VariableMatrix mat{
      Eigen::Matrix<T, 2, 3>{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}}};

  Eigen::Matrix<T, 2, 3> expected{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}};
  CHECK(mat.value() == expected);
}

TEMPLATE_TEST_CASE("VariableMatrix - Construct from Eigen::DiagonalBase",
                   "[VariableMatrix]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::VariableMatrix mat{Eigen::Vector<T, 3>{{T(1), T(2), T(3)}}.asDiagonal()};

  Eigen::Matrix<T, 3, 3> expected{
      {T(1), T(0), T(0)}, {T(0), T(2), T(0)}, {T(0), T(0), T(3)}};
  CHECK(mat.value() == expected);
}

TEMPLATE_TEST_CASE("VariableMatrix - Assignment to default", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::VariableMatrix<T> mat;

  CHECK(mat.rows() == 0);
  CHECK(mat.cols() == 0);

  mat = slp::VariableMatrix<T>{2, 2};

  CHECK(mat.rows() == 2);
  CHECK(mat.cols() == 2);
  CHECK(mat[0, 0].value() == T(0));
  CHECK(mat[0, 1].value() == T(0));
  CHECK(mat[1, 0].value() == T(0));
  CHECK(mat[1, 1].value() == T(0));

  mat[0, 0] = T(1);
  mat[0, 1] = T(2);
  mat[1, 0] = T(3);
  mat[1, 1] = T(4);

  CHECK(mat[0, 0].value() == T(1));
  CHECK(mat[0, 1].value() == T(2));
  CHECK(mat[1, 0].value() == T(3));
  CHECK(mat[1, 1].value() == T(4));
}

TEMPLATE_TEST_CASE("VariableMatrix - Assignment aliasing", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::VariableMatrix<T> A{{1, 2}, {3, 4}};
  slp::VariableMatrix<T> B{{5, 6}, {7, 8}};

  // A and B initially contain different values
  Eigen::Matrix<T, 2, 2> expected_A{{T(1), T(2)}, {T(3), T(4)}};
  Eigen::Matrix<T, 2, 2> expected_B{{T(5), T(6)}, {T(7), T(8)}};
  CHECK(A.value() == expected_A);
  CHECK(B.value() == expected_B);

  // Make A point to B's storage
  A = B;
  CHECK(A.value() == expected_B);
  CHECK(B.value() == expected_B);

  // Changes to B should be reflected in A
  B[0, 0].set_value(T(2));
  expected_B(0, 0) = T(2);
  CHECK(A.value() == expected_B);
  CHECK(B.value() == expected_B);
}

TEMPLATE_TEST_CASE("VariableMatrix - block() member function",
                   "[VariableMatrix]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::VariableMatrix<T> A{
      {T(1), T(2), T(3)}, {T(4), T(5), T(6)}, {T(7), T(8), T(9)}};

  // Block assignment
  A.block(1, 1, 2, 2) = Eigen::Matrix<T, 2, 2>{{T(10), T(11)}, {T(12), T(13)}};

  Eigen::Matrix<T, 3, 3> expected1{
      {T(1), T(2), T(3)}, {T(4), T(10), T(11)}, {T(7), T(12), T(13)}};
  CHECK(A.value() == expected1);

  // Block-of-block assignment
  A.block(1, 1, 2, 2).block(1, 1, 1, 1) = T(14);

  Eigen::Matrix<T, 3, 3> expected2{
      {T(1), T(2), T(3)}, {T(4), T(10), T(11)}, {T(7), T(12), T(14)}};
  CHECK(A.value() == expected2);
}

TEMPLATE_TEST_CASE("VariableMatrix - Slicing", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using namespace slp::slicing;

  slp::VariableMatrix<T> mat{{T(1), T(2), T(3), T(4)},
                             {T(5), T(6), T(7), T(8)},
                             {T(9), T(10), T(11), T(12)},
                             {T(13), T(14), T(15), T(16)}};
  CHECK(mat.rows() == 4);
  CHECK(mat.cols() == 4);

  // Single-arg index operator on full matrix
  for (int i = 0; i < mat.rows() * mat.cols(); ++i) {
    CHECK(mat[i] == T(i + 1));
  }

  // Slice from start
  {
    auto s = mat[slp::Slice{1, _}, slp::Slice{2, _}];
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 2);
    // Single-arg index operator on forward slice
    CHECK(s[0] == T(7));
    CHECK(s[1] == T(8));
    CHECK(s[2] == T(11));
    CHECK(s[3] == T(12));
    CHECK(s[4] == T(15));
    CHECK(s[5] == T(16));
    // Double-arg index operator on forward slice
    CHECK(s[0, 0] == T(7));
    CHECK(s[0, 1] == T(8));
    CHECK(s[1, 0] == T(11));
    CHECK(s[1, 1] == T(12));
    CHECK(s[2, 0] == T(15));
    CHECK(s[2, 1] == T(16));
  }

  // Slice from end
  {
    auto s = mat[slp::Slice{-1, _}, slp::Slice{-2, _}];
    CHECK(s.rows() == 1);
    CHECK(s.cols() == 2);
    // Single-arg index operator on reverse slice
    CHECK(s[0] == T(15));
    CHECK(s[1] == T(16));
    // Double-arg index operator on reverse slice
    CHECK(s[0, 0] == T(15));
    CHECK(s[0, 1] == T(16));
  }

  // Slice from start with step of 2
  {
    auto s = mat[_, slp::Slice{_, _, 2}];
    CHECK(s.rows() == 4);
    CHECK(s.cols() == 2);
    CHECK(s.value() ==
          Eigen::Matrix<T, 4, 2>{
              {T(1), T(3)}, {T(5), T(7)}, {T(9), T(11)}, {T(13), T(15)}});
  }

  // Slice from end with negative step for row and column
  {
    auto s = mat[slp::Slice{_, _, -1}, slp::Slice{_, _, -2}];
    CHECK(s.rows() == 4);
    CHECK(s.cols() == 2);
    CHECK(s.value() ==
          Eigen::Matrix<T, 4, 2>{
              {T(16), T(14)}, {T(12), T(10)}, {T(8), T(6)}, {T(4), T(2)}});
  }

  // Slice from start and column -1
  {
    auto s = mat[slp::Slice{1, _}, -1];
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 1);
    CHECK(s.value() == Eigen::Matrix<T, 3, 1>{{T(8)}, {T(12)}, {T(16)}});
  }

  // Slice from start and column -2
  {
    auto s = mat[slp::Slice{1, _}, -2];
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 1);
    CHECK(s.value() == Eigen::Matrix<T, 3, 1>{{T(7)}, {T(11)}, {T(15)}});
  }

  // Block assignment
  {
    auto s = mat[slp::Slice{_, _, 2}, slp::Slice{_, _, 2}];
    CHECK(s.rows() == 2);
    CHECK(s.cols() == 2);
    s = Eigen::Matrix<T, 2, 2>{{T(17), T(18)}, {T(19), T(20)}};
    CHECK(mat.value() == Eigen::Matrix<T, 4, 4>{{T(17), T(2), T(18), T(4)},
                                                {T(5), T(6), T(7), T(8)},
                                                {T(19), T(10), T(20), T(12)},
                                                {T(13), T(14), T(15), T(16)}});
  }
}

TEMPLATE_TEST_CASE("VariableMatrix - Subslicing", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using namespace slp::slicing;

  // Block-of-block assignment (row skip forward)
  {
    slp::VariableMatrix<T> mat{5, 5};
    auto s = mat[slp::Slice{_, _, 2}, slp::Slice{_, _, 1}]
                [slp::Slice{1, 3}, slp::Slice{1, 4}];
    CHECK(s.rows() == 2);
    CHECK(s.cols() == 3);
    s = Eigen::Matrix<T, 2, 3>{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}};

    CHECK(mat.value() ==
          Eigen::Matrix<T, 5, 5>{{T(0), T(0), T(0), T(0), T(0)},
                                 {T(0), T(0), T(0), T(0), T(0)},
                                 {T(0), T(1), T(2), T(3), T(0)},
                                 {T(0), T(0), T(0), T(0), T(0)},
                                 {T(0), T(4), T(5), T(6), T(0)}});
  }

  // Block-of-block assignment (row skip backward)
  {
    slp::VariableMatrix<T> mat{5, 5};
    auto s = mat[slp::Slice{_, _, -2}, slp::Slice{_, _, -1}]
                [slp::Slice{1, 3}, slp::Slice{1, 4}];
    CHECK(s.rows() == 2);
    CHECK(s.cols() == 3);
    s = Eigen::Matrix<T, 2, 3>{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}};

    CHECK(mat.value() ==
          Eigen::Matrix<T, 5, 5>{{T(0), T(6), T(5), T(4), T(0)},
                                 {T(0), T(0), T(0), T(0), T(0)},
                                 {T(0), T(3), T(2), T(1), T(0)},
                                 {T(0), T(0), T(0), T(0), T(0)},
                                 {T(0), T(0), T(0), T(0), T(0)}});
  }

  // Block-of-block assignment (column skip forward)
  {
    slp::VariableMatrix<T> mat{5, 5};
    auto s = mat[slp::Slice{_, _, 1}, slp::Slice{_, _, 2}]
                [slp::Slice{1, 4}, slp::Slice{1, 3}];
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 2);
    s = Eigen::Matrix<T, 3, 2>{{T(1), T(2)}, {T(3), T(4)}, {T(5), T(6)}};

    CHECK(mat.value() ==
          Eigen::Matrix<T, 5, 5>{{T(0), T(0), T(0), T(0), T(0)},
                                 {T(0), T(0), T(1), T(0), T(2)},
                                 {T(0), T(0), T(3), T(0), T(4)},
                                 {T(0), T(0), T(5), T(0), T(6)},
                                 {T(0), T(0), T(0), T(0), T(0)}});
  }

  // Block-of-block assignment (column skip backward)
  {
    slp::VariableMatrix<T> mat{5, 5};
    auto s = mat[slp::Slice{_, _, -1}, slp::Slice{_, _, -2}]
                [slp::Slice{1, 4}, slp::Slice{1, 3}];
    CHECK(s.rows() == 3);
    CHECK(s.cols() == 2);
    s = Eigen::Matrix<T, 3, 2>{{T(1), T(2)}, {T(3), T(4)}, {T(5), T(6)}};

    CHECK(mat.value() ==
          Eigen::Matrix<T, 5, 5>{{T(0), T(0), T(0), T(0), T(0)},
                                 {T(6), T(0), T(5), T(0), T(0)},
                                 {T(4), T(0), T(3), T(0), T(0)},
                                 {T(2), T(0), T(1), T(0), T(0)},
                                 {T(0), T(0), T(0), T(0), T(0)}});
  }
}

TEMPLATE_TEST_CASE("VariableMatrix - Compound assignment operators",
                   "[VariableMatrix]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::VariableMatrix<T> A1{{T(1)}};
  slp::VariableMatrix<T> A2{{T(1), T(2)}, {T(3), T(4)}};
  const slp::VariableMatrix<T> B{{T(1), T(2)}, {T(3), T(4)}};
  const T b{2};

  // VariableMatrix matrix-matrix operators
  A2 += B;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(2), T(4)}, {T(6), T(8)}});
  A2 -= B;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(1), T(2)}, {T(3), T(4)}});
  A2 *= B;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(7), T(10)}, {T(15), T(22)}});
  A2.set_value(Eigen::Matrix<T, 2, 2>{{T(1), T(2)}, {T(3), T(4)}});

  // VariableBlock matrix-matrix operators
  A2.block(0, 0, A2.rows(), A2.cols()) += B;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(2), T(4)}, {T(6), T(8)}});
  A2.block(0, 0, A2.rows(), A2.cols()) -= B;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(1), T(2)}, {T(3), T(4)}});
  A2.block(0, 0, A2.rows(), A2.cols()) *= B;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(7), T(10)}, {T(15), T(22)}});
  A2.set_value(Eigen::Matrix<T, 2, 2>{{T(1), T(2)}, {T(3), T(4)}});

  // VariableMatrix matrix-scalar operators
  A1 += b;
  CHECK(A1.value() == Eigen::Matrix<T, 1, 1>{{T(3)}});
  A1 -= b;
  CHECK(A1.value() == Eigen::Matrix<T, 1, 1>{{T(1)}});
  A2 *= b;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(2), T(4)}, {T(6), T(8)}});
  A2 /= b;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(1), T(2)}, {T(3), T(4)}});

  // VariableBlock matrix-scalar operators
  A2.block(0, 0, 1, 1) += b;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(3), T(2)}, {T(3), T(4)}});
  A2.block(0, 0, 1, 1) -= b;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(1), T(2)}, {T(3), T(4)}});
  A2.block(0, 0, A2.rows(), A2.cols()) *= b;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(2), T(4)}, {T(6), T(8)}});
  A2.block(0, 0, A2.rows(), A2.cols()) /= b;
  CHECK(A2.value() == Eigen::Matrix<T, 2, 2>{{T(1), T(2)}, {T(3), T(4)}});
}

TEMPLATE_TEST_CASE("VariableMatrix - Iterators", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::VariableMatrix<T> A{
      {T(1), T(2), T(3)}, {T(4), T(5), T(6)}, {T(7), T(8), T(9)}};
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
    CHECK(elem.value() == T(i));
    ++i;
  }

  i = 9;
  for (auto& elem : A | std::views::reverse) {
    CHECK(elem.value() == T(i));
    --i;
  }

  i = 8;
  for (auto& elem : sub_A) {
    CHECK(elem.value() == T(i));
    ++i;
  }

  i = 9;
  for (auto& elem : sub_A | std::views::reverse) {
    CHECK(elem.value() == T(i));
    --i;
  }
}

TEMPLATE_TEST_CASE("VariableMatrix - Value", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using namespace slp::slicing;

  slp::VariableMatrix<T> A{
      {T(1), T(2), T(3)}, {T(4), T(5), T(6)}, {T(7), T(8), T(9)}};
  Eigen::Matrix<T, 3, 3> expected{
      {T(1), T(2), T(3)}, {T(4), T(5), T(6)}, {T(7), T(8), T(9)}};

  // Full matrix
  CHECK(A.value() == expected);
  CHECK(A.value(3) == T(4));
  CHECK(A.T().value(3) == T(2));

  // Block
  CHECK(A.block(1, 1, 2, 2).value() == expected.block(1, 1, 2, 2));
  CHECK(A.block(1, 1, 2, 2).value(2) == T(8));
  CHECK(A.T().block(1, 1, 2, 2).value(2) == T(6));

  // Slice
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}].value() ==
        expected.block(1, 1, 2, 2));
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}].value(2) == T(8));
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}].T().value(2) == T(6));

  // Block-of-block
  CHECK(A.block(1, 1, 2, 2).block(0, 1, 2, 1).value() ==
        expected.block(1, 1, 2, 2).block(0, 1, 2, 1));
  CHECK(A.block(1, 1, 2, 2).block(0, 1, 2, 1).value(1) == T(9));
  CHECK(A.block(1, 1, 2, 2).T().block(0, 1, 2, 1).value(1) == T(9));

  // Slice-of-slice
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}][_, slp::Slice{1, _}].value() ==
        expected.block(1, 1, 2, 2).block(0, 1, 2, 1));
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}][_, slp::Slice{1, _}].value(1) ==
        T(9));
  CHECK(A[slp::Slice{1, 3}, slp::Slice{1, 3}].T()[_, slp::Slice{1, _}].value(
            1) == T(9));
}

TEMPLATE_TEST_CASE("VariableMatrix - to_eigen()", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::VariableMatrix<T> A{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}};
  auto eigen_A = A.to_eigen();

  CHECK(A.rows() == eigen_A.rows());
  CHECK(A.cols() == eigen_A.cols());

  for (int row = 0; row < A.rows(); ++row) {
    for (int col = 0; col < A.cols(); ++col) {
      CHECK(A[row, col] == eigen_A(row, col));
    }
  }
}

TEMPLATE_TEST_CASE("VariableMatrix - cwise_transform()", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  // VariableMatrix cwise_transform
  slp::VariableMatrix<T> A{{T(-2), T(-3), T(-4)}, {T(-5), T(-6), T(-7)}};

  slp::VariableMatrix result1 = A.cwise_transform(slp::abs<T>);
  Eigen::Matrix<T, 2, 3> expected1{{T(2), T(3), T(4)}, {T(5), T(6), T(7)}};

  // Don't modify original matrix
  CHECK(A.value() == -expected1);

  CHECK(result1.value() == expected1);

  // VariableBlock cwise_transform
  auto sub_A = A.block(0, 0, 2, 2);

  slp::VariableMatrix result2 = sub_A.cwise_transform(slp::abs<T>);
  Eigen::Matrix<T, 2, 2> expected2{{T(2), T(3)}, {T(5), T(6)}};

  // Don't modify original matrix
  CHECK(A.value() == -expected1);
  CHECK(sub_A.value() == -expected2);

  CHECK(result2.value() == expected2);
}

TEMPLATE_TEST_CASE("VariableMatrix - identity() static function",
                   "[VariableMatrix]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  auto A = slp::VariableMatrix<T>::identity(3);

  CHECK(A.rows() == 3);
  CHECK(A.cols() == 3);

  CHECK(A.value() == Eigen::Matrix<T, 3, 3>::Identity());
}

TEMPLATE_TEST_CASE("VariableMatrix - zero() static function",
                   "[VariableMatrix]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  auto A = slp::VariableMatrix<T>::zero(2, 3);

  CHECK(A.rows() == 2);
  CHECK(A.cols() == 3);

  for (auto& elem : A) {
    CHECK(elem.value() == T(0));
  }
}

TEMPLATE_TEST_CASE("VariableMatrix - one() static function", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  auto A = slp::VariableMatrix<T>::one(2, 3);

  CHECK(A.rows() == 2);
  CHECK(A.cols() == 3);

  for (auto& elem : A) {
    CHECK(elem.value() == T(1));
  }
}

TEMPLATE_TEST_CASE("VariableMatrix - constant() static function",
                   "[VariableMatrix]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  auto A = slp::VariableMatrix<T>::constant(2, 3, T(2));

  CHECK(A.rows() == 2);
  CHECK(A.cols() == 3);

  for (auto& elem : A) {
    CHECK(elem.value() == T(2));
  }
}

TEMPLATE_TEST_CASE("VariableMatrix - cwise_reduce()", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::VariableMatrix<T> A{{T(2), T(3), T(4)}, {T(5), T(6), T(7)}};
  slp::VariableMatrix<T> B{{T(8), T(9), T(10)}, {T(11), T(12), T(13)}};
  slp::VariableMatrix result = slp::cwise_reduce<T>(A, B, std::multiplies<>{});

  Eigen::Matrix<T, 2, 3> expected{{T(16), T(27), T(40)}, {T(55), T(72), T(91)}};
  CHECK(result.value() == expected);
}

TEMPLATE_TEST_CASE("VariableMatrix - exp()", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;
  using namespace slp::slicing;
  using std::exp;

  // Scalar
  auto A1 = slp::VariableMatrix<T>{{T(4)}};
  CHECK_THAT(A1.exp().value(),
             MatrixWithinAbs(Eigen::Matrix<T, 1, 1>{exp(T(4))}, T(1e-13)));
  CHECK_THAT((A1[_, _].exp().value()),
             MatrixWithinAbs(Eigen::Matrix<T, 1, 1>{exp(T(4))}, T(1e-13)));

  auto A2 = slp::VariableMatrix<T>{{T(0), T(1)}, {T(0), T(-0.5)}};
  CHECK_THAT(A2.exp().value() * (-A2).exp().value(),
             MatrixWithinAbs(Eigen::Matrix<T, 2, 2>::Identity(), T(1e-15)));
  CHECK_THAT((A2[_, _].exp().value() * (-A2[_, _]).exp().value()),
             MatrixWithinAbs(Eigen::Matrix<T, 2, 2>::Identity(), T(1e-15)));

  auto A3 = slp::VariableMatrix<T>{{T(0), T(1)}, {T(0), T(10)}};
  CHECK_THAT(A3.exp().value() * (-A3).exp().value(),
             MatrixWithinAbs(Eigen::Matrix<T, 2, 2>::Identity(), T(1e-15)));
  CHECK_THAT((A3[_, _].exp().value() * (-A3[_, _]).exp().value()),
             MatrixWithinAbs(Eigen::Matrix<T, 2, 2>::Identity(), T(1e-15)));

  auto A4 = slp::VariableMatrix<T>{{T(1), T(10)}, {T(0), T(0)}};
  CHECK_THAT(A4.exp().value() * (-A4).exp().value(),
             MatrixWithinAbs(Eigen::Matrix<T, 2, 2>::Identity(), T(1e-15)));
  CHECK_THAT((A4[_, _].exp().value() * (-A4[_, _]).exp().value()),
             MatrixWithinAbs(Eigen::Matrix<T, 2, 2>::Identity(), T(1e-15)));

  auto A5 = slp::VariableMatrix<T>{{T(2), T(3)}, {T(4), T(5)}};
  CHECK_THAT(A5.exp().value() * (-A5).exp().value(),
             MatrixWithinAbs(Eigen::Matrix<T, 2, 2>::Identity(), T(1e-12)));
  CHECK_THAT((A5[_, _].exp().value() * (-A5[_, _]).exp().value()),
             MatrixWithinAbs(Eigen::Matrix<T, 2, 2>::Identity(), T(1e-12)));

  // Pascal matrix
  //
  //    ([0  0  0  0  0  0  0])   [1  0   0   0   0  0  0]
  //    ([1  0  0  0  0  0  0])   [1  1   0   0   0  0  0]
  //    ([0  2  0  0  0  0  0])   [1  2   1   0   0  0  0]
  // exp([0  0  3  0  0  0  0]) = [1  3   3   1   0  0  0]
  //    ([0  0  0  4  0  0  0])   [1  4   6   4   1  0  0]
  //    ([0  0  0  0  5  0  0])   [1  5  10  10   5  1  0]
  //    ([0  0  0  0  0  6  0])   [1  6  15  20  15  6  1]
  auto pascal = slp::VariableMatrix<T>::zero(7, 7);
  for (int col = 0; col < 6; ++col) {
    pascal[col + 1, col] = T(col + 1);
  }
  Eigen::Matrix<T, 7, 7> expected_pascal = Eigen::Matrix<T, 7, 7>::Zero();
  expected_pascal.col(0).setConstant(T(1));
  for (int col = 1; col < 7; ++col) {
    for (int row = col; row < 7; ++row) {
      expected_pascal(row, col) =
          expected_pascal(row - 1, col - 1) + expected_pascal(row - 1, col);
    }
  }
  CHECK_THAT(pascal.exp().value(), MatrixWithinAbs(expected_pascal, T(1e-14)));
  CHECK_THAT((pascal[_, _].exp().value()),
             MatrixWithinAbs(expected_pascal, T(1e-14)));
}

TEMPLATE_TEST_CASE("VariableMatrix - block() free function", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::VariableMatrix<T> A{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}};
  slp::VariableMatrix<T> B{{T(7)}, {T(8)}};

  slp::VariableMatrix mat1 = slp::block({{A, B}});
  Eigen::Matrix<T, 2, 4> expected1{{T(1), T(2), T(3), T(7)},
                                   {T(4), T(5), T(6), T(8)}};
  CHECK(mat1.rows() == 2);
  CHECK(mat1.cols() == 4);
  CHECK(mat1.value() == expected1);

  slp::VariableMatrix<T> C{{T(9), T(10), T(11), T(12)}};

  slp::VariableMatrix mat2 = slp::block({{A, B}, {C}});
  Eigen::Matrix<T, 3, 4> expected2{{T(1), T(2), T(3), T(7)},
                                   {T(4), T(5), T(6), T(8)},
                                   {T(9), T(10), T(11), T(12)}};
  CHECK(mat2.rows() == 3);
  CHECK(mat2.cols() == 4);
  CHECK(mat2.value() == expected2);
}

template <typename T>
void check_solve(slp::VariableMatrix<T> A, slp::VariableMatrix<T> B) {
  INFO(std::format("solve(A, B) {}x{}", A.rows(), A.cols()));

  auto X = solve(A, B);

  CHECK(X.rows() == A.cols());
  CHECK(X.cols() == B.cols());
  CHECK(T((A.value() * X.value() - B.value()).norm()) < T(1e-12));
}

TEMPLATE_TEST_CASE("VariableMatrix - solve() free function", "[VariableMatrix]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  // 1x1 special case
  check_solve(slp::VariableMatrix<T>{{T(2)}}, slp::VariableMatrix<T>{{T(5)}});

  // 2x2 special case
  check_solve(slp::VariableMatrix<T>{{T(1), T(2)}, {T(3), T(4)}},
              slp::VariableMatrix<T>{{T(5)}, {T(6)}});

  // 3x3 special case
  check_solve(
      slp::VariableMatrix<T>{
          {T(1), T(2), T(3)}, {T(-4), T(-5), T(6)}, {T(7), T(8), T(9)}},
      slp::VariableMatrix<T>{{T(10)}, {T(11)}, {T(12)}});

  // 4x4 special case
  check_solve(slp::VariableMatrix<T>{{T(1), T(2), T(3), T(-4)},
                                     {T(-5), T(6), T(7), T(8)},
                                     {T(9), T(10), T(11), T(12)},
                                     {T(13), T(14), T(15), T(16)}},
              slp::VariableMatrix<T>{{T(17)}, {T(18)}, {T(19)}, {T(20)}});

  // 5x5 general case
  check_solve(
      slp::VariableMatrix<T>{{T(1), T(2), T(3), T(-4), T(5)},
                             {T(-5), T(6), T(7), T(8), T(9)},
                             {T(9), T(10), T(11), T(12), T(13)},
                             {T(13), T(14), T(15), T(16), T(17)},
                             {T(17), T(18), T(19), T(20), T(21)}},
      slp::VariableMatrix<T>{{T(21)}, {T(22)}, {T(23)}, {T(24)}, {T(25)}});
}
