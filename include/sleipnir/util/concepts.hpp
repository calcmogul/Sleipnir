// Copyright (c) Sleipnir contributors

#pragma once

#include <complex>
#include <concepts>
#include <type_traits>

#include <Eigen/Core>

#include "sleipnir/autodiff/sleipnir_base.hpp"

namespace slp {

template <std::floating_point Scalar>
class Complex;

template <std::floating_point Scalar>
class ComplexMatrix;

template <typename Mat>
class ComplexSubmatrix;

namespace detail {

template <typename T>
inline constexpr bool is_complex = false;

template <typename T>
inline constexpr bool is_complex<std::complex<T>> = true;

template <std::floating_point Scalar>
inline constexpr bool is_complex<Complex<Scalar>> = true;

template <std::floating_point Scalar>
inline constexpr bool is_complex<ComplexMatrix<Scalar>> = true;

template <typename Mat>
inline constexpr bool is_complex<ComplexSubmatrix<Mat>> = true;

template <typename T>
  requires std::derived_from<T, Eigen::EigenBase<T>> &&
               (!std::is_void_v<typename T::Scalar>)
inline constexpr bool is_complex<T> =
    Eigen::NumTraits<typename T::Scalar>::IsComplex;

}  // namespace detail

/// A complex scalar or a matrix of complex entries.
///
/// The real-valued arithmetic and comparison operators exclude complex types.
/// Complex arithmetic lives in complex.hpp and complex_matrix.hpp.
template <typename T>
concept ComplexLike = detail::is_complex<std::decay_t<T>>;

template <typename T>
concept SleipnirType = std::derived_from<std::decay_t<T>, SleipnirBase>;

template <typename T>
concept MatrixLike = !ComplexLike<T> && requires(std::decay_t<T> t) {
  t.rows();
  t.cols();
};

template <typename T>
concept ScalarLike = !MatrixLike<T> && !ComplexLike<T> &&
                     std::constructible_from<std::decay_t<T>, int>;

template <typename T>
concept EigenMatrixLike =
    std::derived_from<std::decay_t<T>, Eigen::MatrixBase<std::decay_t<T>>> &&
    MatrixLike<T>;

template <typename T>
concept EigenSparseMatrixLike =
    std::derived_from<std::decay_t<T>,
                      Eigen::SparseMatrixBase<std::decay_t<T>>> &&
    MatrixLike<T>;

template <typename T, typename Scalar>
concept SleipnirMatrixLike =
    SleipnirType<T> && MatrixLike<T> &&
    std::same_as<typename std::decay_t<T>::Scalar, Scalar>;

template <typename T, typename Scalar>
concept SleipnirScalarLike =
    SleipnirType<T> && ScalarLike<T> &&
    std::same_as<typename std::decay_t<T>::Scalar, Scalar>;

template <typename Scalar>
class UnitVector;

namespace detail {

template <typename T>
inline constexpr bool is_unit_vector = false;

template <typename Scalar>
inline constexpr bool is_unit_vector<UnitVector<Scalar>> = true;

}  // namespace detail

/// A unit vector. Its equality constraints are on the sphere's tangent space,
/// so the generic elementwise equality operators exclude it. Elementwise
/// inequalities are componentwise bounds, which are still allowed.
template <typename T>
concept UnitVectorLike = detail::is_unit_vector<std::decay_t<T>>;

}  // namespace slp
