// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <optional>
#include <utility>

#include <Eigen/Core>
#include <emscripten/bind.h>
#include <sleipnir/util/assert.hpp>

namespace em = emscripten;

namespace slp {

/// Converts the given em::object to a C++ type.
template <typename T>
std::optional<T> try_cast(const em::object& obj) {
  if (em::isinstance<T>(obj)) {
    return em::cast<T>(obj);
  } else {
    return std::nullopt;
  }
}

/// Converts the given em::ndarray to an Eigen matrix.
///
/// @tparam Scalar The Eigen matrix's scalar type.
/// @param obj The em::ndarray.
template <typename Scalar>
std::optional<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
try_cast_to_eigen(const em::object& obj) {
  if (em::isinstance<em::ndarray<Scalar>>(obj)) {
    auto arr = em::cast<em::ndarray<Scalar>>(obj);
    slp_assert(arr.ndim() == 2);

    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      alignof(Scalar), Stride>{
        arr.data(), static_cast<Eigen::Index>(arr.shape(0)),
        static_cast<Eigen::Index>(arr.shape(1)),
        Stride{arr.stride(0), arr.stride(1)}};
  } else {
    return std::nullopt;
  }
}

namespace detail {

/// Converts the given em::ndarray to an Eigen type with the first scalar type
/// that works, then calls a function on it and returns the result.
///
/// @tparam Scalar Scalar type to try.
/// @tparam Scalars Rest of Scalar types to try.
/// @tparam F Type of function to apply.
/// @param f Function to apply.
/// @param obj The em::ndarray.
template <typename Scalar, typename... Scalars, typename F>
std::optional<em::object> apply_eigen_op(F&& f, const em::object& obj) {
  if (auto mat = try_cast_to_eigen<Scalar>(obj)) {
    return em::cast(f(mat.value()));
  } else if constexpr (sizeof...(Scalars) > 0) {
    return apply_eigen_op<Scalars...>(f, obj);
  } else {
    return std::nullopt;
  }
}

}  // namespace detail

/// Converts the given em::ndarray to an Eigen type, then calls a function on it
/// and returns the result.
///
/// @tparam F Type of function to apply.
/// @param f Function to apply.
/// @param obj The em::ndarray.
template <typename F>
std::optional<em::object> apply_eigen_op(F&& f, const em::object& obj) {
  return detail::apply_eigen_op<double, float, int64_t, int32_t>(
      std::forward<F>(f), obj);
}

}  // namespace slp
