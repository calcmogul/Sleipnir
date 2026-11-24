// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/tuple.h>

#include <complex>
#include <format>
#include <functional>
#include <stdexcept>
#include <tuple>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <sleipnir/autodiff/complex.hpp>
#include <sleipnir/autodiff/complex_matrix.hpp>
#include <sleipnir/autodiff/complex_submatrix.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>

#include "complex_submatrix_assign.hpp"
#include "docstrings.hpp"
#include "for_each_type.hpp"
#include "slice_tuple.hpp"

namespace nb = nanobind;

namespace slp {

void bind_complex_submatrix(
    nb::class_<ComplexSubmatrix<ComplexMatrix<double>>>& cls) {
  using namespace nb::literals;

  using C = Complex<double>;
  using M = ComplexMatrix<double>;
  using S = ComplexSubmatrix<M>;
  using V = Real<double>;

  using DenseReal = nb::DRef<Eigen::MatrixXd>;
  using DenseComplex = nb::DRef<Eigen::MatrixXcd>;
  using SparseReal = const Eigen::SparseMatrix<double>&;
  using SparseComplex = const Eigen::SparseMatrix<std::complex<double>>&;

  cls.def(nb::init<M&>(), "mat"_a, nb::keep_alive<1, 2>(),
          DOC(slp, ComplexSubmatrix, ComplexSubmatrix, 3));
  cls.def(nb::init<M&, int, int, int, int>(), "mat"_a, "row_offset"_a,
          "col_offset"_a, "block_rows"_a, "block_cols"_a,
          nb::keep_alive<1, 2>(),
          DOC(slp, ComplexSubmatrix, ComplexSubmatrix, 4));

  cls.def(
      "set_value",
      [](S& self, const std::complex<double>& value) { self.set_value(value); },
      "value"_a, DOC(slp, ComplexSubmatrix, set_value));
  cls.def(
      "set_value", [](S& self, DenseReal values) { self.set_value(values); },
      "values"_a, DOC(slp, ComplexSubmatrix, set_value, 2));
  cls.def(
      "set_value", [](S& self, DenseComplex values) { self.set_value(values); },
      "values"_a, DOC(slp, ComplexSubmatrix, set_value, 2));

  cls.def(
      "__getitem__",
      [](S& self, int index) -> C& {
        if (index < 0) {
          index += self.size();
        }
        if (index < 0 || index >= static_cast<int>(self.size())) {
          throw std::out_of_range("Index out of bounds");
        }
        return self[index];
      },
      nb::rv_policy::reference_internal, "index"_a,
      DOC(slp, ComplexSubmatrix, operator, array, 3));
  cls.def(
      "__getitem__",
      [](S& self, std::tuple<int, int> indices) -> C& {
        auto [row, col] = indices;
        if (row < 0) {
          row += self.rows();
        }
        if (col < 0) {
          col += self.cols();
        }
        if (row < 0 || row >= self.rows() || col < 0 || col >= self.cols()) {
          throw std::out_of_range("Index out of bounds");
        }
        return self[row, col];
      },
      nb::rv_policy::reference_internal, "indices"_a,
      DOC(slp, ComplexSubmatrix, operator, array));
  cls.def(
      "__getitem__",
      [](S& self, SliceTuple slices) -> S {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }
        auto [row_slice, row_slice_length] = to_slice(slices[0], self.rows());
        auto [col_slice, col_slice_length] = to_slice(slices[1], self.cols());
        return self[row_slice, row_slice_length, col_slice, col_slice_length];
      },
      nb::keep_alive<0, 1>(), "slices"_a,
      DOC(slp, ComplexSubmatrix, operator, array, 7));
  cls.def(
      "__setitem__",
      [](S& self, int index, const C& value) {
        if (index < 0) {
          index += self.size();
        }
        if (index < 0 || index >= static_cast<int>(self.size())) {
          throw std::out_of_range("Index out of bounds");
        }
        self[index] = value;
      },
      "index"_a, "value"_a);
  cls.def(
      "__setitem__",
      [](S& self, std::tuple<int, int> indices, const C& value) {
        auto [row, col] = indices;
        if (row < 0) {
          row += self.rows();
        }
        if (col < 0) {
          col += self.cols();
        }
        if (row < 0 || row >= self.rows() || col < 0 || col >= self.cols()) {
          throw std::out_of_range("Index out of bounds");
        }
        self[row, col] = value;
      },
      "indices"_a, "value"_a);
  cls.def(
      "__setitem__",
      [](S& self, SliceTuple slices, nb::object value) {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }
        auto [row_slice, row_slice_length] = to_slice(slices[0], self.rows());
        auto [col_slice, col_slice_length] = to_slice(slices[1], self.cols());
        assign(self[row_slice, row_slice_length, col_slice, col_slice_length],
               value);
      },
      "slices"_a, "value"_a);

  cls.def("block", nb::overload_cast<int, int, int, int>(&S::block),
          "row_offset"_a, "col_offset"_a, "block_rows"_a, "block_cols"_a,
          nb::keep_alive<0, 1>(), DOC(slp, ComplexSubmatrix, block));
  cls.def("segment", nb::overload_cast<int, int>(&S::segment), "offset"_a,
          "length"_a, nb::keep_alive<0, 1>(),
          DOC(slp, ComplexSubmatrix, segment));
  cls.def("row", nb::overload_cast<int>(&S::row), "row"_a,
          nb::keep_alive<0, 1>(), DOC(slp, ComplexSubmatrix, row));
  cls.def("col", nb::overload_cast<int>(&S::col), "col"_a,
          nb::keep_alive<0, 1>(), DOC(slp, ComplexSubmatrix, col));

  // Matrix multiplication
  for (const char* name : {"__mul__", "__matmul__"}) {
    for_each_type<const S&, const M&, const RealMatrix<double>&, DenseReal,
                  DenseComplex>([&]<typename T> {
      cls.def(
          name, [](const S& lhs, T rhs) { return lhs * rhs; },
          nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 5));
    });
  }
  for (const char* name : {"__rmul__", "__rmatmul__"}) {
    for_each_type<const M&, const RealMatrix<double>&, DenseReal, DenseComplex>(
        [&]<typename T> {
          cls.def(
              name, [](const S& rhs, T lhs) { return lhs * rhs; },
              nb::is_operator(), "lhs"_a, DOC(slp, operator, mul, 5));
        });
  }

  // Matrix-scalar operations
  for_each_type<const C&, std::complex<double>, const V&, double>(
      [&]<typename T> {
        cls.def(
            "__mul__", [](const S& lhs, T rhs) { return lhs * rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 6));
        cls.def(
            "__rmul__", [](const S& rhs, T lhs) { return lhs * rhs; },
            nb::is_operator(), "lhs"_a, DOC(slp, operator, mul, 7));
        cls.def(
            "__truediv__", [](const S& lhs, T rhs) { return lhs / rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, div, 5));
        cls.def(
            "__eq__", [](const S& lhs, T rhs) { return lhs == rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 5));
      });

  // Elementwise operations with matrices
  for_each_type<const S&, const M&, const RealMatrix<double>&, DenseReal,
                DenseComplex>([&]<typename T> {
    cls.def(
        "__add__", [](const S& lhs, T rhs) { return lhs + rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, add, 6));
    cls.def(
        "__sub__", [](const S& lhs, T rhs) { return lhs - rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, sub, 6));
    cls.def(
        "__eq__", [](const S& lhs, T rhs) { return lhs == rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 4));
    cls.def(
        "cwise_product",
        [](const S& self, T rhs) { return self.cwise_product(rhs); }, "rhs"_a,
        DOC(slp, ComplexSubmatrix, cwise_product));
  });
  for_each_type<const M&, const RealMatrix<double>&, DenseReal, DenseComplex>(
      [&]<typename T> {
        cls.def(
            "__radd__", [](const S& rhs, T lhs) { return lhs + rhs; },
            nb::is_operator(), "lhs"_a, DOC(slp, operator, add, 6));
        cls.def(
            "__rsub__", [](const S& rhs, T lhs) { return lhs - rhs; },
            nb::is_operator(), "lhs"_a, DOC(slp, operator, sub, 6));
      });

  cls.def(
      "__neg__", [](const S& self) { return -M{self}; },
      DOC(slp, operator, sub, 7));
  cls.def(
      "__pos__", [](const S& self) { return M{self}; },
      DOC(slp, operator, add, 7));

  // NumPy arrays on the left defer to the reflected operators above
  cls.attr("__array_ufunc__") = nb::none();

  cls.def_prop_ro("T", &S::T, DOC(slp, ComplexSubmatrix, T));
  cls.def_prop_ro("H", &S::H, DOC(slp, ComplexSubmatrix, H));
  cls.def("conj", &S::conj, DOC(slp, ComplexSubmatrix, conj));
  cls.def("real", &S::real, DOC(slp, ComplexSubmatrix, real));
  cls.def("imag", &S::imag, DOC(slp, ComplexSubmatrix, imag));
  cls.def("abs2", &S::abs2, DOC(slp, ComplexSubmatrix, abs2));
  cls.def("trace", &S::trace, DOC(slp, ComplexSubmatrix, trace));
  cls.def(
      "cwise_map",
      [](const S& self, const std::function<C(const C& z)>& unary_op) {
        return self.cwise_transform(unary_op);
      },
      "func"_a, DOC(slp, ComplexSubmatrix, cwise_transform));
  cls.def("rows", &S::rows, DOC(slp, ComplexSubmatrix, rows));
  cls.def("cols", &S::cols, DOC(slp, ComplexSubmatrix, cols));
  cls.def_prop_ro("shape", [](const S& self) {
    return nb::make_tuple(self.rows(), self.cols());
  });
  cls.def("value", nb::overload_cast<int, int>(&S::value), "row"_a, "col"_a,
          DOC(slp, ComplexSubmatrix, value));
  cls.def("value", nb::overload_cast<int>(&S::value), "index"_a,
          DOC(slp, ComplexSubmatrix, value, 2));
  cls.def("value", nb::overload_cast<>(&S::value),
          DOC(slp, ComplexSubmatrix, value, 3));

  cls.def("__len__", &S::rows, DOC(slp, ComplexSubmatrix, rows));
  cls.def(
      "__iter__",
      [](S& self) {
        return nb::make_iterator(nb::type<S>(), "value_iterator", self.begin(),
                                 self.end());
      },
      nb::keep_alive<0, 1>());

  // Matrix multiplication with SciPy sparse matrices on the right. These come
  // after every other overload because the sparse caster converts scalars to
  // 1x1 matrices. SciPy sparse matrices on the left don't defer to reflected
  // operators.
  for (const char* name : {"__mul__", "__matmul__"}) {
    for_each_type<SparseReal, SparseComplex>([&]<typename T> {
      cls.def(
          name, [](const S& lhs, T rhs) { return lhs * rhs; },
          nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 5));
    });
  }
}

}  // namespace slp
