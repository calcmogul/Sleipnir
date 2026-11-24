// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <format>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <sleipnir/autodiff/complex.hpp>
#include <sleipnir/autodiff/complex_matrix.hpp>
#include <sleipnir/autodiff/complex_submatrix.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/slice.hpp>

#include "complex_submatrix_assign.hpp"
#include "docstrings.hpp"
#include "for_each_type.hpp"
#include "slice_tuple.hpp"
#include "try_cast.hpp"

namespace nb = nanobind;

namespace slp {

void bind_complex_matrix(nb::class_<ComplexMatrix<double>>& cls) {
  using namespace nb::literals;

  using C = Complex<double>;
  using M = ComplexMatrix<double>;
  using V = Real<double>;

  // RealMatrix operands other than ComplexMatrix itself
  using DenseReal = nb::DRef<Eigen::MatrixXd>;
  using DenseComplex = nb::DRef<Eigen::MatrixXcd>;
  using SparseReal = const Eigen::SparseMatrix<double>&;
  using SparseComplex = const Eigen::SparseMatrix<std::complex<double>>&;

  cls.def(nb::init<>(), DOC(slp, ComplexMatrix, ComplexMatrix));
  cls.def(nb::init<int>(), "rows"_a, DOC(slp, ComplexMatrix, ComplexMatrix, 2));
  cls.def(nb::init<int, int>(), "rows"_a, "cols"_a,
          DOC(slp, ComplexMatrix, ComplexMatrix, 3));
  cls.def(nb::init<const std::vector<std::vector<C>>&>(), "list"_a,
          DOC(slp, ComplexMatrix, ComplexMatrix, 6));
  cls.def(nb::init<const RealMatrix<double>&, const RealMatrix<double>&>(),
          "real"_a, "imag"_a, DOC(slp, ComplexMatrix, ComplexMatrix, 7));
  cls.def(nb::init<const RealMatrix<double>&>(), "values"_a,
          DOC(slp, ComplexMatrix, ComplexMatrix, 8));
  cls.def(nb::init<DenseReal>(), "values"_a,
          DOC(slp, ComplexMatrix, ComplexMatrix, 8));
  cls.def(nb::init<DenseComplex>(), "values"_a,
          DOC(slp, ComplexMatrix, ComplexMatrix, 8));
  cls.def(nb::init<C>(), "value"_a, DOC(slp, ComplexMatrix, ComplexMatrix, 9));

  cls.def(
      "set_value", [](M& self, DenseReal values) { self.set_value(values); },
      "values"_a, DOC(slp, ComplexMatrix, set_value));
  cls.def(
      "set_value", [](M& self, DenseComplex values) { self.set_value(values); },
      "values"_a, DOC(slp, ComplexMatrix, set_value));

  cls.def(
      "__getitem__",
      [](M& self, int index) -> C& {
        if (index < 0) {
          index += self.size();
        }
        if (index < 0 || index >= static_cast<int>(self.size())) {
          throw std::out_of_range("Index out of bounds");
        }
        return self[index];
      },
      nb::rv_policy::reference_internal, "index"_a,
      DOC(slp, ComplexMatrix, operator, array, 3));
  cls.def(
      "__getitem__",
      [](M& self, std::tuple<int, int> indices) -> C& {
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
      DOC(slp, ComplexMatrix, operator, array));
  cls.def(
      "__getitem__",
      [](M& self, SliceTuple slices) -> ComplexSubmatrix<M> {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }
        auto [row_slice, row_slice_length] = to_slice(slices[0], self.rows());
        auto [col_slice, col_slice_length] = to_slice(slices[1], self.cols());
        return self[row_slice, row_slice_length, col_slice, col_slice_length];
      },
      nb::keep_alive<0, 1>(), "slices"_a,
      DOC(slp, ComplexMatrix, operator, array, 7));
  cls.def(
      "__setitem__",
      [](M& self, int index, const C& value) {
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
      [](M& self, std::tuple<int, int> indices, const C& value) {
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
      [](M& self, SliceTuple slices, nb::object value) {
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

  cls.def("block", nb::overload_cast<int, int, int, int>(&M::block),
          "row_offset"_a, "col_offset"_a, "block_rows"_a, "block_cols"_a,
          nb::keep_alive<0, 1>(), DOC(slp, ComplexMatrix, block));
  cls.def("segment", nb::overload_cast<int, int>(&M::segment), "offset"_a,
          "length"_a, nb::keep_alive<0, 1>(), DOC(slp, ComplexMatrix, segment));
  cls.def("row", nb::overload_cast<int>(&M::row), "row"_a,
          nb::keep_alive<0, 1>(), DOC(slp, ComplexMatrix, row));
  cls.def("col", nb::overload_cast<int>(&M::col), "col"_a,
          nb::keep_alive<0, 1>(), DOC(slp, ComplexMatrix, col));

  // RealMatrix multiplication
  for (const char* name : {"__mul__", "__matmul__"}) {
    for_each_type<const M&, const RealMatrix<double>&, DenseReal, DenseComplex>(
        [&]<typename T> {
          cls.def(
              name, [](const M& lhs, T rhs) { return lhs * rhs; },
              nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 5));
        });
  }
  for (const char* name : {"__rmul__", "__rmatmul__"}) {
    for_each_type<const RealMatrix<double>&, DenseReal, DenseComplex>(
        [&]<typename T> {
          cls.def(
              name, [](const M& rhs, T lhs) { return lhs * rhs; },
              nb::is_operator(), "lhs"_a, DOC(slp, operator, mul, 5));
        });
  }

  // RealMatrix-scalar operations
  for_each_type<const C&, std::complex<double>, const V&, double>(
      [&]<typename T> {
        cls.def(
            "__mul__", [](const M& lhs, T rhs) { return lhs * rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 6));
        cls.def(
            "__rmul__", [](const M& rhs, T lhs) { return lhs * rhs; },
            nb::is_operator(), "lhs"_a, DOC(slp, operator, mul, 7));
        cls.def(
            "__truediv__", [](const M& lhs, T rhs) { return lhs / rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, div, 5));
        cls.def(
            "__eq__", [](const M& lhs, T rhs) { return lhs == rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 5));
      });

  // Elementwise operations with matrices
  for_each_type<const M&, const RealMatrix<double>&, DenseReal, DenseComplex>(
      [&]<typename T> {
        cls.def(
            "__add__", [](const M& lhs, T rhs) { return lhs + rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, add, 6));
        cls.def(
            "__sub__", [](const M& lhs, T rhs) { return lhs - rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, sub, 6));
        cls.def(
            "__eq__", [](const M& lhs, T rhs) { return lhs == rhs; },
            nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 4));
        cls.def(
            "cwise_product",
            [](const M& self, T rhs) { return self.cwise_product(rhs); },
            "rhs"_a, DOC(slp, ComplexMatrix, cwise_product));
      });
  for_each_type<const RealMatrix<double>&, DenseReal, DenseComplex>(
      [&]<typename T> {
        cls.def(
            "__radd__", [](const M& rhs, T lhs) { return lhs + rhs; },
            nb::is_operator(), "lhs"_a, DOC(slp, operator, add, 6));
        cls.def(
            "__rsub__", [](const M& rhs, T lhs) { return lhs - rhs; },
            nb::is_operator(), "lhs"_a, DOC(slp, operator, sub, 6));
      });

  cls.def(
      "__neg__", [](const M& self) { return -self; },
      DOC(slp, operator, sub, 7));
  cls.def(
      "__pos__", [](const M& self) { return +self; },
      DOC(slp, operator, add, 7));

  // NumPy arrays on the left defer to the reflected operators above
  cls.attr("__array_ufunc__") = nb::none();

  cls.def_prop_ro("T", &M::T, DOC(slp, ComplexMatrix, T));
  cls.def_prop_ro("H", &M::H, DOC(slp, ComplexMatrix, H));
  cls.def("conj", &M::conj, DOC(slp, ComplexMatrix, conj));
  cls.def("real", &M::real, DOC(slp, ComplexMatrix, real));
  cls.def("imag", &M::imag, DOC(slp, ComplexMatrix, imag));
  cls.def("abs2", &M::abs2, DOC(slp, ComplexMatrix, abs2));
  cls.def("trace", &M::trace, DOC(slp, ComplexMatrix, trace));
  cls.def(
      "cwise_map",
      [](const M& self, const std::function<C(const C& z)>& unary_op) {
        return self.cwise_transform(unary_op);
      },
      "func"_a, DOC(slp, ComplexMatrix, cwise_transform));
  cls.def("rows", &M::rows, DOC(slp, ComplexMatrix, rows));
  cls.def("cols", &M::cols, DOC(slp, ComplexMatrix, cols));
  cls.def_prop_ro("shape", [](const M& self) {
    return nb::make_tuple(self.rows(), self.cols());
  });
  cls.def("value", nb::overload_cast<int, int>(&M::value), "row"_a, "col"_a,
          DOC(slp, ComplexMatrix, value));
  cls.def("value", nb::overload_cast<int>(&M::value), "index"_a,
          DOC(slp, ComplexMatrix, value, 2));
  cls.def("value", nb::overload_cast<>(&M::value),
          DOC(slp, ComplexMatrix, value, 3));
  cls.def_static("identity", &M::identity, "rows"_a,
                 DOC(slp, ComplexMatrix, identity));
  cls.def_static("zero", &M::zero, "rows"_a, "cols"_a,
                 DOC(slp, ComplexMatrix, zero));

  cls.def("__len__", &M::rows, DOC(slp, ComplexMatrix, rows));
  cls.def(
      "__iter__",
      [](const M& self) {
        return nb::make_iterator(nb::type<M>(), "value_iterator", self.begin(),
                                 self.end());
      },
      nb::keep_alive<0, 1>());

  // RealMatrix multiplication with SciPy sparse matrices on the right. These
  // come after every other overload because the sparse caster converts scalars
  // to 1x1 matrices. SciPy sparse matrices on the left don't defer to reflected
  // operators.
  for (const char* name : {"__mul__", "__matmul__"}) {
    for_each_type<SparseReal, SparseComplex>([&]<typename T> {
      cls.def(
          name, [](const M& lhs, T rhs) { return lhs * rhs; },
          nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 5));
    });
  }
}

}  // namespace slp
