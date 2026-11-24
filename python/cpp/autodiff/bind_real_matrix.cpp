// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <stdint.h>

#include <concepts>
#include <format>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <sleipnir/autodiff/complex_matrix.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/real_submatrix.hpp>
#include <sleipnir/autodiff/slice.hpp>
#include <sleipnir/util/concepts.hpp>

#include "complex_scalar.hpp"
#include "docstrings.hpp"
#include "for_each_type.hpp"
#include "slice_tuple.hpp"
#include "try_cast.hpp"

namespace nb = nanobind;

namespace slp {

void bind_real_matrix(nb::module_& autodiff,
                      nb::class_<RealMatrix<double>>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<>(), DOC(slp, RealMatrix, RealMatrix));
  cls.def(nb::init<int>(), "rows"_a, DOC(slp, RealMatrix, RealMatrix, 2));
  cls.def(nb::init<int, int>(), "rows"_a, "cols"_a,
          DOC(slp, RealMatrix, RealMatrix, 3));
  cls.def(nb::init<const std::vector<std::vector<Real<double>>>&>(), "list"_a,
          DOC(slp, RealMatrix, RealMatrix, 5));
  cls.def(nb::init<const std::vector<std::vector<double>>&>(), "list"_a,
          DOC(slp, RealMatrix, RealMatrix, 6));
  cls.def(nb::init<nb::DRef<Eigen::MatrixXd>>(), "values"_a,
          DOC(slp, RealMatrix, RealMatrix, 8));
  cls.def(nb::init<nb::DRef<Eigen::MatrixXf>>(), "values"_a,
          DOC(slp, RealMatrix, RealMatrix, 8));
  cls.def(nb::init<const Real<double>&>(), "variable"_a,
          DOC(slp, RealMatrix, RealMatrix, 10));
  cls.def(nb::init<const RealSubmatrix<RealMatrix<double>>&>(), "values"_a,
          DOC(slp, RealMatrix, RealMatrix, 12));
  cls.def(
      "set_value",
      [](RealMatrix<double>& self, nb::DRef<Eigen::MatrixXd> values) {
        self.set_value(values);
      },
      "values"_a, DOC(slp, RealMatrix, set_value));
  cls.def(
      "set_value",
      [](RealMatrix<double>& self, nb::DRef<Eigen::MatrixXf> values) {
        self.set_value(values.cast<double>());
      },
      "values"_a, DOC(slp, RealMatrix, set_value));
  cls.def(
      "set_value",
      [](RealMatrix<double>& self, nb::DRef<Eigen::MatrixX<int64_t>> values) {
        self.set_value(values.cast<double>());
      },
      "values"_a, DOC(slp, RealMatrix, set_value));
  cls.def(
      "set_value",
      [](RealMatrix<double>& self, nb::DRef<Eigen::MatrixX<int32_t>> values) {
        self.set_value(values.cast<double>());
      },
      "values"_a, DOC(slp, RealMatrix, set_value));
  cls.def(
      "__setitem__",
      [](RealMatrix<double>& self, int row, const Real<double>& value) {
        if (row < 0) {
          row += self.size();
        }
        return self[row] = value;
      },
      "row"_a, "value"_a);
  cls.def(
      "__setitem__",
      [](RealMatrix<double>& self, SliceTuple slices, nb::object value) {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        auto [row_slice, row_slice_length] = to_slice(slices[0], self.rows());
        auto [col_slice, col_slice_length] = to_slice(slices[1], self.cols());

        auto lhs =
            self[row_slice, row_slice_length, col_slice, col_slice_length];
        if (auto rhs = try_cast<RealMatrix<double>>(value)) {
          lhs = rhs.value();
        } else if (auto rhs =
                       try_cast<RealSubmatrix<RealMatrix<double>>>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast_to_eigen<double>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast_to_eigen<float>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast_to_eigen<int64_t>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast_to_eigen<int32_t>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast<Real<double>>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast<double>(value)) {
          lhs = rhs.value();
        } else if (auto rhs = try_cast<int>(value)) {
          lhs = rhs.value();
        } else {
          throw nb::value_error(
              "RealMatrix.__setitem__ not implemented for value");
        }
      },
      "slices"_a, "value"_a);
  cls.def(
      "__getitem__",
      [](RealMatrix<double>& self, int row) -> Real<double>& {
        if (row < 0) {
          row += self.size();
        }
        return self[row];
      },
      nb::keep_alive<0, 1>(), "row"_a,
      DOC(slp, RealMatrix, operator, array, 3));
  cls.def(
      "__getitem__",
      [](RealMatrix<double>& self,
         std::tuple<int, int> slices) -> Real<double>& {
        auto [row, col] = slices;

        if (row >= self.rows() || col >= self.cols()) {
          throw std::out_of_range("Index out of bounds");
        }

        if (row < 0) {
          row += self.rows();
        }
        if (col < 0) {
          col += self.cols();
        }
        return self[row, col];
      },
      nb::rv_policy::reference_internal, "slices"_a,
      DOC(slp, RealMatrix, operator, array));
  cls.def(
      "__getitem__",
      [](RealMatrix<double>& self,
         SliceTuple slices) -> RealSubmatrix<RealMatrix<double>> {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        auto [row_slice, row_slice_length] = to_slice(slices[0], self.rows());
        auto [col_slice, col_slice_length] = to_slice(slices[1], self.cols());

        return self[row_slice, row_slice_length, col_slice, col_slice_length];
      },
      nb::keep_alive<0, 1>(), "slices"_a,
      DOC(slp, RealMatrix, operator, array));
  cls.def("row", nb::overload_cast<int>(&RealMatrix<double>::row), "row"_a,
          DOC(slp, RealMatrix, row));
  cls.def("col", nb::overload_cast<int>(&RealMatrix<double>::col), "col"_a,
          DOC(slp, RealMatrix, col));

  // RealMatrix-matrix multiplication. Products with a RealSubmatrix on the left
  // are bound on RealSubmatrix.
  for (const char* name : {"__mul__", "__matmul__"}) {
    cls.def(
        name,
        [](const RealMatrix<double>& lhs, const RealMatrix<double>& rhs) {
          return lhs * rhs;
        },
        nb::is_operator(), "rhs"_a);
    cls.def(
        name,
        [](const RealMatrix<double>& lhs,
           const RealSubmatrix<RealMatrix<double>>& rhs) { return lhs * rhs; },
        nb::is_operator(), "rhs"_a);
  }
  for (const char* name : {"__rmul__", "__rmatmul__"}) {
    cls.def(
        name,
        [](const RealMatrix<double>& rhs,
           const RealSubmatrix<RealMatrix<double>>& lhs) { return lhs * rhs; },
        nb::is_operator(), "lhs"_a);
  }

  // RealMatrix multiplication with NumPy arrays. These are also handled by
  // __array_ufunc__ below, but type checkers only see these overloads.
  cls.def(
      "__matmul__",
      [](const RealMatrix<double>& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rmatmul__",
      [](const RealMatrix<double>& rhs, nb::DRef<Eigen::MatrixXd> lhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "lhs"_a);

  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.def(
      "__array_ufunc__",
      [](RealMatrix<double>& self, nb::object ufunc, nb::str method,
         nb::args inputs, const nb::kwargs&) -> nb::object {
        std::string method_name = nb::cast<std::string>(method);
        std::string ufunc_name =
            nb::cast<std::string>(ufunc.attr("__repr__")());

        if (method_name == "__call__") {
          if (ufunc_name == "<ufunc 'matmul'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs * self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self * rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'multiply'>") {
            // NumPy complex scalars, which subclass Python's complex
            for (const auto& input : inputs) {
              if (PyComplex_Check(input.ptr())) {
                return nb::cast(nb::cast<ComplexScalar>(input).value * self);
              }
            }
          } else if (ufunc_name == "<ufunc 'add'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs + self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self + rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'subtract'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs - self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self - rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'equal'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs == self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self == rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'less'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs < self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self < rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'less_equal'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs <= self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self <= rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'greater'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs > self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self > rhs; }, inputs[1])) {
              return result.value();
            }
          } else if (ufunc_name == "<ufunc 'greater_equal'>") {
            if (auto result = apply_eigen_op(
                    [&](auto& lhs) { return lhs >= self; }, inputs[0])) {
              return result.value();
            } else if (auto result = apply_eigen_op(
                           [&](auto& rhs) { return self >= rhs; }, inputs[1])) {
              return result.value();
            }
          }
        }

        return nb::not_implemented();
      },
      // nanobind can't annotate nb::args or nb::kwargs, so the stub needs an
      // explicit signature
      nb::sig("def __array_ufunc__(self, ufunc: object, method: str, "
              "*inputs: object, **kwargs: object) -> object"),
      "ufunc"_a, "method"_a, "inputs"_a, "kwargs"_a);

  cls.def(nb::self + nb::self, "rhs"_a);
  cls.def(nb::self - nb::self, "rhs"_a);
  cls.def(-nb::self);

  // RealMatrix-scalar/scalar-matrix operations
  cls.def(nb::self * Real<double>(), "rhs"_a);
  cls.def(Real<double>() * nb::self, "lhs"_a);
  cls.def(nb::self / Real<double>(), "rhs"_a);
  cls.def(nb::self /= Real<double>(), "rhs"_a,
          DOC(slp, RealMatrix, operator, idiv));
  for_each_type<double, int, const Real<double>&,
                const RealSubmatrix<RealMatrix<double>>,
                nb::DRef<Eigen::MatrixXd>>([&]<typename T> {
    cls.def(
        "__add__",
        [](const RealMatrix<double>& lhs, T&& rhs) {
          if constexpr (ScalarLike<T>) {
            return lhs + Real<double>{rhs};
          } else {
            return lhs + rhs;
          }
        },
        nb::is_operator(), "rhs"_a);
    cls.def(
        "__radd__",
        [](const RealMatrix<double>& rhs, T&& lhs) {
          if constexpr (ScalarLike<T>) {
            return Real<double>{lhs} + rhs;
          } else {
            return lhs + rhs;
          }
        },
        nb::is_operator(), "lhs"_a);
    cls.def(
        "__sub__",
        [](const RealMatrix<double>& lhs, T&& rhs) {
          if constexpr (ScalarLike<T>) {
            return lhs - Real<double>{rhs};
          } else {
            return lhs - rhs;
          }
        },
        nb::is_operator(), "rhs"_a);
    cls.def(
        "__rsub__",
        [](const RealMatrix<double>& rhs, T&& lhs) {
          if constexpr (ScalarLike<T>) {
            return Real<double>{lhs} - rhs;
          } else {
            return lhs - rhs;
          }
        },
        nb::is_operator(), "lhs"_a);
  });

  cls.def(
      "__pow__",
      [](const RealMatrix<double>& self, int power) {
        return self.cwise_transform(
            [=](const auto& elem) { return pow(elem, power); });
      },
      nb::is_operator(), "power"_a);
  cls.def_prop_ro("T", &RealMatrix<double>::T, DOC(slp, RealMatrix, T));
  cls.def("trace", &RealMatrix<double>::trace, DOC(slp, RealMatrix, trace));
  cls.def("rows", &RealMatrix<double>::rows, DOC(slp, RealMatrix, rows));
  cls.def("cols", &RealMatrix<double>::cols, DOC(slp, RealMatrix, cols));
  cls.def_prop_ro("shape", [](const RealMatrix<double>& self) {
    return nb::make_tuple(self.rows(), self.cols());
  });
  cls.def("value", nb::overload_cast<int, int>(&RealMatrix<double>::value),
          "row"_a, "col"_a, DOC(slp, RealMatrix, value));
  cls.def("value", nb::overload_cast<int>(&RealMatrix<double>::value),
          "index"_a, DOC(slp, RealMatrix, value, 2));
  cls.def("value", nb::overload_cast<>(&RealMatrix<double>::value),
          DOC(slp, RealMatrix, value, 3));
  cls.def(
      "cwise_map",
      [](const RealMatrix<double>& self,
         const std::function<Real<double>(const Real<double>& x)>& unary_op) {
        return self.cwise_transform(unary_op);
      },
      "func"_a, DOC(slp, RealMatrix, cwise_transform));
  cls.def("exp", &RealMatrix<double>::exp, DOC(slp, RealMatrix, exp));
  cls.def_static("identity", &RealMatrix<double>::identity, "rows"_a,
                 DOC(slp, RealMatrix, identity));
  cls.def_static("zero", &RealMatrix<double>::zero, "rows"_a, "cols"_a,
                 DOC(slp, RealMatrix, zero));
  cls.def_static("one", &RealMatrix<double>::one, "rows"_a, "cols"_a,
                 DOC(slp, RealMatrix, one));
  cls.def_static("constant", &RealMatrix<double>::constant, "rows"_a, "cols"_a,
                 "constant"_a, DOC(slp, RealMatrix, constant));

  // Comparison operators
  for_each_type<nb::detail::self_t, double, int, Real<double>>([&]<typename T> {
    cls.def(nb::self == T(), "rhs"_a, DOC(slp, operator, eq, 7));
    cls.def(nb::self < T(), "rhs"_a, DOC(slp, operator, lt));
    cls.def(nb::self <= T(), "rhs"_a, DOC(slp, operator, le));
    cls.def(nb::self > T(), "rhs"_a, DOC(slp, operator, gt));
    cls.def(nb::self >= T(), "rhs"_a, DOC(slp, operator, ge));
    if constexpr (!std::same_as<nb::detail::self_t, T>) {
      cls.def(T() == nb::self, "rhs"_a, DOC(slp, operator, eq, 7));
      cls.def(T() < nb::self, "rhs"_a, DOC(slp, operator, lt));
      cls.def(T() <= nb::self, "rhs"_a, DOC(slp, operator, le));
      cls.def(T() > nb::self, "rhs"_a, DOC(slp, operator, gt));
      cls.def(T() >= nb::self, "rhs"_a, DOC(slp, operator, ge));
    }
  });
  for_each_type<const RealSubmatrix<RealMatrix<double>>&,
                nb::DRef<Eigen::MatrixXd>>([&]<typename T> {
    cls.def(
        "__eq__",
        [](const RealMatrix<double>& lhs, T&& rhs) { return lhs == rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, eq, 7));
    cls.def(
        "__lt__",
        [](const RealMatrix<double>& lhs, T&& rhs) { return lhs < rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, lt));
    cls.def(
        "__le__",
        [](const RealMatrix<double>& lhs, T&& rhs) { return lhs <= rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, le));
    cls.def(
        "__gt__",
        [](const RealMatrix<double>& lhs, T&& rhs) { return lhs > rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, gt));
    cls.def(
        "__ge__",
        [](const RealMatrix<double>& lhs, T&& rhs) { return lhs >= rhs; },
        nb::is_operator(), "rhs"_a, DOC(slp, operator, ge));
  });

  cls.def("__len__", &RealMatrix<double>::rows, DOC(slp, RealMatrix, rows));

  cls.def(
      "__iter__",
      [](const RealMatrix<double>& self) {
        return nb::make_iterator(nb::type<RealMatrix<double>>(),
                                 "value_iterator", self.begin(), self.end());
      },
      nb::keep_alive<0, 1>());

  autodiff.def(
      "cwise_reduce",
      [](const RealMatrix<double>& lhs, const RealMatrix<double>& rhs,
         const std::function<Real<double>(const Real<double>& x,
                                          const Real<double>& y)>& binary_op) {
        return cwise_reduce<double>(lhs, rhs, binary_op);
      },
      "lhs"_a, "rhs"_a, "func"_a, DOC(slp, cwise_reduce));

  autodiff.def(
      "block",
      nb::overload_cast<const std::vector<std::vector<RealMatrix<double>>>&>(
          &block<double>),
      "list"_a, DOC(slp, block));

  autodiff.def(
      "solve",
      nb::overload_cast<const RealMatrix<double>&, const RealMatrix<double>&>(
          &solve<double>),
      "A"_a, "B"_a, DOC(slp, solve));
  autodiff.def(
      "solve",
      [](const RealMatrix<double>& A, nb::DRef<Eigen::MatrixXd> B) {
        return solve(A, RealMatrix<double>(B));
      },
      "A"_a, "B"_a, DOC(slp, solve));
  autodiff.def(
      "solve",
      [](nb::DRef<Eigen::MatrixXd> A, const RealMatrix<double>& B) {
        return solve(RealMatrix<double>(A), B);
      },
      "A"_a, "B"_a, DOC(slp, solve));
  // Products with constant complex numbers
  cls.def(
      "__mul__",
      [](const RealMatrix<double>& lhs, ComplexScalar rhs) {
        return lhs * rhs.value;
      },
      nb::is_operator(), "rhs"_a, DOC(slp, operator, mul, 6));
  cls.def(
      "__rmul__",
      [](const RealMatrix<double>& rhs, ComplexScalar lhs) {
        return lhs.value * rhs;
      },
      nb::is_operator(), "lhs"_a, DOC(slp, operator, mul, 7));

  // RealMatrix multiplication with SciPy sparse matrices on the right. These
  // come after every other overload because the sparse caster converts scalars
  // to 1x1 matrices. SciPy sparse matrices on the left don't defer to reflected
  // operators.
  for (const char* name : {"__mul__", "__matmul__"}) {
    cls.def(
        name,
        [](const RealMatrix<double>& lhs,
           const Eigen::SparseMatrix<double>& rhs) { return lhs * rhs; },
        nb::is_operator(), "rhs"_a);
  }
}

}  // namespace slp
