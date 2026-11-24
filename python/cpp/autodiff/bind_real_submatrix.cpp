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

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/real_submatrix.hpp>
#include <sleipnir/autodiff/slice.hpp>
#include <sleipnir/util/concepts.hpp>

#include "docstrings.hpp"
#include "for_each_type.hpp"
#include "slice_tuple.hpp"
#include "try_cast.hpp"

namespace nb = nanobind;

namespace slp {

void bind_real_submatrix(nb::class_<RealSubmatrix<RealMatrix<double>>>& cls) {
  using namespace nb::literals;

  cls.def(
      "set_value",
      [](RealSubmatrix<RealMatrix<double>>& self, double value) {
        self.set_value(value);
      },
      "value"_a, DOC(slp, RealSubmatrix, set_value));
  cls.def(
      "set_value",
      [](RealSubmatrix<RealMatrix<double>>& self,
         nb::DRef<Eigen::MatrixXd> values) { self.set_value(values); },
      "values"_a, DOC(slp, RealSubmatrix, set_value, 2));
  for_each_type<Eigen::MatrixXf, Eigen::MatrixX<int64_t>,
                Eigen::MatrixX<int32_t>>([&]<typename T> {
    cls.def(
        "set_value",
        [](RealSubmatrix<RealMatrix<double>>& self, nb::DRef<T> values) {
          self.set_value(values.template cast<double>());
        },
        "values"_a, DOC(slp, RealSubmatrix, set_value, 2));
  });
  cls.def(
      "__setitem__",
      [](RealSubmatrix<RealMatrix<double>>& self, int row,
         const Real<double>& value) {
        if (row < 0) {
          row += self.size();
        }
        return self[row] = value;
      },
      "row"_a, "value"_a);
  cls.def(
      "__setitem__",
      [](RealSubmatrix<RealMatrix<double>>& self, SliceTuple slices,
         nb::object value) {
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
              "RealSubmatrix.__setitem__ not implemented for value");
        }
      },
      "slices"_a, "value"_a);
  cls.def(
      "__getitem__",
      [](RealSubmatrix<RealMatrix<double>>& self, int row) -> Real<double>& {
        if (row < 0) {
          row += self.size();
        }
        return self[row];
      },
      nb::keep_alive<0, 1>(), "row"_a,
      DOC(slp, RealSubmatrix, operator, array, 3));
  cls.def(
      "__getitem__",
      [](RealSubmatrix<RealMatrix<double>>& self,
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
      DOC(slp, RealSubmatrix, operator, array));
  cls.def(
      "__getitem__",
      [](RealSubmatrix<RealMatrix<double>>& self,
         SliceTuple slices) -> RealSubmatrix<RealMatrix<double>> {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        auto [row_slice, row_slice_length] = to_slice(slices[0], self.rows());
        auto [col_slice, col_slice_length] = to_slice(slices[1], self.cols());

        return self[row_slice, row_slice_length, col_slice, col_slice_length];
      },
      nb::keep_alive<0, 1>(), DOC(slp, RealSubmatrix, operator, array),
      "slices"_a);
  cls.def("row",
          nb::overload_cast<int>(&RealSubmatrix<RealMatrix<double>>::row),
          "row"_a, DOC(slp, RealSubmatrix, row));
  cls.def("col",
          nb::overload_cast<int>(&RealSubmatrix<RealMatrix<double>>::col),
          "col"_a, DOC(slp, RealSubmatrix, col));

  // RealMatrix-matrix multiplication. Products with a RealMatrix on the left
  // are bound on RealMatrix.
  for (const char* name : {"__mul__", "__matmul__"}) {
    cls.def(
        name,
        [](const RealSubmatrix<RealMatrix<double>>& lhs,
           const RealSubmatrix<RealMatrix<double>>& rhs) { return lhs * rhs; },
        nb::is_operator(), "rhs"_a);
  }

  // RealMatrix multiplication with NumPy arrays. These are also handled by
  // __array_ufunc__ below, but type checkers only see these overloads.
  cls.def(
      "__matmul__",
      [](const RealSubmatrix<RealMatrix<double>>& lhs,
         nb::DRef<Eigen::MatrixXd> rhs) { return lhs * rhs; },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rmatmul__",
      [](const RealSubmatrix<RealMatrix<double>>& rhs,
         nb::DRef<Eigen::MatrixXd> lhs) { return lhs * rhs; },
      nb::is_operator(), "lhs"_a);

  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.def(
      "__array_ufunc__",
      [](RealSubmatrix<RealMatrix<double>>& self, nb::object ufunc,
         nb::str method, nb::args inputs, const nb::kwargs&) -> nb::object {
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
          DOC(slp, RealSubmatrix, operator, idiv));
  for_each_type<double, int, const Real<double>&, nb::DRef<Eigen::MatrixXd>>(
      [&]<typename T> {
        cls.def(
            "__add__",
            [](const RealSubmatrix<RealMatrix<double>>& lhs, T&& rhs) {
              if constexpr (ScalarLike<T>) {
                return lhs + Real<double>{rhs};
              } else {
                return lhs + rhs;
              }
            },
            nb::is_operator(), "rhs"_a);
        cls.def(
            "__radd__",
            [](const RealSubmatrix<RealMatrix<double>>& rhs, T&& lhs) {
              if constexpr (ScalarLike<T>) {
                return Real<double>{lhs} + rhs;
              } else {
                return lhs + rhs;
              }
            },
            nb::is_operator(), "lhs"_a);
        cls.def(
            "__sub__",
            [](const RealSubmatrix<RealMatrix<double>>& lhs, T&& rhs) {
              if constexpr (ScalarLike<T>) {
                return lhs - Real<double>{rhs};
              } else {
                return lhs - rhs;
              }
            },
            nb::is_operator(), "rhs"_a);
        cls.def(
            "__rsub__",
            [](const RealSubmatrix<RealMatrix<double>>& rhs, T&& lhs) {
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
      [](const RealSubmatrix<RealMatrix<double>>& self, int power) {
        return self.cwise_transform(
            [=](const auto& elem) { return pow(elem, power); });
      },
      nb::is_operator(), "power"_a);
  cls.def_prop_ro("T", &RealSubmatrix<RealMatrix<double>>::T,
                  DOC(slp, RealSubmatrix, T));
  cls.def("trace", &RealSubmatrix<RealMatrix<double>>::trace,
          DOC(slp, RealSubmatrix, trace));
  cls.def("rows", &RealSubmatrix<RealMatrix<double>>::rows,
          DOC(slp, RealSubmatrix, rows));
  cls.def("cols", &RealSubmatrix<RealMatrix<double>>::cols,
          DOC(slp, RealSubmatrix, cols));
  cls.def_prop_ro("shape", [](const RealSubmatrix<RealMatrix<double>>& self) {
    return nb::make_tuple(self.rows(), self.cols());
  });
  cls.def(
      "value",
      nb::overload_cast<int, int>(&RealSubmatrix<RealMatrix<double>>::value),
      "row"_a, "col"_a, DOC(slp, RealSubmatrix, value));
  cls.def("value",
          nb::overload_cast<int>(&RealSubmatrix<RealMatrix<double>>::value),
          "index"_a, DOC(slp, RealSubmatrix, value, 2));
  cls.def("value",
          nb::overload_cast<>(&RealSubmatrix<RealMatrix<double>>::value),
          DOC(slp, RealSubmatrix, value, 3));
  cls.def(
      "cwise_map",
      [](const RealSubmatrix<RealMatrix<double>>& self,
         const std::function<Real<double>(const Real<double>& x)>& unary_op) {
        return self.cwise_transform(unary_op);
      },
      "func"_a, DOC(slp, RealSubmatrix, cwise_transform));
  cls.def("exp", &RealSubmatrix<RealMatrix<double>>::exp,
          DOC(slp, RealSubmatrix, exp));

  // Comparison operators. Like C++'s, which are templates, they don't apply
  // implicit conversions to matrix operands.
  for_each_type<nb::detail::self_t, double, int, Real<double>>([&]<typename T> {
    auto rhs = [] {
      if constexpr (std::same_as<nb::detail::self_t, T>) {
        return "rhs"_a.noconvert();
      } else {
        return "rhs"_a;
      }
    }();
    cls.def(nb::self == T(), rhs, DOC(slp, operator, eq, 7));
    cls.def(nb::self < T(), rhs, DOC(slp, operator, lt));
    cls.def(nb::self <= T(), rhs, DOC(slp, operator, le));
    cls.def(nb::self > T(), rhs, DOC(slp, operator, gt));
    cls.def(nb::self >= T(), rhs, DOC(slp, operator, ge));
    if constexpr (!std::same_as<nb::detail::self_t, T>) {
      cls.def(T() == nb::self, "rhs"_a, DOC(slp, operator, eq, 7));
      cls.def(T() < nb::self, "rhs"_a, DOC(slp, operator, lt));
      cls.def(T() <= nb::self, "rhs"_a, DOC(slp, operator, le));
      cls.def(T() > nb::self, "rhs"_a, DOC(slp, operator, gt));
      cls.def(T() >= nb::self, "rhs"_a, DOC(slp, operator, ge));
    }
  });
  for_each_type<const RealMatrix<double>&, nb::DRef<Eigen::MatrixXd>>(
      [&]<typename T> {
        auto rhs = [] {
          if constexpr (std::same_as<T, const RealMatrix<double>&>) {
            return "rhs"_a.noconvert();
          } else {
            return "rhs"_a;
          }
        }();
        cls.def(
            "__eq__",
            [](const RealSubmatrix<RealMatrix<double>>& lhs, T&& rhs) {
              return lhs == rhs;
            },
            nb::is_operator(), rhs, DOC(slp, operator, eq, 7));
        cls.def(
            "__lt__",
            [](const RealSubmatrix<RealMatrix<double>>& lhs, T&& rhs) {
              return lhs < rhs;
            },
            nb::is_operator(), rhs, DOC(slp, operator, lt));
        cls.def(
            "__le__",
            [](const RealSubmatrix<RealMatrix<double>>& lhs, T&& rhs) {
              return lhs <= rhs;
            },
            nb::is_operator(), rhs, DOC(slp, operator, le));
        cls.def(
            "__gt__",
            [](const RealSubmatrix<RealMatrix<double>>& lhs, T&& rhs) {
              return lhs > rhs;
            },
            nb::is_operator(), rhs, DOC(slp, operator, gt));
        cls.def(
            "__ge__",
            [](const RealSubmatrix<RealMatrix<double>>& lhs, T&& rhs) {
              return lhs >= rhs;
            },
            nb::is_operator(), rhs, DOC(slp, operator, ge));
      });

  cls.def("__len__", &RealSubmatrix<RealMatrix<double>>::rows,
          DOC(slp, RealSubmatrix, rows));

  cls.def(
      "__iter__",
      [](const RealSubmatrix<RealMatrix<double>>& self) {
        return nb::make_iterator(nb::type<RealSubmatrix<RealMatrix<double>>>(),
                                 "value_iterator", self.begin(), self.end());
      },
      nb::keep_alive<0, 1>());

  // RealMatrix multiplication with SciPy sparse matrices on the right. These
  // come after every other overload because the sparse caster converts scalars
  // to 1x1 matrices. SciPy sparse matrices on the left don't defer to reflected
  // operators.
  for (const char* name : {"__mul__", "__matmul__"}) {
    cls.def(
        name,
        [](const RealSubmatrix<RealMatrix<double>>& lhs,
           const Eigen::SparseMatrix<double>& rhs) { return lhs * rhs; },
        nb::is_operator(), "rhs"_a);
  }
}

}  // namespace slp
